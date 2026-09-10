"""CPU contract checks for isolated Gridworld DPT/IDT and comparison plots."""

import importlib.util
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest

import h5py
import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gridworld'))

from baseline_dataset import DPTDataset, IDTDataset, baseline_collate, optimal_labels, selected_group_ids, task_for_group
from env import make_env
from model import AD, RAD, DPT, IDT
from model.baseline_common import run_transformer
from model.gpt2 import GPT2Transformer
from utils import ad_collate_fn, get_traj_file_name

spec = importlib.util.spec_from_file_location('baseline_curves', ROOT / 'gridworld/scripts/evaluate_ad_rad_curves.py')
curves = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = curves
spec.loader.exec_module(curves)


def config(env='darkroom', model='DPT', **updates):
    result = dict(env=env, model=model, device='cpu', grid_size=2, dim_states=2,
                  num_actions=5, horizon=4, env_split_seed=0, train_env_ratio=0.5,
                  alg='PPO', alg_seed=0, n_transit=4, low_per_high=2, dim_z=3,
                  tf_n_embd=8, tf_n_head=2, tf_n_layer=1, tf_dim_feedforward=16,
                  tf_dropout=0., mixed_precision='no', label_smoothing=0.,
                  dynamics=False, n_compress_tokens=3, short_memory_keep=1,
                  compress_n_heads=2, compress_n_layers=1)
    result.update(updates)
    return result


def env_args(cfg, group):
    task = task_for_group(cfg, group)
    return {'goal': task} if cfg['env'] == 'darkroom' else {'key': task[:2], 'goal': task[2:]}


def write_history(directory, cfg):
    path = Path(directory) / f'{get_traj_file_name(cfg)}.hdf5'
    labels_by_group = {}
    with h5py.File(path, 'w') as file:
        for group_id in range(cfg['grid_size'] ** (2 if cfg['env'] == 'darkroom' else 4)):
            env = make_env(cfg, **env_args(cfg, group_id))()
            states, actions, rewards, next_states, dones, labels = [], [], [], [], [], []
            state, _ = env.reset()
            for step in range(3 * cfg['horizon']):
                oracle = env.get_optimal_action(state, env.have_key) if cfg['env'] == 'dktd' else env.get_optimal_action(state)
                action = 4 if step < cfg['horizon'] else oracle
                states.append(state.copy())
                labels.append(oracle)
                state, reward, done, _, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                if done:
                    state, _ = env.reset()
                next_states.append(state.copy())  # Existing collector's terminal reset behavior.
            group = file.create_group(str(group_id))
            for name, values in dict(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones).items():
                group[name] = np.asarray(values)[:, None]
            labels_by_group[group_id] = np.asarray(labels)
            env.close()
    return path, labels_by_group


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle_labels_match_real_env_including_dktd_key_at_reset(self):
        for env in ('darkroom', 'dktd'):
            cfg = config(env)
            with tempfile.TemporaryDirectory() as directory:
                path, labels = write_history(directory, cfg)
                with h5py.File(path, 'r') as file:
                    for group, expected in labels.items():
                        values = [file[str(group)][key][()].swapaxes(0, 1) for key in ('states', 'actions', 'rewards')]
                        np.testing.assert_array_equal(optimal_labels(cfg, group, *values)[0], expected)

    def test_datasets_collation_and_no_history_mutation(self):
        for env in ('darkroom', 'dktd'):
            cfg = config(env)
            with tempfile.TemporaryDirectory() as directory:
                path, _ = write_history(directory, cfg)
                original = path.read_bytes()
                for model, cls in (('DPT', DPTDataset), ('IDT', IDTDataset)):
                    cfg['model'] = model
                    dataset = cls(cfg, directory, 'all', 1, 12)
                    batch = baseline_collate([dataset[0], dataset[1]])
                    self.assertEqual(batch['actions'].dtype, torch.int64)
                    self.assertEqual(batch['states'].shape[1], 3 if model == 'DPT' else 4)
                    if model == 'DPT':
                        self.assertEqual(batch['target_actions'].shape, (2,))
                    else:
                        starts = dataset.return_to_go[:, ::cfg['horizon']]
                        np.testing.assert_array_equal(starts, np.repeat(starts[:, :1], starts.shape[1], 1))
                        returns = dataset.rewards.reshape(len(dataset.rewards), -1, cfg['horizon']).sum(-1)
                        self.assertTrue(np.all(np.diff(returns, axis=1) >= 0))
                self.assertEqual(path.read_bytes(), original)
                # AD still gets three-token S/A/R batches with one-hot actions.
                legacy = ad_collate_fn([dataset[0]], cfg['grid_size'], cfg['num_actions'])
                self.assertEqual(legacy['actions'].shape, (1, 4, 5))

    def test_forward_backward_all_backbones_and_dynamics(self):
        for env in ('darkroom', 'dktd'):
            for cls, data_cls in ((DPT, DPTDataset), (IDT, IDTDataset)):
                with self.subTest(env=env, model=cls.__name__):
                    cfg = config(env, cls.__name__, dynamics=True)
                    with tempfile.TemporaryDirectory() as directory:
                        write_history(directory, cfg)
                        data = data_cls(cfg, directory, 'all', 1, 12)
                        batch = baseline_collate([data[0], data[1]])
                        model = cls(cfg)
                        output = model(batch)
                        loss = sum(v for k, v in output.items() if k.startswith('loss_'))
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        for name, module in model.named_modules():
                            if isinstance(module, GPT2Transformer):
                                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), name)
                        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_dpt_prefix_causality_and_query_placement(self):
        model = DPT(config()).eval()
        query = torch.tensor([[1., 0.]])
        states = torch.zeros(1, 3, 2)
        actions = torch.zeros(1, 3, dtype=torch.long)
        rewards = torch.zeros(1, 3)
        tokens = model.build_tokens(query, states, actions, rewards, states)
        expected_query = model.embed_context(torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]))
        torch.testing.assert_close(tokens[:, :1], expected_query)
        original = run_transformer(model.transformer, tokens)
        changed = tokens.clone()
        changed[:, -1] += 100 * torch.randn_like(changed[:, -1])
        torch.testing.assert_close(original[:, :-1], run_transformer(model.transformer, changed)[:, :-1])

    def test_idt_current_review_and_action_cannot_leak(self):
        model = IDT(config(model='IDT')).eval()
        states = torch.zeros(1, 2, dtype=torch.long)
        rtg = torch.ones(1, 2)
        reviews = torch.randn(1, 2, 6)
        before = model.h_decision_transformer(rtg, states, reviews)
        reviews[:, 1] += 100
        torch.testing.assert_close(before, model.h_decision_transformer(rtg, states, reviews))
        z = torch.randn(1, 2, 2, 3)
        states = torch.zeros(1, 2, 2, dtype=torch.long)
        actions = torch.zeros_like(states)
        before = model.decisions_to_go(z, states, actions, actions)['action']
        after = model.decisions_to_go(z, states, actions + 1, actions)['action']
        torch.testing.assert_close(before[:, :, 0], after[:, :, 0])

    def test_rollout_reset_query_and_exact_partial_horizon(self):
        for env in ('darkroom', 'dktd'):
            for cls in (DPT, IDT):
                for n_transit in (2, 4):
                    cfg = config(env, cls.__name__, n_transit=n_transit)
                    vec = DummyVecEnv([make_env(cfg, **env_args(cfg, 0))])
                    try:
                        result = cls(cfg).eval().evaluate_in_context(vec, 11, sample=False)
                        self.assertEqual(result['reward_episode'].shape, (1, 2))
                        self.assertEqual(vec.get_attr('current_step'), [3])
                    finally:
                        vec.close()

    def test_strict_checkpoint_roundtrip_all_four_models(self):
        for cls in (AD, RAD, DPT, IDT):
            cfg = config(model=cls.__name__)
            first, second = cls(cfg), cls(cfg)
            second.load_state_dict(first.state_dict(), strict=True)
            if cls in (AD, RAD):
                self.assertEqual(first.max_seq_length, 12)
                self.assertEqual(first.type_embedding.shape, (1, 1, 3, 8))
            for name in first.state_dict():
                torch.testing.assert_close(first.state_dict()[name], second.state_dict()[name])

    def test_legacy_head_ad_rad_checkpoints_load_without_parameter_changes(self):
        for cls, filename in ((AD, 'ad.py'), (RAD, 'compressed_ad.py')):
            result = subprocess.run(['git', 'show', f'HEAD:gridworld/model/{filename}'],
                                    cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                self.skipTest('Legacy source comparison requires a Git checkout')
            module = types.ModuleType('model._legacy_checkpoint_probe')
            module.__package__ = 'model'
            exec(compile(result.stdout, filename, 'exec'), module.__dict__)
            cfg = config(model=cls.__name__)
            torch.manual_seed(42)
            legacy = getattr(module, cls.__name__)(cfg)
            torch.manual_seed(42)
            current = cls(cfg)
            self.assertEqual(set(legacy.state_dict()), set(current.state_dict()))
            for key, value in legacy.state_dict().items():
                torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)
            current.load_state_dict(legacy.state_dict(), strict=True)

    def test_terminal_transition_and_reset_query_are_distinct(self):
        cfg = config()
        for cls in (DPT, IDT):
            model = cls(cfg).eval()
            head = model.pred_actions if cls is DPT else model.decisions_to_go.pred_action
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
                head.bias[3] = 10  # Move away from reset state, ending at (1, 0).
            captured, reviewed = [], []
            if cls is DPT:
                model.embed_context.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
            else:
                model.decisions_to_go.embed_state.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
                model.reviewing_decisions.register_forward_pre_hook(lambda _, args: reviewed.append(args[3].clone()))
            vec = DummyVecEnv([make_env(cfg, **env_args(cfg, 0))])
            try:
                model.evaluate_in_context(vec, 5, sample=False)
            finally:
                vec.close()
            if cls is DPT:
                torch.testing.assert_close(captured[7][0, -2:], torch.tensor([1., 0.]))
                torch.testing.assert_close(captured[8][0, :2], torch.tensor([1., 1.]))
            else:
                self.assertEqual(reviewed[1][0, 0, -1], 2)
                self.assertEqual(captured[4][0], 3)

    def test_comparison_discovery_source_data_and_all_five_curves(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_history(root, cfg)
            for cls in (AD, RAD, DPT, IDT):
                local = {**cfg, 'model': cls.__name__}
                run = root / f'{cls.__name__}-darkroom-seed0'
                run.mkdir()
                target = run / ('best-model.pt' if cls is RAD else 'ckpt-2.pt')
                torch.save(dict(model=cls(local).state_dict(), config=local, step=2), target)
            specs = curves.discover_checkpoints(root, ['darkroom'], list(curves.METHOD_LABELS), None)
            self.assertEqual({s.method for s in specs}, {'AD', 'RAD', 'DPT', 'IDT'})
            rows = []
            for item in specs:
                rows.extend(curves.evaluate_checkpoint(item, [0], 2, None, torch.device('cpu'), root / 'out', False, False))
            rows += curves.source_history_rows(specs, {}, root, root / 'out', 2)
            self.assertEqual(sum(row['method'] == 'SOURCE' for row in rows), 1)
            self.assertEqual(rows[-1]['source_algorithm'], 'PPO')
            source = np.load(rows[-1]['cache_path'])
            np.testing.assert_array_equal(source['group_ids'], [2, 3])
            source.close()
            figures = curves.plot_environment('darkroom', rows, root / 'out', 1, ['png'], 50)
            self.assertTrue(figures[0].is_file())
            for method in curves.METHOD_LABELS:
                self.assertIsNotNone(curves.load_plot_rewards(rows, 'darkroom', method))
            item = specs[0]
            self.assertNotEqual(curves.cache_path_for(root, item, 0, 'sample'), curves.cache_path_for(root, item, 0, 'greedy'))

    def test_invalid_idt_lengths_and_oracle_seed_fail(self):
        with self.assertRaises(ValueError):
            IDT(config(low_per_high=3))
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            write_history(directory, cfg)
            with self.assertRaisesRegex(ValueError, 'complete episodes'):
                IDTDataset(cfg, directory, 'all', 1, 10)
            with self.assertRaisesRegex(ValueError, 'oracle rewards disagree'):
                DPTDataset({**cfg, 'collection_env_split_seed': 77}, directory, 'all', 1, 12)


    def test_train_save_resume_with_accumulation_and_validation(self):
        for method in ('DPT', 'IDT'):
            cfg = config(model=method, num_workers=0, train_batch_size=2, test_batch_size=2,
                         train_n_stream=1, train_source_timesteps=12, train_timesteps=2,
                         lr=0.001, beta1=0.9, beta2=0.99, weight_decay=0., num_warmup_steps=0,
                         summary_interval=1, eval_interval=1, ckpt_interval=1,
                         gradient_accumulation_steps=2)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_history(root, cfg)
                config_path = root / 'tiny.yaml'
                config_path.write_text(yaml.safe_dump(cfg))
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'ACCELERATE_USE_CPU': 'true', 'OMP_NUM_THREADS': '1'}
                command = [sys.executable, str(ROOT / 'gridworld/train_baseline.py'), '--config', str(config_path),
                           '--traj-dir', str(root), '--runs-root', str(root / 'runs'), '--mixed-precision', 'no']
                result = subprocess.run(command, cwd=ROOT / 'gridworld', env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                checkpoint = root / 'runs' / f'{method}-darkroom-seed0' / 'ckpt-2.pt'
                self.assertTrue(checkpoint.is_file())
                result = subprocess.run(command + ['--resume', str(checkpoint), '--train-timesteps', '3'],
                                        cwd=ROOT / 'gridworld', env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                saved = torch.load(checkpoint.with_name('ckpt-3.pt'), weights_only=False)
                self.assertEqual(saved['step'], 3)
                self.assertEqual(saved['config']['baseline_tokenization'], f'gridworld-{method.lower()}-gpt2-v1')
                self.assertTrue(all(int(state['step']) == 3 for state in saved['optimizer']['state'].values()))

    def test_task_split_matches_environment_even_when_collection_seed_differs(self):
        cfg = config('dktd', env_split_seed=11, collection_env_split_seed=2)
        groups = selected_group_ids(cfg, 'test')
        _, test_tasks = curves.SAMPLE_ENVIRONMENT['dktd'](cfg)
        np.testing.assert_array_equal([task_for_group(cfg, group) for group in groups], test_tasks)
        self.assertFalse(set(groups) & set(selected_group_ids(cfg, 'train')))


if __name__ == '__main__':
    unittest.main()
