"""CPU behavioral tests for the isolated AD memory-transfer experiment."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gridworld'))
from ad_memory_transfer import (AD, RAD, AD_PREFIXES, EvaluationRAD, audit_dataset,
    configure_stage, digest, evaluate_policy, initialize_ad, load_source, resolve_config)
from train_ad_memory_transfer import (load_settings, train_stage, read_checkpoint,
    evaluate_experiment, stage_directory, validate_stage_checkpoint)
from baseline_dataset import task_for_group
from env import make_env


def source_config(env='darkroom', mapping='legacy'):
    return dict(model='AD', env=env, device='cpu', n_transit=10, mixed_precision='no',
                grid_size=3, num_actions=5, horizon=5, env_split_seed=0,
                collection_env_split_seed=0, train_env_ratio=0.67, alg='ppo', alg_seed=0,
                tf_n_embd=8, tf_n_layer=1, tf_n_head=2, tf_dim_feedforward=16,
                tf_dropout=0., label_smoothing=0., dataset_task_mapping=mapping)


def settings(env='darkroom'):
    cfg = load_settings(ROOT / 'gridworld/config/model/rad_ad_transfer_dr.yaml')
    cfg.update(env=env, collection_env_split_seed=0, n_transit=6, n_compress_tokens=3,
               short_memory_keep=1, compress_n_layers=1, compress_n_heads=2,
               pretrain_timesteps=2, train_timesteps=4, pretrain_batch_size=2,
               train_batch_size=2, train_n_stream=1, train_source_timesteps=40,
               max_context_length=40, min_context_length=3, eval_interval=2,
               ckpt_interval=1, log_interval=2, eval_episodes=3, eval_batch_size=32,
               transfer_curriculum=[dict(fraction=0., max_compressions=None,
                   length_distribution=dict(short=0., medium=0., long=0., very_long=1.))])
    return cfg


def make_fixture(directory, env='darkroom', mapping='legacy'):
    directory = Path(directory)
    cfg = source_config(env, mapping)
    torch.manual_seed(71)
    source = AD(cfg)
    source_path = directory / 'ad.pt'
    torch.save(dict(step=1, model=source.state_dict(), config=cfg), source_path)
    loaded = load_source(source_path)
    resolved = resolve_config(settings(env), loaded, source_path, directory, 5)
    rng = random.Random(17)
    with h5py.File(directory / f'history_{env}_ppo_alg-seed0.hdf5', 'w') as history:
        for group_id in range(3 ** (2 if env == 'darkroom' else 4)):
            task = task_for_group(resolved, group_id)
            kwargs = dict(goal=task) if env == 'darkroom' else dict(key=task[:2], goal=task[2:])
            environment = make_env(cfg, **kwargs)()
            state, _ = environment.reset()
            states, actions, rewards, next_states = [], [], [], []
            for _ in range(40):
                action = rng.randrange(5)
                states.append(state.copy())
                state, reward, terminated, truncated, _ = environment.step(action)
                actions.append(action)
                rewards.append(reward)
                if terminated or truncated:
                    state, _ = environment.reset()
                next_states.append(state.copy())
            group = history.create_group(str(group_id))
            group['states'] = np.asarray(states)[:, None]
            group['actions'] = np.asarray(actions)[:, None]
            group['rewards'] = np.asarray(rewards, dtype=np.float32)[:, None]
            group['next_states'] = np.asarray(next_states)[:, None]
            environment.close()
    return resolved, loaded, source_path


def batch(length=30):
    generator = torch.Generator().manual_seed(4)
    return dict(states=torch.randint(0, 3, (2, length, 2), generator=generator),
                actions=F.one_hot(torch.randint(0, 5, (2, length), generator=generator), 5),
                rewards=torch.randn(2, length, generator=generator),
                context_lengths=torch.full((2,), length))


class ADMemoryTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_transfer_logits_positions_compiled_keys_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, source, _ = make_fixture(directory)
            # Exercise the normalized compiled checkpoint format.
            wrapped = Path(directory) / 'compiled.pt'
            torch.save(dict(source, model={k.replace('transformer.', 'transformer._orig_mod.'): v
                                          for k, v in source['model'].items()}), wrapped)
            loaded = load_source(wrapped)
            model = RAD(cfg).eval()
            initialize_ad(model, loaded)
            ad = AD(source['config']).eval()
            ad.load_state_dict(source['model'])
            sample = batch(6)
            tokens = ad._build_token_sequence(sample['states'], sample['actions'], sample['rewards'])
            with torch.no_grad():
                expected = ad.pred_action(ad.transformer(tokens)[:, ::3])
                actual = model.pred_action(model._forward_ad_transformer(tokens, False)[:, ::3])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(model(sample)['loss_action'], ad(sample)['loss_action'], rtol=0, atol=0)
            torch.testing.assert_close(model.ad_transformer.pos_embedding,
                                       source['model']['transformer.pos_embedding'][:, :18], rtol=0, atol=0)
            broken = deepcopy(source)
            del broken['model']['embed_action.bias']
            with self.assertRaisesRegex(ValueError, 'Missing AD'):
                initialize_ad(model, broken)
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                resolve_config(dict(settings(), n_transit=11), source, wrapped, directory, 5)
            model.config['tf_n_head'] = 1
            with self.assertRaisesRegex(ValueError, 'tf_n_head'):
                initialize_ad(model, source)

    def test_pretraining_freezes_ad_and_joint_training_updates_both_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, source, _ = make_fixture(directory)
            model = RAD(cfg)
            initialize_ad(model, source)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            configure_stage(model, 'pretrain')
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
            model(batch(6), pretrain=True)['loss_recon'].backward()
            opt.step()
            for name, parameter in model.named_parameters():
                if name.startswith(AD_PREFIXES):
                    self.assertIsNone(parameter.grad, name)
                    self.assertTrue(torch.equal(before[name], parameter), name)
            self.assertFalse(torch.equal(before['compression_transformer.compress_queries'],
                                         model.compression_transformer.compress_queries))
            configure_stage(model, 'finetune')
            model.zero_grad(set_to_none=True)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            output = model(batch())
            self.assertGreater(output['num_compressions'], 1)
            output['loss_action'].backward()
            names = ['ad_transformer.blocks.0.attn.qkv_proj.weight',
                     'compression_transformer.compress_queries', 'latent_type_embedding',
                     'latent_gru_gate.weight']
            parameters = dict(model.named_parameters())
            for name in names:
                self.assertTrue(torch.isfinite(parameters[name].grad).all(), name)
                self.assertGreater(float(parameters[name].grad.abs().sum()), 0, name)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
            opt.step()
            for name in names:
                self.assertFalse(torch.equal(before[name], parameters[name]), name)
            for name in before:
                if name.startswith(('reconstruction_decoder.', 'null_latent_tokens', 'latent_multiplicative_gate.')):
                    self.assertTrue(torch.equal(before[name], model.state_dict()[name]), name)

    def test_data_split_audit_uses_actual_source_membership(self):
        for mapping in ('legacy', 'collection_order'):
            with self.subTest(mapping=mapping), tempfile.TemporaryDirectory() as directory:
                cfg, _, _ = make_fixture(directory, mapping=mapping)
                audit = audit_dataset(cfg)
                self.assertFalse(set(audit['eval_task_ids']) & set(audit['source_train_task_ids']))
                self.assertEqual(audit['train_task_ids'], audit['source_train_task_ids'])
                with h5py.File(audit['path'], 'r+') as history:
                    history[str(audit['train_groups'][0])]['rewards'][0, 0] = 42
                with self.assertRaisesRegex(ValueError, 'rewards disagree'):
                    audit_dataset(cfg)

    def test_complete_stages_resume_exactly_and_evaluate_both_environments(self):
        for env in ('darkroom', 'dktd'):
            with self.subTest(env=env), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg, source, ad_path = make_fixture(root, env)
                source_hash = digest(ad_path)
                data_path = root / f'history_{env}_ppo_alg-seed0.hdf5'
                data_hash = digest(data_path)
                run = root / 'run'
                partial = train_stage(cfg, 'pretrain', run, 'cpu', source=source, stop_after=1)
                with self.assertRaisesRegex(ValueError, 'completed'):
                    validate_stage_checkpoint(read_checkpoint(partial), 'pretrain', complete=True)
                pretrain = train_stage(read_checkpoint(partial)['config'], 'pretrain', run, 'cpu', resume=partial)
                complete = read_checkpoint(pretrain)
                comparison = train_stage(cfg, 'pretrain', root / 'uninterrupted', 'cpu', source=source)
                uninterrupted = read_checkpoint(comparison)
                for key in complete['model']:
                    self.assertTrue(torch.equal(complete['model'][key], uninterrupted['model'][key]), key)
                cfg_ft = dict(complete['config'], pretrain_source=dict(path=str(pretrain.resolve()), sha256=digest(pretrain)))
                partial = train_stage(cfg_ft, 'finetune', run, 'cpu', pretrain=pretrain, stop_after=2)
                final = train_stage(cfg_ft, 'finetune', run, 'cpu', resume=partial)
                reference_final = train_stage(cfg_ft, 'finetune', root / 'uninterrupted', 'cpu', pretrain=pretrain)
                final_ckpt, reference_ckpt = read_checkpoint(final), read_checkpoint(reference_final)
                for key in final_ckpt['model']:
                    self.assertTrue(torch.equal(final_ckpt['model'][key], reference_ckpt['model'][key]), key)
                self.assertEqual(final_ckpt['lr_sched'], reference_ckpt['lr_sched'])
                self.assertEqual(final_ckpt['best_step'], reference_ckpt['best_step'])
                self.assertEqual(final_ckpt['step'], 4)
                with self.assertRaisesRegex(ValueError, 'newer checkpoint'):
                    stage_directory(run, 'finetune', partial)
                rows = evaluate_experiment(run, ad_path, pretrain, root / 'eval', 'cpu', [0, 1])
                self.assertEqual(len(rows), 5)
                self.assertGreater(rows[2]['total_compressions'][0], 1)
                self.assertEqual(digest(ad_path), source_hash)
                self.assertEqual(digest(data_path), data_hash)
                (run / 'finetune/best-model.pt').unlink()
                with self.assertRaises(FileNotFoundError):
                    evaluate_experiment(run, ad_path, pretrain, root / 'missing-best', 'cpu', [0])
                with self.assertRaises(FileExistsError):
                    train_stage(cfg, 'pretrain', run, 'cpu', source=source)

    def test_isolation_and_memory_intervention_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, source, _ = make_fixture(directory)
            cfg['dataset_audit'] = audit_dataset(cfg)
            torch.manual_seed(9)
            legacy = RAD(cfg)
            before = {k: v.clone() for k, v in legacy.state_dict().items()}
            model = EvaluationRAD(cfg)
            initialize_ad(model, source)
            rng = torch.get_rng_state().clone()
            data = random.getstate()
            for intervention in ('intact', 'zero', 'shuffle'):
                rewards, count = evaluate_policy(model, cfg, 0, intervention)
                self.assertGreater(count, 0)
                self.assertTrue(np.isfinite(rewards).all())
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(data, random.getstate())
            for key in before:
                self.assertTrue(torch.equal(before[key], legacy.state_dict()[key]), key)
            with self.assertRaisesRegex(ValueError, 'non-transfer'):
                stage_directory(directory, 'pretrain')

    def test_cli_runs_outside_gridworld_and_config_resolution(self):
        result = subprocess.run([sys.executable, str(ROOT / 'gridworld/train_ad_memory_transfer.py'), '--help'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = load_settings(ROOT / 'gridworld/config/model/rad_ad_transfer_dktd.yaml')
        self.assertEqual(cfg['pretrain_timesteps'], 50000)
        self.assertEqual(cfg['n_compress_tokens'], 60)

    def test_cli_all_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, source_path = make_fixture(root)
            config_path = root / 'transfer.yaml'
            config_path.write_text(yaml.safe_dump(settings()), encoding='utf-8')
            result = subprocess.run([sys.executable, str(ROOT / 'gridworld/train_ad_memory_transfer.py'),
                '--stage', 'all', '--config', str(config_path), '--ad-checkpoint', str(source_path),
                '--dataset-dir', str(root), '--run-dir', str(root / 'run'), '--device', 'cpu',
                '--threads', '1'], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / 'run/pretrain/pretrain-final.pt').exists())
            self.assertTrue((root / 'run/finetune/best-model.pt').exists())
            self.assertEqual(read_checkpoint(root / 'run/finetune/ckpt-4.pt')['step'], 4)


if __name__ == '__main__':
    unittest.main()
