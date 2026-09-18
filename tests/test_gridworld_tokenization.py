"""Contract and end-to-end checks for isolated query-last tokenization variants."""

import copy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gridworld'))
sys.path.insert(0, str(ROOT / 'tests'))

from test_gridworld_baselines import config as base_config, write_history
from dataset import ADDataset
from env import make_env
from model import AD, RAD, MODEL
from model.transition_ad import ADTransition, RADTransition, MemorySchedule, TOKENIZATION
from transition_dataset import TransitionDataset, EndpointBatchSampler, exact_length_collate
from train_tokenization import load_pretraining


def config(method='RAD_DPT', **changes):
    result = base_config(model=method, tokenization=TOKENIZATION, policy_token_budget=7,
                         n_transit=4, n_compress_tokens=2, short_memory_keep=1,
                         latent_update_mode='gru_gate', max_gradient_rounds=2,
                         min_context_length=0, max_context_length=100, max_compressions=None,
                         lengths_per_batch=4, seed=17)
    result.update(changes)
    return result


def batch(length, size=2):
    generator = torch.Generator().manual_seed(100 + length)
    return dict(states=torch.randint(0, 2, (size, length, 2), generator=generator),
                actions=torch.randint(0, 5, (size, length), generator=generator),
                rewards=torch.randint(0, 2, (size, length), generator=generator),
                next_states=torch.randint(0, 2, (size, length, 2), generator=generator),
                query_states=torch.randint(0, 2, (size, 2), generator=generator),
                target_actions=torch.randint(0, 5, (size,), generator=generator))


def tokens(model, values):
    return model.transition_tokens(*(values[key] for key in ('states', 'actions', 'rewards', 'next_states')))


class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_registry_is_unchanged(self):
        self.assertIs(MODEL['AD'], AD)
        self.assertIs(MODEL['RAD'], RAD)
        self.assertNotIn('AD_DPT', MODEL)
        self.assertNotIn('RAD_DPT', MODEL)

    def test_query_is_last_and_only_query_predicts_target(self):
        model = ADTransition(config('AD_DPT')).eval()
        values = batch(4)
        history = tokens(model, values)
        sequence = torch.cat((history, model.query_token(values['query_states'])), 1)
        expected = model.pred_action(model.ad_transformer(sequence)[:, -1])
        actual = model.policy_logits(values['query_states'], model.replay(history))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model(values)['loss_action'], model.loss_fn(expected, values['target_actions']))
        changed = sequence.clone()
        changed[:, -1] += 50
        torch.testing.assert_close(model.ad_transformer(sequence)[:, :-1], model.ad_transformer(changed)[:, :-1])
        # Adding irrelevant target/future fields cannot affect predictions/loss.
        with_extra = {**values, 'target_rewards': torch.ones(2), 'future_states': torch.zeros(2, 9, 2)}
        torch.testing.assert_close(model(values)['loss_action'], model(with_extra)['loss_action'])
        self.assertEqual(model(batch(0))['num_compressions'], 0)

    def test_memory_replay_matches_online_at_every_length(self):
        for null_prefix in (False, True):
            for keep in (0, 1, 3):
                model = RADTransition(config(always_use_latent_prefix=null_prefix, short_memory_keep=keep)).eval()
                history = tokens(model, batch(45))
                memory = model.empty_memory(2)
                for length in range(46):
                    replay = model.replay(history[:, :length])
                    count, recent = model.schedule.state_after(length)
                    self.assertEqual((memory[2], memory[1].shape[1]), (count, recent))
                    if memory[0] is not None:
                        torch.testing.assert_close(memory[0], replay[0])
                    torch.testing.assert_close(memory[1], replay[1])
                    query = torch.tensor([[0, 1], [1, 0]])
                    torch.testing.assert_close(model.policy_logits(query, memory), model.policy_logits(query, replay))
                    if length < 45:
                        memory = model.append_transitions(memory, history[:, length:length + 1])

    def test_latent_prefix_cannot_read_query_or_recent_history(self):
        model = RADTransition(config()).eval()
        mask = model._get_attention_mask_for_latent(7)
        self.assertTrue(mask[:2, 2:].all())
        self.assertFalse(mask[2:, :2].any())
        self.assertFalse(mask[-1].any())
        values = batch(15)
        memory = model.replay(tokens(model, values))
        old = memory[0].clone()
        model.policy_logits(torch.zeros(2, 2), memory)
        model.policy_logits(torch.ones(2, 2), memory)
        torch.testing.assert_close(old, memory[0])

    def test_only_latest_compression_rounds_receive_gradients(self):
        model = RADTransition(config(max_gradient_rounds=2)).train()
        calls = []
        original = model._compress_sequence

        def record(context, allow_gradient, old_latent_tokens=None):
            result = original(context, allow_gradient, old_latent_tokens)
            calls.append((allow_gradient, result.requires_grad))
            return result

        model._compress_sequence = record
        loss = model(batch(25))['loss_action']
        loss.backward()
        self.assertGreater(len(calls), 2)
        self.assertEqual(calls[-2:], [(True, True)] * 2)
        self.assertTrue(all(call == (False, False) for call in calls[:-2]))
        self.assertGreater(model.compression_transformer.compress_queries.grad.abs().sum(), 0)
        self.assertIsNotNone(model.latent_gru_gate.weight.grad)
        self.assertIsNone(model.reconstruction_decoder.position_queries.grad)
        zero = RADTransition(config(max_gradient_rounds=0)).train()
        zero(batch(25))['loss_action'].backward()
        self.assertIsNone(zero.compression_transformer.compress_queries.grad)

    def test_microbatch_weighting_matches_individual_examples(self):
        model = RADTransition(config()).eval()  # Disable dropout while retaining gradients.
        reference = copy.deepcopy(model)
        examples = [{key: value[0].numpy() for key, value in batch(length, 1).items()} for length in (0, 4, 8, 8, 20)]
        for micro in exact_length_collate(examples):
            (model(micro)['loss_action'] * len(micro['states']) / len(examples)).backward()
        for example in examples:
            (reference(exact_length_collate([example])[0])['loss_action'] / len(examples)).backward()
        for (name, left), (_, right) in zip(model.named_parameters(), reference.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5, msg=name)
        padded = batch(4)
        padded['context_lengths'] = torch.tensor([2, 4])
        with self.assertRaisesRegex(ValueError, 'padded'):
            model(padded)

    def test_reader_preserves_legacy_source_actions_and_group_selection(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            path, _ = write_history(directory, cfg)
            original = path.read_bytes()
            legacy = ADDataset(cfg, directory)
            dataset = TransitionDataset(cfg, directory)
            np.testing.assert_array_equal(dataset.states, legacy.states)
            np.testing.assert_array_equal(dataset.actions, legacy.actions)
            endpoint = dataset[(0, 0, 2)]
            self.assertEqual(endpoint['target_actions'], legacy.actions[0, 2])
            np.testing.assert_array_equal(endpoint['query_states'], legacy.states[0, 2])
            self.assertEqual(len(endpoint['states']), 2)
            np.testing.assert_array_equal(dataset.next_states[0, 3], dataset.states[0, 3])  # Stay action at terminal.
            self.assertEqual(original, path.read_bytes())

    def test_sampler_varies_refill_lengths_and_resumes_exactly(self):
        cfg = config(curriculum_schedule=[{'step': 0, 'max_compressions': 1, 'length_distribution': {'short': 1.}},
                                         {'step': 20, 'max_compressions': None, 'length_distribution': {'short': 1.}}])
        with tempfile.TemporaryDirectory() as directory:
            write_history(directory, cfg)
            dataset = TransitionDataset(cfg, directory)
            sampler = EndpointBatchSampler(dataset, cfg, 12, 0, 60)
            resumed = EndpointBatchSampler(dataset, cfg, 12, 20, 60)
            self.assertEqual(list(sampler)[20:], list(resumed))
            other_rank = EndpointBatchSampler(dataset, cfg, 12, 0, 60, rank=1)
            observed = set()
            for step, indices in enumerate(sampler):
                self.assertEqual([row[2] for row in indices], [row[2] for row in other_rank.batch_at(step)])
                for _, _, length in indices:
                    state = sampler.schedule.state_after(length)
                    observed.add(state)
                    if step < 20:
                        self.assertLessEqual(state[0], 1)
            self.assertIn((0, 0), observed)
            self.assertGreater(len({recent for count, recent in observed if count == 1}), 1)
            self.assertGreater(max(count for count, _ in observed), 1)

    def test_pretraining_and_cross_tokenization_rejection(self):
        cfg = config()
        model = RADTransition(cfg)
        model.configure_phase(True)
        model(batch(4), pretrain=True)['loss_recon'].backward()
        self.assertIsNotNone(model.embed_context.weight.grad)
        self.assertIsNotNone(model.reconstruction_decoder.position_queries.grad)
        self.assertIsNone(model.pred_action.weight.grad)
        checkpoint = {'config': cfg, 'model': model.state_dict(), 'phase': 'pretrain'}
        restored = RADTransition(cfg)
        load_pretraining(restored, checkpoint)
        torch.testing.assert_close(model.embed_context.weight, restored.embed_context.weight)
        bad = {**checkpoint, 'config': {**cfg, 'tokenization': 'sar'}}
        with self.assertRaises(ValueError):
            load_pretraining(restored, bad)
        with self.assertRaises(ValueError):
            MemorySchedule(3, 2, 1)

    def test_rollout_queries_reset_observations_and_keeps_terminal_transition(self):
        for cls, method in ((ADTransition, 'AD_DPT'), (RADTransition, 'RAD_DPT')):
            cfg = config(method)
            model = cls(cfg).eval()
            queries, transitions = [], []
            original = model.policy_logits
            def capture(query, memory):
                queries.append(query.clone())
                logits = original(query, memory)
                forced = torch.full_like(logits, -100.)
                forced[:, 1] = 100.  # Move left from reset center.
                return forced
            model.policy_logits = capture
            embed = model.transition_tokens
            def capture_transition(*fields):
                transitions.append(fields[-1].clone())
                return embed(*fields)
            model.transition_tokens = capture_transition
            envs = DummyVecEnv([make_env(cfg, goal=np.array([0, 0]))])
            try:
                output = model.evaluate_in_context(envs, 12, sample=False)
            finally:
                envs.close()
            self.assertEqual(output['reward_episode'].shape, (1, 3))
            torch.testing.assert_close(queries[4], torch.tensor([[1., 1.]]))
            torch.testing.assert_close(transitions[3], torch.tensor([[[0, 1]]], dtype=transitions[3].dtype))
            if method == 'RAD_DPT':
                self.assertGreater(output['total_compressions'], 0)

    def test_training_checkpoint_resume_and_evaluation_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(max_context_length=20, train_n_stream=1, train_source_timesteps=12,
                         train_timesteps=2, train_batch_size=4, test_batch_size=4,
                         num_workers=0, torch_compile=False, num_warmup_steps=1,
                         summary_interval=1, eval_interval=1, ckpt_interval=1, progress_interval=1,
                         lr=.001, beta1=.9, beta2=.99, weight_decay=.01,
                         pretrain=dict(n_transit=4, pretrain_timesteps=2, pretrain_batch_size=4,
                                       pretrain_warmup_steps=1, pretrain_lr=.001, num_workers=0))
            write_history(root, cfg)
            base_command = [sys.executable] + (['-S'] if sys.flags.no_site else [])
            def run(arguments):
                result = subprocess.run(base_command + arguments, cwd=ROOT / 'gridworld', env=os.environ.copy(),
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result
            checkpoints = []
            for method in ('AD_DPT', 'RAD_DPT'):
                cfg['model'] = method
                config_path = root / f'{method}.yaml'
                config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
                common = ['train_tokenization.py', '--config', str(config_path), '--traj-dir', str(root),
                          '--runs-root', str(root / 'runs'), '--mixed-precision', 'no']
                extra = []
                if method == 'RAD_DPT':
                    run(common + ['--phase', 'pretrain'])
                    extra = ['--pretrain-ckpt', str(root / 'runs' / f'{method}-pretrain-darkroom-seed17' / 'ckpt-2.pt')]
                run(common + extra)
                trained = root / 'runs' / f'{method}-darkroom-seed17'
                resumed = root / f'{method}-resume'
                resumed.mkdir()
                shutil.copyfile(trained / 'ckpt-1.pt', resumed / 'ckpt-1.pt')
                run(['train_tokenization.py', '--resume', str(resumed / 'ckpt-1.pt'), '--traj-dir', str(root)])
                left = torch.load(trained / 'ckpt-2.pt', weights_only=False, map_location='cpu')
                right = torch.load(resumed / 'ckpt-2.pt', weights_only=False, map_location='cpu')
                for key in left['model']:
                    torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0, msg=key)
                checkpoints += ['--checkpoint', f'{method}={trained}']
            for cls, method in ((AD, 'AD'), (RAD, 'RAD')):
                legacy_config = {**cfg, 'model': method, 'n_compress_tokens': 3}
                checkpoint_path = root / f'{method}.pt'
                torch.save({'config': legacy_config, 'model': cls(legacy_config).state_dict(), 'step': 0}, checkpoint_path)
                checkpoints += ['--checkpoint', f'{method}={checkpoint_path}']
            run(['scripts/evaluate_tokenization.py', *checkpoints, '--episodes', '2',
                 '--eval-seeds', '0', '--device', 'cpu', '--output-dir', str(root / 'comparison')])
            self.assertTrue((root / 'comparison' / 'comparison.png').is_file())
            self.assertTrue((root / 'comparison' / 'metrics.json').is_file())


if __name__ == '__main__':
    unittest.main()
