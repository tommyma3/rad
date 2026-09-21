"""Behavioral checks for the fixed-recent-history Darkroom memory-size sweep."""

from pathlib import Path
import random
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gridworld'))
sys.path.insert(0, str(ROOT / 'gridworld/scripts'))
from test_gridworld_compressor_comparison import config as toy_config, batch, synthetic_history
from compressor_experiment import audit_darkroom_dataset, validate_checkpoint_config, apply_experiment_arguments
from dataset import RADDataset, CompressionPretrainDataset
from memory_capacity import recent_capacities
from memory_size_experiment import PROTOCOL, SIZES, validate_memory_size_config
from model.compressed_ad import RAD
from train_rad import get_rad_data_loader
from train_pretrain_compression import get_pretrain_data_loader
from evaluate_memory_size_comparison import aggregate, paired_differences, comparison_signature, metrics, load_best_checkpoint
from run_memory_size_comparison import training_commands, evaluation_command
from env import make_env, SAMPLE_ENVIRONMENT
from stable_baselines3.common.vec_env import DummyVecEnv


def config(size=15, pretrain=False):
    cfg = toy_config()
    cfg.pop('compressor_comparison')
    cfg.update(memory_size_comparison=PROTOCOL, grid_size=9, train_env_ratio=0.9,
               n_compress_tokens=size, n_transit=40 if pretrain else 30,
               first_recent_capacity=35 if pretrain else 30,
               recurrent_recent_capacity=35 if pretrain else 25,
               short_memory_keep=5, min_context_length=20, max_context_length=120,
               train_source_timesteps=120, train_timesteps=2, pretrain_timesteps=2,
               torch_compile=False, save_best_model=True)
    if not pretrain:
        cfg.pop('pretrain_timesteps')
    return cfg


class MemorySizeComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_best_checkpoint_selection_and_no_final_fallback(self):
        cfg = {**config(), 'train_timesteps': 4, 'gen_interval': 2}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            torch.save(dict(config=cfg, step=4, model={'weight': torch.tensor(4)}), run / 'ckpt-4.pt')
            with self.assertRaises(FileNotFoundError):
                load_best_checkpoint(run, 15, cfg['seed'], 4)
            best = dict(config=cfg, step=2, eval_reward=10., model={'weight': torch.tensor(2)})
            torch.save(best, run / 'best-model.pt')
            filename, checkpoint, _ = load_best_checkpoint(run, 15, cfg['seed'], 4)
            self.assertEqual(filename.name, 'best-model.pt')
            self.assertEqual(checkpoint['step'], 2)
            self.assertEqual(checkpoint['model']['weight'].item(), 2)
            for wrong in ({**best, 'step': 6}, {**best, 'step': 1},
                          {key: value for key, value in best.items() if key != 'eval_reward'},
                          {**best, 'config': {**cfg, 'save_best_model': False}},
                          {**best, 'config': {**cfg, 'train_timesteps': 6}}):
                torch.save(wrong, run / 'best-model.pt')
                with self.assertRaises(ValueError):
                    load_best_checkpoint(run, 15, cfg['seed'], 4)
            torch.save(best, run / 'best-model.pt')
            for size, seed in ((3, cfg['seed']), (15, cfg['seed'] + 1)):
                with self.assertRaises(ValueError):
                    load_best_checkpoint(run, size, seed, 4)
            (run / 'ckpt-4.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                load_best_checkpoint(run, 15, cfg['seed'], 4)

    def test_pilot_evaluates_within_budget_and_old_pretraining_is_reusable(self):
        args = SimpleNamespace(steps=2, batch_size=None, no_compile=False, cpu=False,
                               seed=None, runs_root=None, run_name=None, traj_dir=None, num_workers=None)
        cfg = {**config(), 'gen_interval': 10000}
        apply_experiment_arguments(cfg, args)
        self.assertEqual(cfg['gen_interval'], 2)
        self.assertTrue(cfg['save_best_model'])
        pretrained = {**config(pretrain=True), 'save_best_model': False}
        validate_checkpoint_config(cfg, pretrained)

    def test_fifteen_latents_matches_legacy_policy(self):
        cfg = config()
        legacy_cfg = {key: value for key, value in cfg.items()
                      if key not in ('first_recent_capacity', 'recurrent_recent_capacity', 'memory_size_comparison')}
        legacy = RAD(legacy_cfg).eval()
        fixed = RAD(cfg).eval()
        fixed.load_state_dict(legacy.state_dict(), strict=True)
        for limit in (None, 0, 1, 3):
            for length in (20, 30, 31, 50, 70, 120):
                legacy.set_curriculum(limit)
                fixed.set_curriculum(limit)
                with torch.no_grad():
                    left, right = legacy(batch(length)), fixed(batch(length))
                self.assertEqual(left['num_compressions'], right['num_compressions'])
                torch.testing.assert_close(left['loss_total'], right['loss_total'], rtol=0, atol=0)

    def test_token_boundaries_and_gradient_truncation_are_size_independent(self):
        for size in SIZES:
            model = RAD(config(size))
            decisions = []
            def compress(context, allow_gradient, old_latent_tokens=None):
                decisions.append(allow_gradient)
                return context.new_zeros((1, size, 8))
            model._compress_sequence = compress
            for limit in (None, 0, 1, 3):
                model.set_curriculum(limit)
                decisions.clear()
                latent, recent, _, _, info = model._roll_context_into_memory(torch.zeros(1, 360, 8))
                expected = 5 if limit is None else min(5, limit)
                self.assertEqual(info['num_compressions'], expected)
                self.assertEqual(model._count_compressions_for_sequence(360), expected)
                self.assertEqual(decisions, [False] * max(0, expected - 2) + [True] * min(expected, 2))
                self.assertLessEqual(recent.shape[1], model._recent_capacity(latent is not None))
            model.set_curriculum(None)
            latent, recent, events = None, None, []
            for index in range(360):
                recent = model._append_recent(recent, torch.zeros(1, 1, 8))
                latent, recent, _, _, info = model._compress_memory_until_fits(latent, recent)
                if info['num_compressions']:
                    events.append(index)
            self.assertEqual(events, [90, 151, 212, 273, 334])

    def test_real_gradients_and_no_initial_null_prefix(self):
        for size in SIZES:
            model = RAD(config(size))
            initial = torch.randn(2, 90, 8)
            packed, has_latent = model._pack_memory_input(None, initial)
            self.assertIs(packed, initial)
            self.assertFalse(has_latent)
            output = model(batch(120))
            output['loss_total'].backward()
            self.assertTrue(torch.isfinite(output['loss_total']))
            self.assertGreater(model.compression_transformer.compress_queries.grad.abs().sum().item(), 0)
            self.assertGreater(model.latent_gru_gate.weight.grad.abs().sum().item(), 0)
            self.assertIsNone(model.null_latent_tokens.grad)
            pretrain = RAD(config(size, pretrain=True))
            seen = []
            handle = pretrain.compression_transformer.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
            loss = pretrain(batch(40), pretrain=True)['loss_total']
            loss.backward()
            handle.remove()
            self.assertEqual(seen, [120])
            self.assertIsNone(pretrain.null_latent_tokens.grad)
            self.assertTrue(torch.isfinite(loss))

    def test_paired_data_for_all_sizes_and_both_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_history(directory, config())
            audit = audit_darkroom_dataset(config(), directory)
            self.assertEqual(len(audit['train_groups']), 73)
            self.assertEqual(len(audit['test_groups']), 8)
            for workers in (0, 1):
                for pretrain in (False, True):
                    reference = None
                    for size in SIZES:
                        cfg = config(size, pretrain)
                        cfg['num_workers'] = workers
                        cls, loader_fn = ((CompressionPretrainDataset, get_pretrain_data_loader) if pretrain
                                          else (RADDataset, get_rad_data_loader))
                        dataset = cls(cfg, directory, 'train', 2, 120)
                        dataset.rewards[:] = np.arange(120)[None, :]
                        if not pretrain:
                            self.assertEqual([dataset._raw_bucket_length_for_compressions(i) for i in range(5)],
                                             [30, 50, 70, 90, 110])
                        torch.randn(size * 3)
                        random.seed(size)
                        loader = loader_fn(dataset, 2, cfg)
                        iterator = iter(loader)
                        observed = [{key: value.clone() for key, value in next(iterator).items()} for _ in range(3)]
                        if reference is not None:
                            for left, right in zip(reference, observed):
                                for key in left:
                                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                        reference = observed
                        del iterator, loader

    def test_checkpoint_roundtrip_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            for size in SIZES:
                cfg = config(size, True)
                pretrained = RAD(cfg).eval()
                filename = Path(directory) / f'pretrain{size}.pt'
                torch.save(dict(config=cfg, step=2, model=pretrained.state_dict()), filename)
                policy = RAD(config(size)).eval()
                policy.load_pretrained_compression(filename)
                context = torch.randn(2, 80, 8)
                torch.testing.assert_close(pretrained._compression_candidate(context)[0],
                                           policy._compression_candidate(context)[0], rtol=0, atol=0)
                for key, value in [('n_compress_tokens', size + 3), ('seed', 123),
                                   ('memory_size_comparison', 'wrong'), ('always_use_latent_prefix', True),
                                   ('first_recent_capacity', 99)]:
                    wrong = {**cfg, key: value}
                    with self.assertRaises(ValueError):
                        validate_checkpoint_config(config(size), wrong)

    def test_protocol_and_training_seed_statistics(self):
        for size in SIZES:
            validate_memory_size_config(config(size))
            validate_memory_size_config(config(size, True), pretrain=True)
        for wrong in (dict(first_recent_capacity=None), dict(recurrent_recent_capacity=5)):
            with self.assertRaises(ValueError):
                recent_capacities({**config(), **wrong})
        self.assertEqual(comparison_signature(config(3)), comparison_signature(config(60)))
        rows = [dict(n_latents=size, train_seed=seed, **metrics(np.full((20, 8, 100), seed * 2 + (size == 30))))
                for size in (15, 30) for seed in range(3)]
        summary = aggregate(rows)[0]
        self.assertEqual(summary['n_training_seeds'], 3)
        self.assertEqual(summary['std_over_training_seeds'], 2)
        for row in paired_differences(rows):
            self.assertEqual(row['mean_return'], 1)


    def test_online_memory_persists_across_episodes_and_resets_between_trials(self):
        for size in SIZES:
            cfg = config(size)
            model = RAD(cfg).eval()
            _, goals = SAMPLE_ENVIRONMENT['darkroom'](cfg)
            envs = DummyVecEnv([make_env(cfg, goal=goals[0])])
            try:
                first = model.evaluate_in_context(envs, 100, action_seed=17)
                second = model.evaluate_in_context(envs, 100, action_seed=17)
            finally:
                envs.close()
            np.testing.assert_array_equal(first['reward_episode'], second['reward_episode'])
            self.assertEqual(first['compression_events'], second['compression_events'])
            self.assertEqual(first['reward_episode'].shape, (1, 5))
            # Episode boundaries are every 20 steps, while memory spans them.
            self.assertEqual(first['compression_events'], [29, 50, 71, 92])

    def test_launcher_pairs_each_size_and_seed_and_separates_pilot(self):
        args = SimpleNamespace(runs_root=Path('runs'), traj_dir=Path('datasets'), config='rad_dr_memory_size',
                               python='python', sizes=list(SIZES), seeds=[0, 1, 2], eval_seeds=[0, 1],
                               cpu=False, num_workers=0, stage='all', pilot_steps=2, pilot_batch_size=2,
                               pilot_episodes=5, episodes=100)
        commands = training_commands(args)
        self.assertEqual(len(commands), 30)
        for pretrain, train in zip(commands[::2], commands[1::2]):
            self.assertEqual(pretrain[pretrain.index('--n_latents') + 1], train[train.index('--n_latents') + 1])
            self.assertEqual(pretrain[pretrain.index('--seed') + 1], train[train.index('--seed') + 1])
            self.assertEqual(Path(train[train.index('--pretrain_ckpt') + 1]).parent.name,
                             pretrain[pretrain.index('--run_name') + 1])
        pilot = training_commands(args, pilot=True)
        self.assertEqual(len(pilot), 10)
        self.assertTrue(all(Path(command[command.index('--runs_root') + 1]).name == 'pilot' for command in pilot))
        self.assertIn('--pilot', evaluation_command(args, pilot=True))


if __name__ == '__main__':
    unittest.main()
