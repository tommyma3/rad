"""Behavioral checks for the Darkroom bottleneck comparison (CPU compatible)."""
import copy
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gridworld'))
from compressor_experiment import PROTOCOL, audit_darkroom_dataset, select_dataset_groups, comparison_optimizer_step
from baseline_dataset import task_for_group
from dataset import ADDataset, RADDataset, CompressionPretrainDataset
from model.compressed_ad import RAD
from optimizer_utils import build_rad_optimizer_param_groups
from train_rad import get_rad_data_loader
from train_pretrain_compression import get_pretrain_data_loader


def config(variant='ae'):
    return dict(compressor_type=variant, device='cpu', n_transit=6, mixed_precision='no',
                grid_size=3, num_actions=5, n_compress_tokens=3, short_memory_keep=1,
                compress_n_layers=1, compress_n_heads=2, max_gradient_rounds=2,
                max_context_length=60, min_context_length=3, always_use_latent_prefix=False,
                latent_update_mode='gru_gate', tf_n_embd=8, tf_n_head=2, tf_n_layer=1,
                tf_dim_feedforward=16, tf_dropout=0, label_smoothing=0, seed=7,
                vq_codebook_size=8, vae_kl_weight=0.001, vae_kl_warmup_steps=10,
                vq_commitment_weight=0.25, env='darkroom', horizon=20, train_env_ratio=0.67,
                env_split_seed=0, collection_env_split_seed=0, dataset_task_mapping='collection_order',
                compressor_comparison=PROTOCOL, data_seed=7, dynamics=False,
                alg='ppo', alg_seed=0, train_n_stream=2, train_source_timesteps=60,
                rad_batching_strategy='compression_buckets', num_workers=0, lr=0.001,
                pretrain_timesteps=2)


def batch(length=40):
    generator = torch.Generator().manual_seed(92)
    return dict(states=torch.randint(0, 3, (2, length, 2), generator=generator),
                actions=F.one_hot(torch.randint(0, 5, (2, length), generator=generator), 5).float(),
                rewards=torch.randn(2, length, generator=generator),
                context_lengths=torch.full((2,), length))


def synthetic_history(directory, cfg):
    filename = Path(directory) / 'history_darkroom_ppo_alg-seed0.hdf5'
    with h5py.File(filename, 'w') as history:
        for group_id in range(cfg['grid_size'] ** 2):
            group = history.create_group(str(group_id))
            goal = task_for_group(cfg, group_id)
            states = np.broadcast_to(goal, (cfg['train_source_timesteps'], cfg['train_n_stream'], 2)).copy()
            group['states'] = states
            group['next_states'] = states
            group['actions'] = np.full(states.shape[:2], 4, dtype=np.int64)
            group['rewards'] = np.ones(states.shape[:2], dtype=np.float32)
    return filename


class CompressorComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_ae_state_and_outputs(self):
        old_config = config()
        old_config.pop('compressor_type')
        torch.manual_seed(1)
        old = RAD(old_config).eval()
        torch.manual_seed(1)
        explicit = RAD(config()).eval()
        self.assertEqual(old.state_dict().keys(), explicit.state_dict().keys())
        explicit.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(old(batch())['loss_total'], explicit(batch())['loss_total'], rtol=0, atol=0)

    def test_shared_initialization_unchanged_by_extra_heads(self):
        torch.manual_seed(5)
        baseline = RAD(config()).state_dict()
        for variant in ('vae', 'vq_vae'):
            torch.manual_seed(5)
            candidate = RAD(config(variant)).state_dict()
            for key, value in baseline.items():
                self.assertTrue(torch.equal(value, candidate[key]), key)

    def test_vae_kl_formula_and_independent_noise(self):
        model = RAD(config('vae')).eval()
        context = torch.randn(2, 10, 8)
        mean, aux = model._compression_candidate(context)
        torch.testing.assert_close(aux['loss_kl'], 0.5 * mean.square().mean())
        state = torch.get_rng_state().clone()
        model.vae_eval_mode = 'sample'
        model.reset_latent_rng(42)
        sampled, _ = model._compression_candidate(context)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        model.reset_latent_rng(42)
        repeated, _ = model._compression_candidate(context)
        self.assertTrue(torch.equal(sampled, repeated))
        self.assertFalse(torch.equal(sampled, mean))

    def test_vq_quantizes_and_has_separate_gradient_paths(self):
        model = RAD(config('vq_vae')).eval()
        tokens, aux = model._compression_candidate(torch.randn(2, 10, 8))
        codes = model.compression_transformer.codebook.weight
        distances = (tokens.unsqueeze(-2) - codes).square().sum(-1)
        self.assertTrue((distances.min(-1).values < 1e-10).all())
        tokens.square().mean().backward(retain_graph=True)
        self.assertIsNone(codes.grad)
        self.assertIsNotNone(model.compression_transformer.compress_queries.grad)
        aux['loss_codebook'].backward()
        self.assertGreater(codes.grad.abs().sum().item(), 0)

    def test_pretrain_total_losses_and_updates(self):
        for variant in ('ae', 'vae', 'vq_vae'):
            model = RAD(config(variant)).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            output = model(batch(6), pretrain=True, pretrain_step=10)
            expected = output['loss_recon']
            if variant == 'vae':
                expected = expected + 0.001 * output['loss_kl']
            if variant == 'vq_vae':
                expected = expected + output['loss_codebook'] + 0.25 * output['loss_commitment']
            torch.testing.assert_close(output['loss_total'], expected)
            output['loss_total'].backward()
            self.assertGreater(model.compression_transformer.compress_queries.grad.abs().sum().item(), 0)
            if variant == 'vae':
                self.assertGreater(model.compression_transformer.posterior_logvar.weight.grad.abs().sum().item(), 0)
            if variant == 'vq_vae':
                before = model.compression_transformer.codebook.weight.detach().clone()
            optimizer.step()
            if variant == 'vq_vae':
                self.assertFalse(torch.equal(before, model.compression_transformer.codebook.weight))

    def test_kl_warmup(self):
        model = RAD(config('vae'))
        aux = {'loss_kl': torch.tensor(2.)}
        model._pretrain_step = 5
        self.assertAlmostEqual(model._bottleneck_loss(aux, True).item(), 0.001, places=7)
        self.assertAlmostEqual(model._bottleneck_loss(aux, False).item(), 0.002, places=7)

    def test_auxiliary_losses_average_only_recent_gradient_rounds(self):
        for variant in ('vae', 'vq_vae'):
            model = RAD(config(variant)).train()
            entries = []
            original = model._compression_candidate
            def capture(context):
                candidate, aux = original(context)
                if torch.is_grad_enabled():
                    entries.append(aux)
                return candidate, aux
            model._compression_candidate = capture
            output = model(batch())
            self.assertGreater(output['num_compressions'], 2)
            self.assertEqual(output['auxiliary_rounds'], 2)
            for key in entries[0]:
                torch.testing.assert_close(output[key], torch.stack([entry[key] for entry in entries]).mean())
            output['loss_total'].backward()
            self.assertTrue(torch.isfinite(output['loss_total']))
            parameter = (model.compression_transformer.posterior_mean.weight if variant == 'vae'
                         else model.compression_transformer.codebook.weight)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_zero_compression_and_zero_gradient_budget(self):
        for variant in ('vae', 'vq_vae'):
            model = RAD(config(variant))
            out = model(batch(3))
            self.assertEqual(out['auxiliary_rounds'], 0)
            torch.testing.assert_close(out['loss_total'], out['loss_action'])
            model.max_gradient_rounds = 0
            out = model(batch())
            self.assertGreater(out['num_compressions'], 0)
            self.assertEqual(out['auxiliary_rounds'], 0)
            out['loss_total'].backward()
            self.assertTrue(all(p.grad is None for p in model.compression_transformer.parameters()))

    def test_eval_does_not_mutate_codebook(self):
        model = RAD(config('vq_vae')).eval()
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            model(batch())
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_checkpoint_roundtrip_and_cross_variant_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            for variant in ('ae', 'vae', 'vq_vae'):
                cfg = config(variant)
                model = RAD(cfg).eval()
                path = Path(directory) / f'{variant}.pt'
                torch.save(dict(config=cfg, model=model.state_dict(), step=2), path)
                loaded = RAD(cfg).eval()
                loaded.load_pretrained_compression(path)
                context = torch.randn(2, 12, 8)
                torch.testing.assert_close(model._compression_candidate(context)[0], loaded._compression_candidate(context)[0])
                wrong = RAD(config('vq_vae' if variant != 'vq_vae' else 'vae'))
                with self.assertRaisesRegex(ValueError, 'compressor_type'):
                    wrong.load_pretrained_compression(path)

    def test_extra_parameters_use_compression_lr(self):
        for variant in ('vae', 'vq_vae'):
            cfg = config(variant)
            model = RAD(cfg)
            groups = build_rad_optimizer_param_groups(model, cfg)
            ids = {id(p) for group in groups if group['group_name'] == 'compression' for p in group['params']}
            self.assertTrue(all(id(p) in ids for p in model.compression_transformer.parameters()))

    def test_dataset_mapping_audit_and_rejection(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            synthetic_history(directory, cfg)
            audit = audit_darkroom_dataset(cfg, directory)
            self.assertFalse(set(audit['train_groups']) & set(audit['test_groups']))
            self.assertEqual(len(audit['data_sha256']), 64)
            for cls in (ADDataset, RADDataset, CompressionPretrainDataset):
                data = cls(cfg, directory, 'train', 2, 60)
                self.assertEqual(data.group_ids, select_dataset_groups(cfg, 'train'))
                pickle.dumps(data)
            with self.assertRaisesRegex(ValueError, 'rewards disagree'):
                audit_darkroom_dataset({**cfg, 'collection_env_split_seed': 1}, directory)

    def test_model_noise_does_not_change_training_batches(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            synthetic_history(directory, cfg)
            for dataset_cls, loader_fn in ((RADDataset, get_rad_data_loader), (CompressionPretrainDataset, get_pretrain_data_loader)):
                left = dataset_cls(cfg, directory, 'train', 2, 60)
                right = dataset_cls(cfg, directory, 'train', 2, 60)
                # Make differing sampled starts visible in constant toy histories.
                for data in (left, right):
                    data.rewards[:] = np.arange(60)[None, :]
                a = iter(loader_fn(left, 2, cfg))
                torch.randn(99)
                random.seed(123456)
                b = iter(loader_fn(right, 2, cfg))
                for _ in range(4):
                    x = next(a)
                    torch.randn(73)
                    random.random()
                    y = next(b)
                    for key in x:
                        self.assertTrue(torch.equal(x[key], y[key]), key)

    def test_summary_uses_training_seed_replicates(self):
        spec = importlib.util.spec_from_file_location('compressor_eval', ROOT / 'gridworld/scripts/evaluate_compressor_comparison.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rewards = np.stack([np.zeros((8, 100)), np.full((8, 100), 2.)])
        self.assertEqual(module.metrics(rewards)['mean_return'], 1.)
        rows = [dict(variant='ae', latent_mode='deterministic', train_seed=seed,
                     **module.metrics(np.full((20, 8, 100), value))) for seed, value in enumerate([0., 2., 4.])]
        summary = module.aggregate(rows)[0]
        self.assertEqual(summary['n_training_seeds'], 3)
        self.assertEqual(summary['mean'], 2.)
        self.assertEqual(summary['std_over_training_seeds'], 2.)

    def test_amp_retry_reuses_batch_and_latent_noise(self):
        cfg = config('vae')
        model = RAD(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = []
        class AcceleratorProbe:
            optimizer_step_was_skipped = False
            def unwrap_model(self, model):
                return model
            def autocast(self):
                return nullcontext()
            def backward(self, loss):
                losses.append(loss.detach().clone())
                loss.backward()
            def clip_grad_norm_(self, parameters, norm):
                return torch.nn.utils.clip_grad_norm_(parameters, norm)
        accelerator = AcceleratorProbe()
        actual_step = optimizer.step
        def skip_first():
            accelerator.optimizer_step_was_skipped = len(losses) == 1
            if not accelerator.optimizer_step_was_skipped:
                actual_step()
        optimizer.step = skip_first
        output = comparison_optimizer_step(model, batch(), optimizer, accelerator, cfg)
        self.assertEqual(cfg['amp_retries'], 1)
        self.assertEqual(len(losses), 2)
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(output['loss_total']))


if __name__ == '__main__':
    unittest.main()
