import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'metaworld'
SPEC = importlib.util.spec_from_file_location(
    'metaworld_sar_model',
    ROOT / 'model' / '__init__.py',
    submodule_search_locations=[str(ROOT / 'model')],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
AD = PACKAGE.AD
RAD = PACKAGE.RAD


def config(model):
    return {
        'model': model,
        'device': torch.device('cpu'),
        'mixed_precision': 'no',
        'n_transit': 4,
        'dim_obs': 3,
        'dim_actions': 2,
        'learn_var': True,
        'tf_n_embd': 8,
        'tf_n_head': 2,
        'tf_n_layer': 1,
        'tf_dim_feedforward': 16,
        'tf_dropout': 0.0,
        'label_smoothing': 0.0,
        'n_compress_tokens': 3,
        'short_memory_keep': 1,
        'compress_n_layers': 1,
        'compress_n_heads': 2,
        'max_context_length': 8,
        'always_use_latent_prefix': True,
        'latent_update_mode': 'gru_gate',
    }


def batch(timesteps=4):
    return {
        'states': torch.randn(2, timesteps, 3),
        'next_states': torch.randn(2, timesteps, 3),
        'actions': torch.randn(2, timesteps, 2).clamp(-1, 1),
        'rewards': torch.randn(2, timesteps),
        'context_lengths': torch.tensor([timesteps, timesteps]),
    }


class FakeVecEnv:
    num_envs = 2

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros((self.num_envs, 3), dtype=np.float32)

    def step(self, actions):
        self.step_count += 1
        dones = np.full(self.num_envs, self.step_count % 2 == 0)
        infos = [
            {
                'success': self.step_count % 2 == 0,
                'terminal_observation': np.ones(3, dtype=np.float32),
            }
            for _ in range(self.num_envs)
        ]
        return (
            np.full((self.num_envs, 3), self.step_count, dtype=np.float32),
            np.ones(self.num_envs, dtype=np.float32),
            dones,
            infos,
        )


class FakeActionSpace:
    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)


class MetaWorldSARTokenTest(unittest.TestCase):
    def test_compression_pretraining_uses_and_trains_enabled_null_prefix(self):
        for enabled, timesteps in ((True, 8), (False, 9)):
            with self.subTest(enabled=enabled):
                model_config = config('RAD')
                model_config['always_use_latent_prefix'] = enabled
                model = RAD(model_config)
                captured_inputs = []
                hook = model.compression_transformer.register_forward_pre_hook(
                    lambda _module, args: captured_inputs.append(args[0].detach().clone())
                )

                output = model(batch(timesteps), pretrain_compression=True)
                hook.remove()
                output['loss_recon'].backward()

                self.assertEqual(captured_inputs[0].shape[1], 10)
                self.assertEqual(len(captured_inputs), 2)
                self.assertEqual(output['num_compressions'], 2)
                self.assertGreater(model.latent_gru_gate.weight.grad.abs().sum().item(), 0.0)
                null_gradient = model.null_latent_tokens.grad
                if enabled:
                    self.assertIsNotNone(null_gradient)
                    self.assertGreater(null_gradient.abs().sum().item(), 0.0)
                else:
                    self.assertIsNone(null_gradient)

    def test_first_recurrent_compression_consumes_null_prefix(self):
        model = RAD(config('RAD'))
        model.eval()
        captured_inputs = []
        hook = model.compression_transformer.register_forward_pre_hook(
            lambda _module, args: captured_inputs.append(args[0].detach().clone())
        )

        model(batch())
        hook.remove()

        self.assertEqual(len(captured_inputs), 1)
        expected_prefix = model.null_latent_tokens.expand(2, -1, -1)
        self.assertTrue(torch.equal(captured_inputs[0][:, :3], expected_prefix))

    def test_ad_uses_three_tokens_and_continuous_action_loss(self):
        model = AD(config('AD'))
        tokens = model._build_token_sequence(
            batch()['states'], batch()['actions'], batch()['rewards']
        )
        self.assertEqual(tokens.shape, (2, 12, 8))
        output = model(batch())
        self.assertTrue(torch.isfinite(output['loss_action']))

    def test_rad_targets_continuous_actions_at_state_positions(self):
        model = RAD(config('RAD'))
        sample = batch()
        tokens, mask, targets = model._build_token_sequence(
            sample['states'], sample['actions'], sample['rewards'], sample['context_lengths']
        )
        self.assertEqual(tokens.shape, (2, 12, 8))
        self.assertEqual(mask.sum().item(), 8)
        self.assertEqual(targets.shape, (2, 12, 2))
        output = model(sample)
        self.assertTrue(torch.isfinite(output['loss_action']))
        self.assertTrue(torch.equal(output['loss_total'], output['loss_action']))
        self.assertNotIn('loss_recon', output)

    def test_ad_and_rad_evaluation_preserve_reward_and_success_outputs(self):
        for model in (AD(config('AD')), RAD(config('RAD'))):
            output = model.evaluate_in_context(FakeVecEnv(), eval_timesteps=4, sample=False)
            self.assertEqual(output['reward_episode'].shape, (2, 2))
            self.assertEqual(output['success'].shape, (2, 2))

    def test_action_bounds_follow_model_device_when_set_after_device_move(self):
        for model in (AD(config('AD')), RAD(config('RAD'))):
            model.to('meta')
            model.set_action_space(FakeActionSpace())
            self.assertEqual(model.action_low.device, model.device)
            self.assertEqual(model.action_high.device, model.device)


if __name__ == '__main__':
    unittest.main()
