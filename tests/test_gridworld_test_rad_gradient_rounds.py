import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDWORLD_ROOT = REPO_ROOT / 'gridworld'
sys.path.insert(0, str(GRIDWORLD_ROOT))

from model.compressed_ad import RAD  # noqa: E402
from optimizer_utils import (  # noqa: E402
    build_rad_optimizer_param_groups,
    configure_compression_pretraining,
    configure_rad_finetuning,
)


def model_config():
    return {
        'device': torch.device('cpu'),
        'mixed_precision': 'no',
        'n_transit': 4,
        'grid_size': 5,
        'num_actions': 5,
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


def make_rad_probe(max_gradient_rounds=2, max_compressions=None):
    rad = object.__new__(RAD)
    rad.max_seq_length = 12
    rad.n_compress_tokens = 3
    rad.short_memory_keep_tokens = 3
    rad.always_use_latent_prefix = True
    rad.max_gradient_rounds = max_gradient_rounds
    rad.max_compressions = max_compressions
    rad.null_latent_tokens = torch.zeros(1, rad.n_compress_tokens, 4)

    decisions = []

    def fake_compress_sequence(context_embed, allow_gradient, old_latent_tokens=None):
        decisions.append(allow_gradient)
        batch_size, _, d_model = context_embed.shape
        return torch.zeros(batch_size, rad.n_compress_tokens, d_model)

    rad._compress_sequence = fake_compress_sequence
    return rad, decisions


class RADRecentGradientRoundsTest(unittest.TestCase):
    def roll_tokens(self, rad, token_count):
        tokens = torch.zeros(1, token_count, 4)
        return rad._roll_context_into_memory(tokens)

    def test_no_compression_has_no_gradient_decisions(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=2)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=9), 0)
        _, _, _, _, info = self.roll_tokens(rad, token_count=9)

        self.assertEqual(info['num_compressions'], 0)
        self.assertEqual(decisions, [])

    def test_all_rounds_trainable_when_total_is_within_limit(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=3)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=17), 2)
        _, _, _, _, info = self.roll_tokens(rad, token_count=17)

        self.assertEqual(info['num_compressions'], 2)
        self.assertEqual(decisions, [True, True])

    def test_only_most_recent_rounds_are_trainable(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=2)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=31), 4)
        _, _, _, _, info = self.roll_tokens(rad, token_count=31)

        self.assertEqual(info['num_compressions'], 4)
        self.assertEqual(decisions, [False, False, True, True])

    def test_zero_gradient_rounds_detaches_all_compressions(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=0)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=24), 3)
        _, _, _, _, info = self.roll_tokens(rad, token_count=24)

        self.assertEqual(info['num_compressions'], 3)
        self.assertEqual(decisions, [False, False, False])

    def test_curriculum_limited_count_uses_actual_compressions(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=1, max_compressions=2)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=31), 2)
        _, _, _, _, info = self.roll_tokens(rad, token_count=31)

        self.assertEqual(info['num_compressions'], 2)
        self.assertEqual(decisions, [False, True])


class RADTrainabilityTest(unittest.TestCase):
    def test_pretraining_updates_null_and_query_tokens(self):
        model = RAD(model_config())
        params = configure_compression_pretraining(model)
        sample = {
            'states': torch.randint(0, 5, (2, 4, 2)),
            'actions': torch.nn.functional.one_hot(
                torch.randint(0, 5, (2, 4)), num_classes=5
            ),
            'rewards': torch.randn(2, 4),
        }

        model.forward_pretrain_compression(sample)['loss_recon'].backward()

        self.assertTrue(any(parameter is model.null_latent_tokens for parameter in params))
        for parameter in (
            model.null_latent_tokens,
            model.compression_transformer.compress_queries,
            model.reconstruction_decoder.position_queries,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_finetuning_uses_exact_allowlist(self):
        model = RAD(model_config())
        configure_rad_finetuning(model)
        groups = build_rad_optimizer_param_groups(model, {'lr': 1e-3})
        optimized = {id(parameter) for group in groups for parameter in group['params']}
        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

        self.assertEqual(optimized, trainable)
        self.assertFalse(model.null_latent_tokens.requires_grad)
        self.assertFalse(model.compression_transformer.compress_queries.requires_grad)
        self.assertFalse(model.reconstruction_decoder.position_queries.requires_grad)
        self.assertTrue(model.latent_gru_gate.weight.requires_grad)
        self.assertTrue(model.latent_gru_candidate.weight.requires_grad)
        self.assertFalse(model.latent_multiplicative_gate.weight.requires_grad)

    def test_only_selected_latent_update_module_is_trainable(self):
        expected = {
            'replace': set(),
            'residual': {'latent_residual_norm'},
            'multiplicative_gate': {'latent_multiplicative_gate'},
            'gru_gate': {'latent_gru_gate', 'latent_gru_candidate'},
        }
        latent_modules = set().union(*expected.values())

        for mode, expected_trainable in expected.items():
            config = model_config()
            config['latent_update_mode'] = mode
            model = RAD(config)
            configure_rad_finetuning(model)
            actual = {
                name
                for name in latent_modules
                if any(parameter.requires_grad for parameter in getattr(model, name).parameters())
            }
            self.assertEqual(actual, expected_trainable)

    def test_finetuning_backward_trains_core_compressor_and_gate_only(self):
        model = RAD(model_config())
        configure_rad_finetuning(model)
        sample = {
            'states': torch.randint(0, 5, (2, 6, 2)),
            'actions': torch.nn.functional.one_hot(
                torch.randint(0, 5, (2, 6)), num_classes=5
            ),
            'rewards': torch.randn(2, 6),
            'context_lengths': torch.tensor([6, 6]),
        }

        model(sample)['loss_action'].backward()

        self.assertTrue(any(parameter.grad is not None for parameter in model.ad_transformer.parameters()))
        self.assertTrue(any(
            parameter.grad is not None
            for name, parameter in model.compression_transformer.named_parameters()
            if name != 'compress_queries'
        ))
        self.assertTrue(any(parameter.grad is not None for parameter in model.latent_gru_gate.parameters()))
        self.assertIsNone(model.null_latent_tokens.grad)
        self.assertIsNone(model.compression_transformer.compress_queries.grad)
        self.assertTrue(all(
            parameter.grad is None for parameter in model.reconstruction_decoder.parameters()
        ))


if __name__ == '__main__':
    unittest.main()
