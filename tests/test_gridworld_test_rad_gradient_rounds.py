import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / 'gridworld'
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    'gridworld_model',
    ROOT / 'model' / '__init__.py',
    submodule_search_locations=[str(ROOT / 'model')],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
RAD = PACKAGE.RAD


def make_rad_probe(max_gradient_rounds=2, max_compressions=None):
    rad = object.__new__(RAD)
    rad.max_seq_length = 12
    rad.n_compress_tokens = 3
    rad.short_memory_keep_tokens = 3
    rad.always_use_latent_prefix = True
    object.__setattr__(rad, 'null_latent_tokens', torch.zeros(1, 3, 4))
    rad.max_gradient_rounds = max_gradient_rounds
    rad.max_compressions = max_compressions

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

        self.assertEqual(rad._count_compressions_for_sequence(token_count=27), 2)
        _, _, _, _, info = self.roll_tokens(rad, token_count=27)

        self.assertEqual(info['num_compressions'], 2)
        self.assertEqual(decisions, [True, True])

    def test_only_most_recent_rounds_are_trainable(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=2)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=45), 4)
        _, _, _, _, info = self.roll_tokens(rad, token_count=45)

        self.assertEqual(info['num_compressions'], 4)
        self.assertEqual(decisions, [False, False, True, True])

    def test_zero_gradient_rounds_detaches_all_compressions(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=0)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=36), 3)
        _, _, _, _, info = self.roll_tokens(rad, token_count=36)

        self.assertEqual(info['num_compressions'], 3)
        self.assertEqual(decisions, [False, False, False])

    def test_curriculum_limited_count_uses_actual_compressions(self):
        rad, decisions = make_rad_probe(max_gradient_rounds=1, max_compressions=2)

        self.assertEqual(rad._count_compressions_for_sequence(token_count=45), 2)
        _, _, _, _, info = self.roll_tokens(rad, token_count=45)

        self.assertEqual(info['num_compressions'], 2)
        self.assertEqual(decisions, [False, True])

    def test_partial_sar_timestep_is_rejected(self):
        rad, _ = make_rad_probe()
        with self.assertRaisesRegex(ValueError, 'complete s/a/r triplets'):
            self.roll_tokens(rad, token_count=10)


if __name__ == '__main__':
    unittest.main()
