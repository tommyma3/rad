import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / 'metaworld'
SPEC = importlib.util.spec_from_file_location(
    'metaworld_model',
    ROOT / 'model' / '__init__.py',
    submodule_search_locations=[str(ROOT / 'model')],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
RAD = PACKAGE.RAD


def make_probe(max_gradient_rounds=2, max_compressions=None):
    rad = object.__new__(RAD)
    rad.max_seq_length = 12
    rad.n_compress_tokens = 3
    rad.short_memory_keep_tokens = 3
    rad.always_use_latent_prefix = True
    object.__setattr__(rad, 'null_latent_tokens', torch.zeros(1, 3, 4))
    rad.max_gradient_rounds = max_gradient_rounds
    rad.max_compressions = max_compressions
    decisions = []

    def compress(context, allow_gradient, old_latent_tokens=None):
        decisions.append(allow_gradient)
        return torch.zeros(context.shape[0], rad.n_compress_tokens, context.shape[-1])

    rad._compress_sequence = compress
    return rad, decisions


class MetaWorldRADGradientRoundsTest(unittest.TestCase):
    def test_only_recent_compressions_keep_gradients(self):
        rad, decisions = make_probe(max_gradient_rounds=2)
        tokens = torch.zeros(1, 45, 4)
        _, _, _, _, info = rad._roll_context_into_memory(tokens)
        self.assertEqual(info['num_compressions'], 4)
        self.assertEqual(decisions, [False, False, True, True])

    def test_curriculum_count_controls_gradient_boundary(self):
        rad, decisions = make_probe(max_gradient_rounds=1, max_compressions=2)
        tokens = torch.zeros(1, 45, 4)
        _, _, _, _, info = rad._roll_context_into_memory(tokens)
        self.assertEqual(info['num_compressions'], 2)
        self.assertEqual(decisions, [False, True])

    def test_partial_sar_timestep_is_rejected(self):
        rad, _ = make_probe()
        with self.assertRaisesRegex(ValueError, 'complete s/a/r triplets'):
            rad._roll_context_into_memory(torch.zeros(1, 10, 4))


if __name__ == '__main__':
    unittest.main()

