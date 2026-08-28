import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDWORLD_ROOT = REPO_ROOT / 'gridworld'
sys.path.insert(0, str(GRIDWORLD_ROOT))

from optimizer_utils import build_rad_optimizer_param_groups  # noqa: E402


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class FakeModel:
    latent_update_mode = 'gru_gate'

    def __init__(self, named_parameters):
        self._named_parameters = named_parameters

    def named_parameters(self):
        return iter(self._named_parameters)


class RADOptimizerParamGroupsTest(unittest.TestCase):
    def test_partitions_only_allowed_parameters(self):
        ad_parameter = FakeParameter()
        compression_parameter = FakeParameter()
        gate_parameter = FakeParameter()
        frozen_query = FakeParameter(requires_grad=False)
        frozen_null = FakeParameter(requires_grad=False)
        model = FakeModel([
            ('ad_transformer._orig_mod.blocks.0.attn.qkv_proj.weight', ad_parameter),
            ('compression_transformer._orig_mod.layers.0.ffn.0.weight', compression_parameter),
            ('latent_gru_gate.weight', gate_parameter),
            ('compression_transformer.compress_queries', frozen_query),
            ('null_latent_tokens', frozen_null),
        ])

        groups = build_rad_optimizer_param_groups(model, {
            'lr': 1e-3,
            'ad_lr': 3e-4,
            'compression_lr': 1e-4,
            'latent_lr': 6e-4,
        })
        groups_by_name = {group['group_name']: group for group in groups}

        self.assertEqual(groups_by_name['ad']['params'], [ad_parameter])
        self.assertEqual(groups_by_name['compression']['params'], [compression_parameter])
        self.assertEqual(groups_by_name['latent']['params'], [gate_parameter])
        self.assertEqual(
            [groups_by_name[name]['lr'] for name in ('ad', 'compression', 'latent')],
            [3e-4, 1e-4, 6e-4],
        )

    def test_rejects_frozen_token_if_marked_trainable(self):
        model = FakeModel([
            ('compression_transformer.compress_queries', FakeParameter()),
        ])

        with self.assertRaisesRegex(ValueError, 'unexpectedly marked trainable'):
            build_rad_optimizer_param_groups(model, {'lr': 2e-4})

    def test_rejects_unknown_trainable_parameter(self):
        model = FakeModel([('latent_type_embedding', FakeParameter())])

        with self.assertRaisesRegex(ValueError, 'unexpectedly marked trainable'):
            build_rad_optimizer_param_groups(model, {'lr': 2e-4})


if __name__ == '__main__':
    unittest.main()
