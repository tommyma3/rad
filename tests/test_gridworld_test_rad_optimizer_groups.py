import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDWORLD_TEST_ROOT = REPO_ROOT / 'gridworld_test'
sys.path.insert(0, str(GRIDWORLD_TEST_ROOT))

from optimizer_utils import build_rad_optimizer_param_groups  # noqa: E402


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, named_parameters):
        self._named_parameters = named_parameters

    def named_parameters(self):
        return iter(self._named_parameters)


class RADOptimizerParamGroupsTest(unittest.TestCase):
    def test_partitions_parameters_and_applies_group_learning_rates(self):
        ad_parameter = FakeParameter()
        compression_parameter = FakeParameter()
        decoder_parameter = FakeParameter()
        latent_parameter = FakeParameter()
        frozen_parameter = FakeParameter(requires_grad=False)
        model = FakeModel([
            ('ad_transformer._orig_mod.blocks.0.attn.qkv_proj.weight', ad_parameter),
            ('compression_transformer._orig_mod.layers.0.ffn.0.weight', compression_parameter),
            ('reconstruction_decoder.layers.0.ffn.0.weight', decoder_parameter),
            ('latent_gru_gate.weight', latent_parameter),
            ('pred_action.bias', frozen_parameter),
        ])

        groups = build_rad_optimizer_param_groups(model, {
            'lr': 1e-3,
            'ad_lr': 3e-4,
            'compression_lr': 1e-4,
            'latent_lr': 6e-4,
        })
        groups_by_name = {group['group_name']: group for group in groups}

        self.assertEqual(groups_by_name['ad']['lr'], 3e-4)
        self.assertEqual(groups_by_name['ad']['params'], [ad_parameter])
        self.assertEqual(groups_by_name['compression']['lr'], 1e-4)
        self.assertEqual(
            groups_by_name['compression']['params'],
            [compression_parameter, decoder_parameter],
        )
        self.assertEqual(groups_by_name['latent']['lr'], 6e-4)
        self.assertEqual(groups_by_name['latent']['params'], [latent_parameter])

    def test_uses_legacy_lr_as_fallback_for_every_group(self):
        model = FakeModel([
            ('embed_state.weight', FakeParameter()),
            ('compression_transformer.final_norm.weight', FakeParameter()),
            ('null_latent_tokens', FakeParameter()),
        ])

        groups = build_rad_optimizer_param_groups(model, {'lr': 2e-4})

        self.assertEqual([group['lr'] for group in groups], [2e-4, 2e-4, 2e-4])


if __name__ == '__main__':
    unittest.main()
