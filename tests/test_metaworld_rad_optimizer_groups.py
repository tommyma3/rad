import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'metaworld'
SPEC = importlib.util.spec_from_file_location('metaworld_optimizer_utils', ROOT / 'optimizer_utils.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_rad_optimizer_param_groups = MODULE.build_rad_optimizer_param_groups


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class FakeModel:
    latent_update_mode = 'gru_gate'

    def __init__(self, named_parameters):
        self._named_parameters = named_parameters

    def named_parameters(self):
        return iter(self._named_parameters)


class MetaWorldRADOptimizerGroupsTest(unittest.TestCase):
    def test_continuous_model_parameters_are_partitioned(self):
        ad = FakeParameter()
        compression = FakeParameter()
        gate = FakeParameter()
        groups = build_rad_optimizer_param_groups(
            FakeModel([
                ('embed_state.weight', ad),
                ('compression_transformer._orig_mod.layers.0.weight', compression),
                ('latent_gru_candidate.weight', gate),
                ('null_latent_tokens', FakeParameter(requires_grad=False)),
                ('reconstruction_decoder.position_queries', FakeParameter(requires_grad=False)),
            ]),
            {
                'lr': 1e-3,
                'ad_lr': 3e-4,
                'compression_lr': 1e-4,
                'latent_lr': 6e-4,
            },
        )
        by_name = {group['group_name']: group for group in groups}
        self.assertEqual(by_name['ad']['params'], [ad])
        self.assertEqual(by_name['compression']['params'], [compression])
        self.assertEqual(by_name['latent']['params'], [gate])

    def test_frozen_queries_cannot_enter_optimizer(self):
        model = FakeModel([
            ('compression_transformer.compress_queries', FakeParameter()),
        ])

        with self.assertRaisesRegex(ValueError, 'unexpectedly marked trainable'):
            build_rad_optimizer_param_groups(model, {'lr': 1e-3})

    def test_non_allowlisted_latent_parameter_is_rejected(self):
        model = FakeModel([('latent_type_embedding', FakeParameter())])

        with self.assertRaisesRegex(ValueError, 'unexpectedly marked trainable'):
            build_rad_optimizer_param_groups(model, {'lr': 1e-3})


if __name__ == '__main__':
    unittest.main()
