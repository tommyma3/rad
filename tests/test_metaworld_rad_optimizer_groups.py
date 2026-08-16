import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'metaworld'
SPEC = importlib.util.spec_from_file_location('metaworld_optimizer_utils', ROOT / 'optimizer_utils.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_rad_optimizer_param_groups = MODULE.build_rad_optimizer_param_groups
freeze_reconstruction_decoder_for_finetuning = MODULE.freeze_reconstruction_decoder_for_finetuning


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, named_parameters):
        self._named_parameters = named_parameters

    def named_parameters(self):
        return iter(self._named_parameters)


class FakeModule:
    def __init__(self, parameters):
        self._parameters = parameters

    def requires_grad_(self, requires_grad):
        for parameter in self._parameters:
            parameter.requires_grad = requires_grad
        return self


class MetaWorldRADOptimizerGroupsTest(unittest.TestCase):
    def test_reconstruction_decoder_is_frozen_for_finetuning(self):
        decoder_parameter = FakeParameter()
        model = FakeModel([
            ('reconstruction_decoder.layers.0.weight', decoder_parameter),
        ])
        model.reconstruction_decoder = FakeModule([decoder_parameter])

        freeze_reconstruction_decoder_for_finetuning(model)
        groups = build_rad_optimizer_param_groups(model, {'lr': 1e-3})

        self.assertFalse(decoder_parameter.requires_grad)
        self.assertEqual(groups, [])

    def test_continuous_model_parameters_are_partitioned(self):
        ad = FakeParameter()
        compression = FakeParameter()
        latent = FakeParameter()
        groups = build_rad_optimizer_param_groups(
            FakeModel([
                ('embed_state.weight', ad),
                ('compression_transformer._orig_mod.layers.0.weight', compression),
                ('latent_gru_gate.weight', latent),
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
        self.assertEqual(by_name['latent']['params'], [latent])
        self.assertEqual(
            [by_name[name]['lr'] for name in ('ad', 'compression', 'latent')],
            [3e-4, 1e-4, 6e-4],
        )


if __name__ == '__main__':
    unittest.main()
