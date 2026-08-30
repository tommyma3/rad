import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'metaworld'
SPEC = importlib.util.spec_from_file_location('metaworld_optimizer_utils', ROOT / 'optimizer_utils.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_rad_optimizer_param_groups = MODULE.build_rad_optimizer_param_groups
build_compression_pretraining_parameters = MODULE.build_compression_pretraining_parameters
freeze_reconstruction_decoder_for_finetuning = MODULE.freeze_reconstruction_decoder_for_finetuning


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def requires_grad_(self, requires_grad):
        self.requires_grad = requires_grad
        return self


class FakeModel:
    def __init__(self, named_parameters):
        self._named_parameters = named_parameters

    def named_parameters(self):
        return iter(self._named_parameters)

    def parameters(self):
        return iter(self._parameters)

    def requires_grad_(self, requires_grad):
        for parameter in self._parameters:
            parameter.requires_grad = requires_grad
        return self


class FakeModule:
    def __init__(self, parameters):
        self._parameters = parameters

    def requires_grad_(self, requires_grad):
        for parameter in self._parameters:
            parameter.requires_grad = requires_grad
        return self

    def parameters(self):
        return iter(self._parameters)


class MetaWorldRADOptimizerGroupsTest(unittest.TestCase):
    def test_null_latent_tokens_only_join_enabled_prefix_pretraining(self):
        module_parameters = [FakeParameter() for _ in range(5)]
        type_embedding = FakeParameter()
        null_latent_tokens = FakeParameter()
        model = FakeModel([])
        (
            model.compression_transformer,
            model.reconstruction_decoder,
            model.embed_state,
            model.embed_action,
            model.embed_reward,
        ) = [FakeModule([parameter]) for parameter in module_parameters]
        model.type_embedding = type_embedding
        model.null_latent_tokens = null_latent_tokens
        model.latent_update_mode = 'replace'
        model._parameters = module_parameters + [type_embedding, null_latent_tokens]

        model.always_use_latent_prefix = True
        enabled = build_compression_pretraining_parameters(model)
        model.always_use_latent_prefix = False
        disabled = build_compression_pretraining_parameters(model)

        self.assertIn(null_latent_tokens, enabled)
        self.assertNotIn(null_latent_tokens, disabled)
        self.assertIn(type_embedding, enabled)
        self.assertIn(type_embedding, disabled)

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
