import importlib.util
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_package(name, root):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        name,
        root / 'model' / '__init__.py',
        submodule_search_locations=[str(root / 'model')],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


def load_optimizer_utils(name, root):
    spec = importlib.util.spec_from_file_location(name, root / 'optimizer_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRIDWORLD_ROOT = REPO_ROOT / 'gridworld'
METAWORLD_ROOT = REPO_ROOT / 'metaworld'
GRIDWORLD = load_package('gridworld_pretrain_model', GRIDWORLD_ROOT)
METAWORLD = load_package('metaworld_pretrain_model', METAWORLD_ROOT)
GRIDWORLD_OPT = load_optimizer_utils('gridworld_pretrain_optimizer', GRIDWORLD_ROOT)
METAWORLD_OPT = load_optimizer_utils('metaworld_pretrain_optimizer', METAWORLD_ROOT)


def gridworld_config(always_use_latent_prefix):
    return {
        'device': torch.device('cpu'),
        'mixed_precision': 'no',
        'n_transit': 4,
        'grid_size': 5,
        'num_actions': 4,
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
        'always_use_latent_prefix': always_use_latent_prefix,
    }


def metaworld_config(always_use_latent_prefix):
    return {
        'device': torch.device('cpu'),
        'mixed_precision': 'no',
        'n_transit': 4,
        'dim_obs': 3,
        'dim_actions': 2,
        'learn_var': False,
        'tf_n_embd': 8,
        'tf_n_head': 2,
        'tf_n_layer': 1,
        'tf_dim_feedforward': 16,
        'tf_dropout': 0.0,
        'n_compress_tokens': 3,
        'short_memory_keep': 1,
        'compress_n_layers': 1,
        'compress_n_heads': 2,
        'max_context_length': 8,
        'always_use_latent_prefix': always_use_latent_prefix,
    }


def gridworld_batch(timesteps):
    action_ids = torch.randint(0, 4, (2, timesteps))
    return {
        'states': torch.randint(0, 5, (2, timesteps, 2)),
        'actions': torch.nn.functional.one_hot(action_ids, num_classes=4).float(),
        'rewards': torch.randn(2, timesteps),
    }


def metaworld_batch(timesteps):
    return {
        'states': torch.randn(2, timesteps, 3),
        'actions': torch.randn(2, timesteps, 2),
        'rewards': torch.randn(2, timesteps),
    }


class CompressionPretrainingNullPrefixTest(unittest.TestCase):
    implementations = (
        (GRIDWORLD.RAD, GRIDWORLD_OPT, gridworld_config, gridworld_batch),
        (METAWORLD.RAD, METAWORLD_OPT, metaworld_config, metaworld_batch),
    )

    def test_full_first_compression_input_is_reconstructed_and_trains_null_prefix(self):
        for model_class, optimizer_utils, config_fn, batch_fn in self.implementations:
            with self.subTest(model=model_class.__module__):
                model = model_class(config_fn(True))
                compressor_inputs = []
                decoder_target_lengths = []
                compressor_hook = model.compression_transformer.register_forward_pre_hook(
                    lambda _module, args: compressor_inputs.append(args[0].detach().clone())
                )
                decoder_hook = model.reconstruction_decoder.register_forward_pre_hook(
                    lambda _module, args: decoder_target_lengths.append(args[1])
                )

                output = model.forward_pretrain_compression(batch_fn(3))
                compressor_hook.remove()
                decoder_hook.remove()
                output['loss_recon'].backward()

                self.assertEqual(compressor_inputs[0].shape[1], 12)
                self.assertEqual(decoder_target_lengths, [12])
                expected_null = model.null_latent_tokens.detach().expand(2, -1, -1)
                self.assertTrue(torch.equal(compressor_inputs[0][:, :3], expected_null))
                self.assertIsNotNone(model.null_latent_tokens.grad)
                self.assertGreater(model.null_latent_tokens.grad.abs().sum().item(), 0.0)
                parameters = optimizer_utils.build_compression_pretraining_parameters(model)
                self.assertTrue(any(item is model.null_latent_tokens for item in parameters))

    def test_disabled_prefix_preserves_raw_only_pretraining(self):
        for model_class, optimizer_utils, config_fn, batch_fn in self.implementations:
            with self.subTest(model=model_class.__module__):
                model = model_class(config_fn(False))
                compressor_lengths = []
                hook = model.compression_transformer.register_forward_pre_hook(
                    lambda _module, args: compressor_lengths.append(args[0].shape[1])
                )
                model.forward_pretrain_compression(batch_fn(4))['loss_recon'].backward()
                hook.remove()

                self.assertEqual(compressor_lengths, [12])
                self.assertIsNone(model.null_latent_tokens.grad)
                parameters = optimizer_utils.build_compression_pretraining_parameters(model)
                self.assertFalse(any(item is model.null_latent_tokens for item in parameters))

    def test_online_first_compression_uses_the_same_null_prefix_region(self):
        for model_class, _optimizer_utils, config_fn, batch_fn in self.implementations:
            with self.subTest(model=model_class.__module__):
                model = model_class(config_fn(True))
                sample = batch_fn(4)
                tokens, _, _ = model._build_token_sequence(
                    sample['states'], sample['actions'], sample['rewards']
                )
                compressor_inputs = []
                hook = model.compression_transformer.register_forward_pre_hook(
                    lambda _module, args: compressor_inputs.append(args[0].detach().clone())
                )
                model._roll_context_into_memory(tokens)
                hook.remove()

                self.assertGreaterEqual(len(compressor_inputs), 1)
                expected_null = model.null_latent_tokens.detach().expand(2, -1, -1)
                self.assertTrue(torch.equal(compressor_inputs[0][:, :3], expected_null))


if __name__ == '__main__':
    unittest.main()
