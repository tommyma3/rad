import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / 'gridworld'
sys.path.insert(0, str(ROOT))
MODEL_SPEC = importlib.util.spec_from_file_location(
    'gridworld_recurrent_model',
    ROOT / 'model' / '__init__.py',
    submodule_search_locations=[str(ROOT / 'model')],
)
MODEL_PACKAGE = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = MODEL_PACKAGE
MODEL_SPEC.loader.exec_module(MODEL_PACKAGE)
RAD = MODEL_PACKAGE.RAD

OPTIMIZER_SPEC = importlib.util.spec_from_file_location(
    'gridworld_recurrent_optimizer_utils',
    ROOT / 'optimizer_utils.py',
)
OPTIMIZER_MODULE = importlib.util.module_from_spec(OPTIMIZER_SPEC)
OPTIMIZER_SPEC.loader.exec_module(OPTIMIZER_MODULE)
build_compression_pretraining_parameters = OPTIMIZER_MODULE.build_compression_pretraining_parameters


def config(always_use_latent_prefix):
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
        'n_compress_tokens': 3,
        'short_memory_keep': 1,
        'compress_n_layers': 1,
        'compress_n_heads': 1,
        'max_context_length': 12,
        'max_gradient_rounds': 2,
        'max_compressions': None,
        'always_use_latent_prefix': always_use_latent_prefix,
        'latent_update_mode': 'gru_gate',
        'label_smoothing': 0.0,
    }


def batch(timesteps):
    states = torch.randint(0, 5, (2, timesteps, 2))
    next_states = torch.randint(0, 5, (2, timesteps, 2))
    action_ids = torch.randint(0, 4, (2, timesteps))
    return {
        'states': states,
        'actions': torch.nn.functional.one_hot(action_ids, num_classes=4).float(),
        'rewards': torch.randn(2, timesteps),
        'next_states': next_states,
    }


class GridworldRecurrentPretrainingTest(unittest.TestCase):
    def test_checkpoint_restores_recurrent_update_and_null_memory(self):
        source = RAD(config(True))
        with torch.no_grad():
            source.null_latent_tokens.fill_(0.25)
            source.latent_gru_gate.weight.fill_(0.5)
            source.latent_gru_candidate.bias.fill_(0.75)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'pretrain.pt'
            torch.save({
                'compression_pretrain_contract': 'recurrent-transition-v1',
                'model': source.state_dict(),
            }, checkpoint_path)
            target = RAD(config(True))
            target.load_pretrained_compression(checkpoint_path)

        self.assertTrue(torch.equal(target.null_latent_tokens, source.null_latent_tokens))
        self.assertTrue(torch.equal(target.latent_gru_gate.weight, source.latent_gru_gate.weight))
        self.assertTrue(torch.equal(target.latent_gru_candidate.bias, source.latent_gru_candidate.bias))

    def test_replays_two_transitions_and_trains_only_enabled_null_memory(self):
        for enabled, timesteps in ((True, 8), (False, 9)):
            with self.subTest(enabled=enabled):
                model = RAD(config(enabled))
                optimizer_parameters = build_compression_pretraining_parameters(model)
                captured_inputs = []
                hook = model.compression_transformer.register_forward_pre_hook(
                    lambda _module, args: captured_inputs.append(args[0].detach().clone())
                )

                output = model(batch(timesteps), pretrain_compression=True)
                hook.remove()
                output['loss_recon'].backward()

                self.assertEqual(output['num_compressions'], 2)
                self.assertEqual(len(captured_inputs), 2)
                self.assertTrue(all(item.shape[1] <= model.max_seq_length for item in captured_inputs))
                self.assertTrue(any(
                    parameter is model.latent_gru_gate.weight
                    for parameter in optimizer_parameters
                ))
                self.assertGreater(model.latent_gru_gate.weight.grad.abs().sum().item(), 0.0)
                if enabled:
                    self.assertTrue(any(
                        parameter is model.null_latent_tokens
                        for parameter in optimizer_parameters
                    ))
                    self.assertGreater(model.null_latent_tokens.grad.abs().sum().item(), 0.0)
                else:
                    self.assertFalse(any(
                        parameter is model.null_latent_tokens
                        for parameter in optimizer_parameters
                    ))
                    self.assertIsNone(model.null_latent_tokens.grad)


if __name__ == '__main__':
    unittest.main()
