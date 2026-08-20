from pathlib import Path
import sys
import unittest

import torch


PROJECT = Path(__file__).resolve().parents[1] / "memory_maze"
sys.path.insert(0, str(PROJECT))

from model import AD, RAD
from optimizer_utils import build_rad_optimizer_param_groups, freeze_reconstruction_decoder_for_finetuning


def _base_config():
    return {
        "n_transit": 8,
        "num_actions": 6,
        "tf_n_embd": 32,
        "tf_n_head": 4,
        "tf_n_layer": 2,
        "tf_dim_feedforward": 64,
        "tf_dropout": 0.0,
        "cnn_depth": 4,
        "n_compress_tokens": 6,
        "short_memory_keep": 2,
        "compress_n_heads": 2,
        "compress_n_layers": 1,
        "latent_update_mode": "gru_gate",
        "max_gradient_rounds": 2,
        "max_compressions": None,
        "lr": 3e-4,
        "ad_lr": 3e-4,
        "compression_lr": 1e-4,
        "latent_lr": 3e-4,
    }


def _batch(length=8):
    return {
        "states": torch.randint(0, 256, (2, length, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, 6, (2, length)),
        "rewards": torch.zeros(2, length),
        "dones": torch.zeros(2, length),
        "valid_mask": torch.ones(2, length, dtype=torch.bool),
    }


class MemoryMazeModelTest(unittest.TestCase):
    def test_ad_visual_sar_forward(self):
        output = AD(_base_config())(_batch())
        self.assertEqual(output["loss_action"].ndim, 0)

    def test_rad_compresses_long_visual_history(self):
        model = RAD(_base_config())
        output = model(_batch(length=14))
        self.assertGreater(int(output["num_compressions"]), 0)

    def test_rad_optimizer_groups_and_decoder_freeze(self):
        model = RAD(_base_config())
        freeze_reconstruction_decoder_for_finetuning(model)
        groups = build_rad_optimizer_param_groups(model, _base_config())
        self.assertEqual({group["name"] for group in groups}, {"ad", "compression", "latent"})
        self.assertFalse(any(parameter.requires_grad for parameter in model.reconstruction_decoder.parameters()))


if __name__ == "__main__":
    unittest.main()
