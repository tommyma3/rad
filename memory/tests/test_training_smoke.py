from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from rad_memory.artifacts import TaskHistoryWriter
from rad_memory.counterfactual_cue import compare_episode
from rad_memory.envs import MemoryTaskSpec
from rad_memory.evaluate import evaluate_checkpoint
from rad_memory.model import MODEL
from rad_memory.training import train_compression, train_distillation


def _observation(value: int) -> dict:
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[..., 0] = value % 3
    return {"image": image, "direction": value % 4, "mission": "memory"}


def _episode(length: int, offset: int) -> list[dict]:
    return [
        {
            "observation": _observation(offset + step),
            "action": step % 3,
            "reward": float(step == length - 1),
            "terminated": step == length - 1,
            "truncated": False,
            "next_observation": _observation(offset + step + 1),
            "cue_id": offset % 2,
            "cue_visible": step == 0,
            "decision": step == length - 1,
            "success": step == length - 1,
        }
        for step in range(length)
    ]


class TrainingSmokeTest(unittest.TestCase):
    def test_one_update_for_ad_compressor_and_rad(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=12)
            artifact = root / "data" / "train" / "recurrent_ppo" / f"{spec.task_id}.hdf5"
            with TaskHistoryWriter(artifact, spec, "recurrent_ppo") as writer:
                for index in range(4):
                    writer.write_episode(_episode(12, index), learner_step=index)
            config = {
                "source_algorithm": "recurrent_ppo",
                "num_actions": 7,
                "tf_n_embd": 16,
                "tf_n_head": 2,
                "tf_n_layer": 1,
                "tf_dim_feedforward": 32,
                "tf_dropout": 0.0,
                "tile_embedding_dim": 2,
                "label_smoothing": 0.0,
                "n_transit": 4,
                "n_compress_tokens": 3,
                "short_memory_keep": 1,
                "max_context_length": 12,
                "always_use_latent_prefix": True,
                "compress_n_heads": 1,
                "compress_n_layers": 1,
                "latent_update_mode": "gru_gate",
                "max_gradient_rounds": 1,
                "max_compressions": None,
                "rad_batching_strategy": "compression_buckets",
                "seed": 0,
                "lr": 3e-4,
                "ad_lr": 3e-4,
                "compression_lr": 1e-4,
                "latent_lr": 3e-4,
                "weight_decay": 0.0,
                "train_batch_size": 2,
                "train_steps": 1,
                "warmup_steps": 0,
                "pretrain_batch_size": 2,
                "pretrain_steps": 1,
                "pretrain_lr": 3e-4,
                "pretrain_warmup_steps": 0,
                "checkpoint_interval": 100,
                "summary_interval": 100,
                "mixed_precision": "no",
                "num_workers": 0,
                "dataset_stride": 4,
            }
            pretrain = train_compression(config, root / "data", root / "pretrain")
            ad = train_distillation(config, root / "data", root / "ad", model_kind="AD")
            rad = train_distillation(
                config,
                root / "data",
                root / "rad",
                model_kind="RAD",
                pretrain_checkpoint=pretrain,
            )
            self.assertTrue(pretrain.exists())
            self.assertTrue(ad.exists())
            self.assertTrue(rad.exists())
            records, summary = evaluate_checkpoint(
                ad,
                MemoryTaskSpec("MiniGrid-MemoryS7-v0", 1000, "test", horizon=5),
                2,
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(summary["episodes"], 2)
            rad_checkpoint = torch.load(rad, map_location="cpu", weights_only=False)
            model = MODEL["RAD"](rad_checkpoint["config"])
            model.load_state_dict(rad_checkpoint["model"])
            model.eval()
            with h5py.File(artifact, "r") as handle:
                first_episode = handle["episodes"][sorted(handle["episodes"].keys())[0]]
                counterfactual = compare_episode(model, first_episode)
            self.assertIsNotNone(counterfactual)
            self.assertIn("probability_l1", counterfactual)


if __name__ == "__main__":
    unittest.main()
