from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

import rad_memory.evaluate as evaluate_module
from rad_memory.artifacts import TaskHistoryWriter
from rad_memory.dataset import ADDataset, RADDataset, collate_trajectories
from rad_memory.effective_length import configured_effective_length, pad_transition_tail
from rad_memory.envs import MemoryTaskSpec, make_memory_env
from rad_memory.evaluate import evaluate_checkpoint
from rad_memory.model import AD, RAD
from rad_memory.task_pool import generate_pool
from rad_memory.train_task_pool import OnlineHistory
from rad_memory.training import train_distillation
from test_artifact_dataset_contract import _episode
from test_models import _config


def _item_from_episode(length: int) -> dict:
    steps = _episode(length)
    return {
        "images": np.stack([step["observation"]["image"] for step in steps]),
        "directions": np.asarray([step["observation"]["direction"] for step in steps]),
        "actions": np.asarray([step["action"] for step in steps], dtype=np.int8),
        "rewards": np.asarray([step["reward"] for step in steps], dtype=np.float32),
        "terminated": np.asarray([step["terminated"] for step in steps], dtype=np.bool_),
        "truncated": np.asarray([step["truncated"] for step in steps], dtype=np.bool_),
        "cue_ids": np.asarray([step["cue_id"] for step in steps], dtype=np.int8),
        "cue_visible": np.asarray([step["cue_visible"] for step in steps], dtype=np.bool_),
        "decision": np.asarray([step["decision"] for step in steps], dtype=np.bool_),
        "success": np.asarray([step["success"] for step in steps], dtype=np.bool_),
        "learner_steps": np.zeros(length, dtype=np.int64),
    }


def _write_episode_artifacts(root: Path) -> None:
    spec = MemoryTaskSpec("MiniGrid-MemoryS13-v0", 3, "train", horizon=8)
    path = root / "train" / "recurrent_ppo" / f"{spec.task_id}.hdf5"
    with TaskHistoryWriter(path, spec, "recurrent_ppo") as writer:
        writer.write_episode(_episode(5), learner_step=100)
        writer.write_episode(_episode(12), learner_step=200)


class EffectiveLengthConfigTest(unittest.TestCase):
    def test_configured_effective_length(self):
        self.assertIsNone(configured_effective_length({}))
        self.assertIsNone(configured_effective_length({"effective_episode_length": None}))
        self.assertEqual(configured_effective_length({"effective_episode_length": "10"}), 10)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            configured_effective_length({"effective_episode_length": 0})

    def test_ad_training_rejects_effective_length_above_n_transit(self):
        config = _config() | {"effective_episode_length": 7}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "effective_episode_length"):
                train_distillation(config, Path(directory) / "data", Path(directory) / "run",
                                   model_kind="AD")


class PadTransitionTailTest(unittest.TestCase):
    def test_zero_count_is_identity(self):
        item = _item_from_episode(3)
        self.assertIs(pad_transition_tail(item, 0), item)

    def test_appends_noop_filler_with_terminated_flags(self):
        item = _item_from_episode(3)
        padded = pad_transition_tail(item, 2)
        self.assertEqual(len(padded["actions"]), 5)
        np.testing.assert_array_equal(padded["actions"], [0, 1, 2, 0, 0])
        np.testing.assert_array_equal(padded["rewards"], [0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            padded["terminated"], [False, False, True, True, True]
        )
        np.testing.assert_array_equal(
            padded["truncated"], [False, False, False, False, False]
        )
        np.testing.assert_array_equal(padded["cue_ids"], [0, 0, 0, -1, -1])
        np.testing.assert_array_equal(padded["cue_visible"], [True, False, False, False, False])
        np.testing.assert_array_equal(padded["decision"], [False, False, True, False, False])
        np.testing.assert_array_equal(padded["success"], [False, False, True, False, False])
        for index in (3, 4):
            np.testing.assert_array_equal(padded["images"][index], item["images"][2])
            self.assertEqual(padded["directions"][index], item["directions"][2])
        np.testing.assert_array_equal(padded["learner_steps"], [0, 0, 0, 1, 2])


class EffectiveLengthDatasetTest(unittest.TestCase):
    def test_short_episode_padded_and_long_episode_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_episode_artifacts(root)
            config = {
                "n_transit": 6,
                "dataset_stride": 3,
                "source_algorithm": "recurrent_ppo",
                "effective_episode_length": 10,
            }
            with ADDataset(config, root, "train") as dataset:
                items = [dataset[index] for index in range(len(dataset))]
            lengths = sorted(len(item["actions"]) for item in items)
            self.assertEqual(lengths, [6, 6, 6, 10])
            item = next(item for item in items if len(item["actions"]) == 10)
            self.assertEqual(int(item["context_length"]), 10)
            np.testing.assert_array_equal(item["actions"][-5:], [0, 0, 0, 0, 0])
            np.testing.assert_array_equal(
                item["terminated"], [False, False, False, False, True, True, True, True, True, True]
            )
            np.testing.assert_array_equal(item["images"][5], item["images"][4])

    def test_legacy_behavior_without_config_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_episode_artifacts(root)
            config = {
                "n_transit": 6,
                "dataset_stride": 3,
                "source_algorithm": "recurrent_ppo",
            }
            with ADDataset(config, root, "train") as dataset:
                for index, window in enumerate(dataset.windows):
                    item = dataset[index]
                    self.assertEqual(len(item["actions"]), window[2] - window[1])
                    self.assertEqual(int(item["context_length"]), window[2] - window[1])

    def test_models_forward_on_padded_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_episode_artifacts(root)
            data_config = {
                "n_transit": 6,
                "dataset_stride": 3,
                "source_algorithm": "recurrent_ppo",
                "effective_episode_length": 6,
            }
            with ADDataset(data_config, root, "train") as dataset:
                items = [dataset[index] for index in range(len(dataset))]
            batch = collate_trajectories(items)
            for kind in (AD, RAD):
                model = kind(_config(kind.__name__)).eval()
                with torch.inference_mode():
                    output = model(batch)
                self.assertTrue(torch.isfinite(output["loss_total"]))

    def test_rad_bucket_window_includes_padded_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_episode_artifacts(root)
            config = {
                "n_transit": 4,
                "short_memory_keep": 1,
                "max_context_length": 16,
                "source_algorithm": "recurrent_ppo",
                "effective_episode_length": 10,
            }
            with RADDataset(config, root, "train") as dataset:
                index = next(i for i, window in enumerate(dataset.windows) if window[2] == 5)
                item = dataset[(index, 2)]
                self.assertEqual(len(item["actions"]), 10)
                np.testing.assert_array_equal(item["actions"][5:], np.zeros(5))
                self.assertTrue(bool(item["terminated"][-1]))
                batch = collate_trajectories([item])
                self.assertEqual(batch["valid_mask"].sum().item(), 10)
                model = RAD(_config("RAD")).eval()
                with torch.inference_mode():
                    output = model(batch)
                self.assertTrue(torch.isfinite(output["loss_total"]))

    def test_task_scope_pads_each_stream_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=3),
                                 2, 0.5, 0)
            manifest = root / "pool.json"
            manifest.write_text(json.dumps(pool), encoding="utf-8")
            spec = MemoryTaskSpec.from_dict(next(t for t in pool["tasks"] if t["split"] == "train"))
            path = root / "train" / "recurrent_ppo" / "run-0.hdf5"
            provenance = {"manifest_fingerprint": pool["fingerprint"], "source_seed": 0,
                          "run_id": "0", "stream_id": 0, "history_kind": "online_training"}
            with TaskHistoryWriter(path, spec, "recurrent_ppo", provenance) as writer:
                env = OnlineHistory(make_memory_env(spec), writer)
                try:
                    for _ in range(4):
                        env.reset()
                        for _ in range(3):
                            env.step(0)
                    writer.handle.attrs["collection_complete"] = True
                    writer.handle.attrs["source_converged"] = True
                finally:
                    env.close()
            config = {
                "n_transit": 6,
                "max_context_length": 20,
                "source_algorithm": "recurrent_ppo",
                "history_scope": "task",
                "task_manifest": str(manifest),
                "dataset_stride": 1,
                "effective_episode_length": 5,
            }
            with ADDataset(config, root, "train") as dataset:
                first = dataset[next(i for i, window in enumerate(dataset.windows) if window[2] == 3)]
                self.assertEqual(first["learner_steps"].tolist(), [1, 2, 3, 4, 5])
                np.testing.assert_array_equal(
                    first["terminated"], [False, False, False, True, True]
                )
                np.testing.assert_array_equal(
                    first["truncated"], [False, False, True, False, False]
                )
                np.testing.assert_array_equal(first["images"][3], first["images"][2])
                self.assertEqual(int(first["context_length"]), 5)
                second = dataset[next(i for i, window in enumerate(dataset.windows) if window[2] == 6)]
                self.assertEqual(second["learner_steps"].tolist(), [1, 2, 3, 4, 5, 4, 5, 6, 7, 8])
                self.assertEqual(len(second["actions"]), 10)
                self.assertEqual(int(second["actions"][-1]), 0)
                self.assertTrue(bool(second["terminated"][-1]))
                np.testing.assert_array_equal(second["images"][3], second["images"][2])
                np.testing.assert_array_equal(second["images"][8], second["images"][7])
                np.testing.assert_array_equal(second["images"][9], second["images"][7])
            with h5py.File(path, "r") as handle:
                for key in handle["episodes"]:
                    self.assertEqual(handle["episodes"][key]["actions"].shape[0], 3)


class EffectiveLengthInferenceTest(unittest.TestCase):
    def test_evaluate_feeds_post_terminal_padding(self):
        class _CountingAD(AD):
            observe_calls = 0

            def observe(self, context, *args, **kwargs):
                type(self).observe_calls += 1
                return super().observe(context, *args, **kwargs)

        pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=3),
                             2, 0.5, 0)
        spec = MemoryTaskSpec.from_dict(next(t for t in pool["tasks"] if t["split"] == "train"))
        config = _config() | {"history_scope": "task", "effective_episode_length": 5}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "ckpt.pt"
            torch.save({"step": 0, "config": config, "model": AD(config).state_dict()},
                       checkpoint_path)
            _CountingAD.observe_calls = 0
            original = evaluate_module.MODEL["AD"]
            evaluate_module.MODEL["AD"] = _CountingAD
            try:
                records, summary = evaluate_checkpoint(checkpoint_path, spec, 2)
            finally:
                evaluate_module.MODEL["AD"] = original
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["mean_length"], 3)
        self.assertEqual(_CountingAD.observe_calls, 2 * (3 + 2))


if __name__ == "__main__":
    unittest.main()
