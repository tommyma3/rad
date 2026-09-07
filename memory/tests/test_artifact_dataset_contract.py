from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import json

import h5py

import numpy as np

from rad_memory.artifacts import TaskHistoryWriter
from rad_memory.dataset import ADDataset, RADDataset, collate_trajectories
from rad_memory.envs import MemoryTaskSpec


def _observation(value: int) -> dict:
    return {
        "image": np.full((7, 7, 3), value, dtype=np.uint8),
        "direction": value % 4,
        "mission": "go to the matching object at the end of the hallway",
    }


def _episode(length: int) -> list[dict]:
    result = []
    for step in range(length):
        result.append(
            {
                "observation": _observation(step),
                "action": step % 7,
                "reward": float(step == length - 1),
                "terminated": step == length - 1,
                "truncated": False,
                "next_observation": _observation(step + 1),
                "cue_id": 0,
                "cue_visible": step == 0,
                "decision": step == length - 1,
                "success": step == length - 1,
            }
        )
    return result


class ArtifactDatasetContractTest(unittest.TestCase):
    def test_legacy_spec_without_configuration_field_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.hdf5"
            spec = MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=8)
            with TaskHistoryWriter(path, spec, "recurrent_ppo") as writer:
                writer.write_episode(_episode(5))
            with h5py.File(path, "a") as handle:
                saved = json.loads(handle.attrs["task_spec"])
                saved.pop("configuration")
                handle.attrs["task_spec"] = json.dumps(saved)
            with TaskHistoryWriter(path, spec, "recurrent_ppo") as writer:
                self.assertEqual(writer.next_episode_index, 1)

    def _write(self, root: Path) -> None:
        spec = MemoryTaskSpec("MiniGrid-MemoryS13-v0", 3, "train", horizon=8)
        path = root / "train" / "recurrent_ppo" / f"{spec.task_id}.hdf5"
        with TaskHistoryWriter(path, spec, "recurrent_ppo") as writer:
            writer.write_episode(_episode(5), learner_step=100)
            writer.write_episode(_episode(7), learner_step=200)

    def test_ad_windows_never_cross_episode_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root)
            with ADDataset(
                {
                    "n_transit": 6,
                    "dataset_stride": 3,
                    "source_algorithm": "recurrent_ppo",
                },
                root,
                "train",
            ) as dataset:
                for item in dataset:
                    self.assertLessEqual(len(item["actions"]), 6)
                    self.assertLessEqual(int(item["terminated"].sum()), 1)
                    if item["terminated"].any():
                        self.assertTrue(bool(item["terminated"][-1]))

    def test_rad_bucket_and_padding_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root)
            with RADDataset(
                {
                    "n_transit": 4,
                    "short_memory_keep": 1,
                    "max_context_length": 10,
                    "source_algorithm": "recurrent_ppo",
                },
                root,
                "train",
            ) as dataset:
                first = dataset[(0, 0)]
                second = dataset[(2, 2)]
                batch = collate_trajectories([first, second])
                self.assertEqual(tuple(batch["images"].shape[:2]), (2, 7))
                self.assertEqual(batch["valid_mask"].sum(1).tolist(), [4, 7])
                self.assertEqual(dataset.available_compression_buckets(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
