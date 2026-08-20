from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "memory_maze"
sys.path.insert(0, str(PROJECT))

from artifacts import TaskHistoryWriter
from dataset import ADDataset
from env import MemoryMazeTaskSpec


class MemoryMazeDatasetContractTest(unittest.TestCase):
    def _write_history(self, root: Path, split: str):
        spec = MemoryMazeTaskSpec(9, 10 if split == "train" else 20, split, task_id=f"{split}-task")
        path = root / split / "ppo" / f"{spec.task_id}.hdf5"
        with TaskHistoryWriter(path, spec, "ppo", {}) as writer:
            for episode in range(2):
                observation = np.full((64, 64, 3), episode, dtype=np.uint8)
                for step in range(4):
                    next_observation = np.full((64, 64, 3), episode + step + 1, dtype=np.uint8)
                    writer.append(
                        observation,
                        step % 6,
                        float(step == 3),
                        next_observation,
                        False,
                        step == 3,
                        learner_step=episode * 4 + step,
                    )
                    observation = next_observation

    def test_windows_cross_episodes_only_within_same_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_history(root, "train")
            self._write_history(root, "test")
            dataset = ADDataset(
                {"n_transit": 6, "source_algorithm": "ppo", "dataset_stride": 1},
                root,
                "train",
            )
            item = dataset[0]
            self.assertEqual(item["states"].shape, (6, 64, 64, 3))
            self.assertEqual(int(item["dones"].sum()), 1)
            self.assertTrue(item["task_id"].startswith("train-task"))


if __name__ == "__main__":
    unittest.main()
