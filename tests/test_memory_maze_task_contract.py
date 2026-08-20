import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "memory_maze"
sys.path.insert(0, str(PROJECT))

from env.fixed_task import (
    FixedMemoryMazeEnv,
    MemoryMazeTaskSpec,
    generate_task_specs,
    load_task_spec,
    save_task_spec,
)


class MemoryMazeTaskContractTest(unittest.TestCase):
    def test_task_seed_splits_are_disjoint(self):
        train, test = generate_task_specs(9, 100, 25, split_seed=17)
        self.assertFalse(
            {task.generation_seed for task in train}
            & {task.generation_seed for task in test}
        )

    def test_capture_hashes_full_fixed_configuration(self):
        observation = {
            "maze_layout": np.eye(9, dtype=np.uint8),
            "targets_pos": np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            "agent_pos": np.asarray([2, 7], dtype=np.float32),
            "agent_dir": np.asarray([0, 1], dtype=np.float32),
        }
        spec = MemoryMazeTaskSpec(9, 3, "train").with_capture(observation)
        changed = dict(observation)
        changed["agent_dir"] = np.asarray([1, 0], dtype=np.float32)
        self.assertNotEqual(spec.task_id, MemoryMazeTaskSpec(9, 3, "train").with_capture(changed).task_id)

    def test_manifest_round_trip(self):
        spec = MemoryMazeTaskSpec(9, 3, "test", task_id="task-id")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            save_task_spec(spec, path)
            self.assertEqual(load_task_spec(path), spec)

    def test_real_environment_restores_identical_task_and_start_pose(self):
        env = FixedMemoryMazeEnv(MemoryMazeTaskSpec(9, 123, "train"))
        _, first = env.reset()
        _, second = env.reset()
        env.close()
        np.testing.assert_array_equal(first["maze_layout"], second["maze_layout"])
        np.testing.assert_allclose(first["targets_pos"], second["targets_pos"])
        np.testing.assert_allclose(first["agent_pos"], second["agent_pos"])
        np.testing.assert_allclose(first["agent_dir"], second["agent_dir"])
        self.assertNotEqual(first["target_sequence_seed"], second["target_sequence_seed"])


if __name__ == "__main__":
    unittest.main()
