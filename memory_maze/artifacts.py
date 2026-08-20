"""Versioned, append-friendly source-learning history artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from env import MemoryMazeTaskSpec


TRAJECTORY_FORMAT = "memory-maze-trajectories-v1"


class TaskHistoryWriter:
    """Write complete episodes without duplicating next observations."""

    def __init__(
        self,
        path: str | Path,
        task_spec: MemoryMazeTaskSpec,
        source_algorithm: str,
        config: dict[str, Any],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")
        self.file.attrs["format"] = TRAJECTORY_FORMAT
        self.file.attrs["task_spec"] = json.dumps(asdict(task_spec), sort_keys=True)
        self.file.attrs["source_algorithm"] = source_algorithm
        self.file.attrs["config"] = json.dumps(config, sort_keys=True, default=str)
        self.streams = self.file.create_group("streams")
        self._episode_counts: dict[int, int] = {}
        self._buffers: dict[int, dict[str, list]] = {}

    def _buffer(self, stream_id: int) -> dict[str, list]:
        if stream_id not in self._buffers:
            self._buffers[stream_id] = {
                "images": [],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "truncated": [],
                "learner_steps": [],
            }
            stream = self.streams.require_group(f"{stream_id:04d}")
            stream.require_group("episodes")
            self._episode_counts[stream_id] = 0
        return self._buffers[stream_id]

    def append(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
        learner_step: int,
        stream_id: int = 0,
    ) -> None:
        buffer = self._buffer(int(stream_id))
        if not buffer["images"]:
            buffer["images"].append(np.asarray(observation, dtype=np.uint8))
        buffer["actions"].append(int(action))
        buffer["rewards"].append(float(reward))
        buffer["terminated"].append(bool(terminated))
        buffer["truncated"].append(bool(truncated))
        buffer["learner_steps"].append(int(learner_step))
        buffer["images"].append(np.asarray(next_observation, dtype=np.uint8))
        if terminated or truncated:
            self.finish_episode(stream_id)

    def finish_episode(self, stream_id: int = 0) -> None:
        buffer = self._buffer(int(stream_id))
        if not buffer["actions"]:
            return
        episodes = self.streams[f"{stream_id:04d}"]["episodes"]
        episode_count = self._episode_counts[stream_id]
        group = episodes.create_group(f"{episode_count:08d}")
        group.create_dataset(
            "images",
            data=np.stack(buffer["images"]),
            dtype=np.uint8,
            chunks=(1, *buffer["images"][0].shape),
            compression="gzip",
            compression_opts=1,
        )
        group.create_dataset("actions", data=np.asarray(buffer["actions"], dtype=np.uint8))
        group.create_dataset("rewards", data=np.asarray(buffer["rewards"], dtype=np.float32))
        group.create_dataset("terminated", data=np.asarray(buffer["terminated"], dtype=np.bool_))
        group.create_dataset("truncated", data=np.asarray(buffer["truncated"], dtype=np.bool_))
        group.create_dataset("learner_steps", data=np.asarray(buffer["learner_steps"], dtype=np.int64))
        self._episode_counts[stream_id] += 1
        self.file.attrs["episode_count"] = sum(self._episode_counts.values())
        self.file.flush()
        self._buffers[stream_id] = {key: [] for key in buffer}

    def close(self) -> None:
        if self.file:
            for stream_id in list(self._buffers):
                self.finish_episode(stream_id)
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
