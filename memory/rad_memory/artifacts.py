"""Append-safe HDF5 artifacts for MiniGrid Memory trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .envs import MemoryTaskSpec, numeric_observation


TRAJECTORY_FORMAT = "rad-minigrid-memory-v1"
FIXED_TRAJECTORY_FORMAT = "rad-minigrid-memory-v2-fixed"


def _as_array(steps: list[dict[str, Any]], key: str, dtype=None) -> np.ndarray:
    values = [step[key] for step in steps]
    return np.asarray(values, dtype=dtype)


class TaskHistoryWriter:
    """Writes complete episodes and never mutates an already-written episode."""

    def __init__(
        self,
        path: str | Path,
        task_spec: MemoryTaskSpec,
        source_algorithm: str,
        source_config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_spec = task_spec
        self.source_algorithm = source_algorithm
        self.source_config = source_config or {}
        self.format = FIXED_TRAJECTORY_FORMAT if task_spec.configuration is not None else TRAJECTORY_FORMAT
        self.handle = h5py.File(self.path, "a", libver="latest")
        if "format" in self.handle.attrs:
            try:
                self._validate_existing()
            except Exception:
                self.handle.close()
                raise
        else:
            self.handle.attrs["format"] = self.format
            self.handle.attrs["task_spec"] = json.dumps(task_spec.to_dict(), sort_keys=True)
            self.handle.attrs["source_algorithm"] = source_algorithm
            self.handle.attrs["source_config"] = json.dumps(source_config or {}, sort_keys=True)
            self.handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self.episodes = self.handle.require_group("episodes")

    def _validate_existing(self) -> None:
        if self.handle.attrs["format"] != self.format:
            raise ValueError(f"Unsupported trajectory format in {self.path}")
        stored_spec = json.loads(self.handle.attrs["task_spec"])
        normalized_spec = MemoryTaskSpec.from_dict(stored_spec).to_dict()
        if stored_spec["task_id"] != normalized_spec["task_id"] or normalized_spec != self.task_spec.to_dict():
            raise ValueError(f"Task mismatch while resuming {self.path}")
        if self.handle.attrs["source_algorithm"] != self.source_algorithm:
            raise ValueError(f"Source-algorithm mismatch while resuming {self.path}")
        if json.loads(self.handle.attrs["source_config"]) != self.source_config:
            raise ValueError(f"Source-run mismatch while resuming {self.path}")

    @property
    def next_episode_index(self) -> int:
        keys = [int(key) for key in self.episodes.keys()]
        return max(keys, default=-1) + 1

    def count_episodes_at_learner_step(self, learner_step: int) -> int:
        return sum(
            int(group.attrs.get("learner_step", -1)) == int(learner_step)
            for group in self.episodes.values()
        )

    def write_episode(
        self,
        steps: list[dict[str, Any]],
        *,
        episode_index: int | None = None,
        learner_step: int = 0,
    ) -> str:
        if not steps:
            raise ValueError("Cannot write an empty episode")
        if not (steps[-1]["terminated"] or steps[-1]["truncated"]):
            raise ValueError("A stored episode must end in terminated or truncated")
        index = self.next_episode_index if episode_index is None else int(episode_index)
        key = f"{index:08d}"
        if key in self.episodes:
            raise ValueError(f"Episode {key} already exists in {self.path}")

        group = self.episodes.create_group(key)
        observations = [numeric_observation(step["observation"]) for step in steps]
        next_observations = [numeric_observation(step["next_observation"]) for step in steps]
        group.create_dataset("images", data=np.stack([item["image"] for item in observations]))
        group.create_dataset("directions", data=_as_array(observations, "direction", np.int8))
        group.create_dataset("next_images", data=np.stack([item["image"] for item in next_observations]))
        group.create_dataset("next_directions", data=_as_array(next_observations, "direction", np.int8))
        group.create_dataset("actions", data=_as_array(steps, "action", np.int8))
        group.create_dataset("rewards", data=_as_array(steps, "reward", np.float32))
        group.create_dataset("terminated", data=_as_array(steps, "terminated", np.bool_))
        group.create_dataset("truncated", data=_as_array(steps, "truncated", np.bool_))
        group.create_dataset("cue_ids", data=_as_array(steps, "cue_id", np.int8))
        group.create_dataset("cue_visible", data=_as_array(steps, "cue_visible", np.bool_))
        group.create_dataset("decision", data=_as_array(steps, "decision", np.bool_))
        group.create_dataset("success", data=_as_array(steps, "success", np.bool_))
        group.create_dataset(
            "learner_steps",
            data=np.asarray([step.get("learner_step", learner_step) for step in steps], dtype=np.int64),
        )
        group.attrs["length"] = len(steps)
        group.attrs["learner_step"] = int(learner_step)
        self.handle.flush()
        return key

    def close(self) -> None:
        if self.handle:
            self.handle.flush()
            self.handle.close()

    def __enter__(self) -> "TaskHistoryWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def transition_record(
    observation: dict[str, Any],
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    next_observation: dict[str, Any],
    info: dict[str, Any],
    observation_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one environment step into the artifact schema."""

    observation_info = info if observation_info is None else observation_info
    return {
        "observation": observation,
        "action": int(action),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "next_observation": next_observation,
        "cue_id": int(observation_info["memory_cue_id"]),
        "cue_visible": bool(observation_info["memory_cue_visible"]),
        "decision": bool(info["memory_decision"]),
        "success": bool(info["memory_success"]),
    }
