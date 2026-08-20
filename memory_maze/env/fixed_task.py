"""Fixed-task adapter for the upstream Memory Maze environment.

The upstream environment regenerates its maze, object positions, and starting
pose on every reset. Algorithm Distillation instead needs one source learner to
make repeated attempts at exactly one task. This adapter reconstructs an
upstream environment from the same generation seed for every episode and
verifies the resulting task fingerprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


_MAZE_SIZES = (9, 11, 13, 15)
_EPISODE_STEPS = {9: 1000, 11: 2000, 13: 3000, 15: 4000}


@dataclass(frozen=True)
class MemoryMazeTaskSpec:
    """Serializable identity of one fixed Memory Maze task.

    ``generation_seed`` fixes maze topology, object placement, start position,
    start orientation, colors, and initial requested object. The optional
    captured fields make the task auditable and protect against upstream
    environment drift.
    """

    maze_size: int
    generation_seed: int
    split: str
    task_id: str = ""
    maze_layout: list[list[int]] | None = None
    object_positions: list[list[float]] | None = None
    start_position: list[float] | None = None
    start_direction: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.maze_size not in _MAZE_SIZES:
            raise ValueError(f"maze_size must be one of {_MAZE_SIZES}, got {self.maze_size}")
        if self.split not in {"train", "test"}:
            raise ValueError(f"split must be train or test, got {self.split!r}")

    def with_capture(self, observation: dict[str, np.ndarray]) -> "MemoryMazeTaskSpec":
        payload = {
            "maze_size": self.maze_size,
            "generation_seed": self.generation_seed,
            "maze_layout": np.asarray(observation["maze_layout"], dtype=np.uint8).tolist(),
            "object_positions": np.asarray(observation["targets_pos"], dtype=np.float32).tolist(),
            "start_position": np.asarray(observation["agent_pos"], dtype=np.float32).tolist(),
            "start_direction": np.asarray(observation["agent_dir"], dtype=np.float32).tolist(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        return MemoryMazeTaskSpec(
            maze_size=self.maze_size,
            generation_seed=self.generation_seed,
            split=self.split,
            task_id=f"memory-{self.maze_size}x{self.maze_size}-{digest}",
            maze_layout=payload["maze_layout"],
            object_positions=payload["object_positions"],
            start_position=payload["start_position"],
            start_direction=payload["start_direction"],
            metadata=dict(self.metadata),
        )


def save_task_spec(spec: MemoryMazeTaskSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(spec), indent=2, sort_keys=True), encoding="utf-8")


def load_task_spec(path: str | Path) -> MemoryMazeTaskSpec:
    return MemoryMazeTaskSpec(**json.loads(Path(path).read_text(encoding="utf-8")))


def generate_task_specs(
    maze_size: int,
    n_train_tasks: int,
    n_test_tasks: int,
    split_seed: int,
) -> tuple[list[MemoryMazeTaskSpec], list[MemoryMazeTaskSpec]]:
    """Generate disjoint task seeds without constructing MuJoCo environments."""
    rng = np.random.default_rng(split_seed)
    seeds = rng.choice(
        np.iinfo(np.int32).max,
        size=n_train_tasks + n_test_tasks,
        replace=False,
    )
    train = [
        MemoryMazeTaskSpec(maze_size, int(seed), "train")
        for seed in seeds[:n_train_tasks]
    ]
    test = [
        MemoryMazeTaskSpec(maze_size, int(seed), "test")
        for seed in seeds[n_train_tasks:]
    ]
    return train, test


def _task_constructor(maze_size: int):
    tasks = importlib.import_module("memory_maze.tasks")
    return getattr(tasks, f"memory_maze_{maze_size}x{maze_size}")


def _as_info(observation: dict[str, np.ndarray], spec: MemoryMazeTaskSpec) -> dict[str, Any]:
    return {
        "task_id": spec.task_id,
        "maze_size": spec.maze_size,
        "agent_pos": np.asarray(observation["agent_pos"], dtype=np.float32),
        "agent_dir": np.asarray(observation["agent_dir"], dtype=np.float32),
        "targets_pos": np.asarray(observation["targets_pos"], dtype=np.float32),
        "target_pos": np.asarray(observation["target_pos"], dtype=np.float32),
        "maze_layout": np.asarray(observation["maze_layout"], dtype=np.uint8),
    }


def _reseed_post_reset_dynamics(env, seed: int) -> bool:
    """Reseed target progression after fixed geometry and pose are restored.

    Memory Maze's wrappers do not expose this through their public API. We walk
    the wrapper chain defensively and update the composer environment RNG only
    after reset, so maze/object/start generation remains fixed while subsequent
    target choices can differ between attempts.
    """
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "_random_state"):
            current._random_state = np.random.RandomState(seed)
            return True
        next_object = None
        for attribute in ("_env", "env", "_environment", "environment"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                next_object = candidate
                break
        current = next_object
    return False


class FixedMemoryMazeEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment that restores one exact task on every reset.

    A new upstream dm_env instance is constructed from the task generation seed
    for each episode. This is deliberately more expensive than a normal reset,
    but it is the reliable public-API boundary for preventing Memory Maze from
    regenerating a different task.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_spec: MemoryMazeTaskSpec,
        *,
        camera_resolution: int = 64,
        control_freq: float = 4.0,
        verify_fixed_task: bool = True,
        target_sequence_stream: int = 0,
    ) -> None:
        super().__init__()
        self.task_spec = task_spec
        self.camera_resolution = int(camera_resolution)
        self.control_freq = float(control_freq)
        self.verify_fixed_task = bool(verify_fixed_task)
        self.target_sequence_stream = int(target_sequence_stream)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.camera_resolution, self.camera_resolution, 3),
            dtype=np.uint8,
        )
        self._env = None
        self._episode_index = 0
        self._captured_spec: MemoryMazeTaskSpec | None = None

    @property
    def episode_steps(self) -> int:
        return _EPISODE_STEPS[self.task_spec.maze_size]

    @property
    def resolved_task_spec(self) -> MemoryMazeTaskSpec:
        return self._captured_spec or self.task_spec

    def _make_env(self):
        constructor = _task_constructor(self.task_spec.maze_size)
        return constructor(
            seed=self.task_spec.generation_seed,
            global_observables=True,
            image_only_obs=False,
            camera_resolution=self.camera_resolution,
            control_freq=self.control_freq,
            discrete_actions=True,
        )

    def _capture_and_verify(self, observation: dict[str, np.ndarray]) -> None:
        captured = self.task_spec.with_capture(observation)
        if self._captured_spec is None:
            self._captured_spec = captured
        elif self.verify_fixed_task and captured.task_id != self._captured_spec.task_id:
            raise RuntimeError(
                "Memory Maze task changed across resets: "
                f"{self._captured_spec.task_id} -> {captured.task_id}"
            )

        expected = self.task_spec
        if self.verify_fixed_task and expected.task_id and expected.task_id != captured.task_id:
            raise RuntimeError(
                f"Task manifest mismatch: expected {expected.task_id}, generated {captured.task_id}"
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if close is not None:
                close()
        self._env = self._make_env()
        timestep = self._env.reset()
        observation = timestep.observation
        self._capture_and_verify(observation)
        self._episode_index += 1
        sequence_seed = int(
            np.random.SeedSequence(
                [
                    self.task_spec.generation_seed,
                    self.target_sequence_stream,
                    self._episode_index,
                ]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        reseeded = _reseed_post_reset_dynamics(self._env, sequence_seed)
        if not reseeded:
            raise RuntimeError(
                "Unable to locate the upstream composer RNG needed to preserve "
                "fixed geometry while varying target-request progression"
            )
        info = _as_info(observation, self.resolved_task_spec)
        info["episode_index"] = self._episode_index
        info["target_sequence_seed"] = sequence_seed
        info["target_sequence_reseeded"] = reseeded
        return np.asarray(observation["image"], dtype=np.uint8), info

    def step(self, action: int):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        timestep = self._env.step(int(action))
        observation = timestep.observation
        truncated = bool(timestep.last())
        reward = float(timestep.reward or 0.0)
        info = _as_info(observation, self.resolved_task_spec)
        info.update(
            {
                "episode_index": self._episode_index,
                "target_reached": reward > 0.0,
                "discount": float(timestep.discount if timestep.discount is not None else 1.0),
            }
        )
        return (
            np.asarray(observation["image"], dtype=np.uint8),
            reward,
            False,
            truncated,
            info,
        )

    def close(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if close is not None:
                close()
            self._env = None
