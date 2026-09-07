"""MiniGrid Memory task construction and benchmark-only instrumentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class MemoryTaskSpec:
    """A fixed configuration, or an explicitly legacy seeded layout stream."""

    env_id: str
    seed: int
    split: str
    horizon: int | None = None
    controlled: bool = False
    size: int | None = None
    random_length: bool = False
    configuration: dict[str, Any] | None = None

    @property
    def task_id(self) -> str:
        if self.configuration is not None:
            return "fixed-" + hashlib.sha256(
                json.dumps(self.configuration, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        mode = "controlled" if self.controlled else "official"
        horizon = "native" if self.horizon is None else f"h{self.horizon}"
        return f"{self.split}-{mode}-{self.env_id}-{horizon}-seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"task_id": self.task_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryTaskSpec":
        fields = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        return cls(**fields)


def _base_env(env: gym.Env):
    return env.unwrapped


class ControlledMemoryEnv:
    """Factory for a MemoryEnv whose cue is guaranteed visible at reset.

    It deliberately retains MiniGrid's grid generation, action space, reward,
    and termination rules. Only the initial pose is fixed after grid creation.
    """

    @staticmethod
    def build(size: int, random_length: bool, max_steps: int | None) -> gym.Env:
        from minigrid.envs.memory import MemoryEnv

        class _ControlledMemoryEnv(MemoryEnv):
            def _gen_grid(self, width: int, height: int) -> None:
                super()._gen_grid(width, height)
                self.agent_pos = np.asarray((1, height // 2), dtype=np.int64)
                self.agent_dir = 0

        kwargs: dict[str, Any] = {"size": size, "random_length": random_length}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        return _ControlledMemoryEnv(**kwargs)


class MemoryInstrumentation(gym.Wrapper):
    """Adds privileged diagnostic metadata without changing observations."""

    def __init__(self, env: gym.Env, task_spec: MemoryTaskSpec):
        super().__init__(env)
        self.task_spec = task_spec
        self._episode_index = -1
        self._cue_id = -1

    def _cue_position(self) -> tuple[int, int]:
        base = _base_env(self.env)
        return 1, int(base.height) // 2 - 1

    def _read_cue_id(self) -> int:
        base = _base_env(self.env)
        cell = base.grid.get(*self._cue_position())
        if cell is None or cell.type not in {"key", "ball"}:
            raise RuntimeError("MiniGrid Memory cue cell is missing a key/ball")
        return 0 if cell.type == "key" else 1

    def _cue_visible(self) -> bool:
        base = _base_env(self.env)
        return bool(base.agent_sees(*self._cue_position()))

    def _diagnostics(self) -> dict[str, Any]:
        base = _base_env(self.env)
        position = tuple(int(value) for value in base.agent_pos)
        at_success = position == tuple(base.success_pos)
        at_failure = position == tuple(base.failure_pos)
        return {
            "memory_task_id": self.task_spec.task_id,
            "memory_episode_index": self._episode_index,
            "memory_cue_id": self._cue_id,
            "memory_cue_visible": self._cue_visible(),
            "memory_decision": at_success or at_failure,
            "memory_success": at_success,
            "memory_failure": at_failure,
            "memory_agent_position": position,
            "memory_success_position": tuple(base.success_pos),
            "memory_failure_position": tuple(base.failure_pos),
            "memory_step": int(base.step_count),
        }

    def reset(self, **kwargs):
        if "seed" not in kwargs and self.task_spec.configuration is None:
            kwargs["seed"] = self.task_spec.seed + self._episode_index + 1
        observation, info = self.env.reset(**kwargs)
        self._episode_index += 1
        self._cue_id = self._read_cue_id()
        return observation, dict(info) | self._diagnostics()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            dict(info) | self._diagnostics(),
        )


def numeric_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Discard the constant mission string while preserving symbolic inputs."""

    return {
        "image": np.asarray(observation["image"], dtype=np.uint8),
        "direction": np.asarray(observation["direction"], dtype=np.int64),
    }


def flatten_numeric_observation(observation: dict[str, Any]) -> np.ndarray:
    """Compact float input used by the recurrent source learner."""

    image = np.asarray(observation["image"], dtype=np.float32).copy()
    image[..., 0] /= 15.0
    image[..., 1] /= 7.0
    image[..., 2] /= 3.0
    image = image.reshape(-1)
    direction = np.zeros(4, dtype=np.float32)
    direction[int(observation["direction"])] = 1.0
    return np.concatenate([image, direction])


class FlattenMemoryObservation(gym.ObservationWrapper):
    """Removes the mission and flattens MiniGrid observations for SB3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        image_shape = env.observation_space["image"].shape
        length = int(np.prod(image_shape)) + 4
        self.observation_space = spaces.Box(0.0, 1.0, shape=(length,), dtype=np.float32)

    def observation(self, observation):
        return flatten_numeric_observation(observation)


def make_memory_env(
    task_spec: MemoryTaskSpec,
    *,
    flatten_for_source: bool = False,
) -> gym.Env:
    """Construct one official or controlled MiniGrid Memory environment."""

    import minigrid  # noqa: F401 - importing registers the official env IDs

    if task_spec.configuration is not None:
        from copy import deepcopy
        from minigrid.core.grid import Grid
        from minigrid.envs.memory import MemoryEnv

        configuration = deepcopy(task_spec.configuration)

        class FixedMemoryEnv(MemoryEnv):
            def _gen_grid(self, width, height):
                self.grid, _ = Grid.decode(np.asarray(configuration["grid"], dtype=np.uint8))
                self.agent_pos = np.asarray(configuration["agent_pos"], dtype=np.int64)
                self.agent_dir = int(configuration["agent_dir"])
                self.success_pos = tuple(configuration["success_pos"])
                self.failure_pos = tuple(configuration["failure_pos"])
                self.mission = configuration["mission"]

        env = FixedMemoryEnv(
            size=int(configuration["size"]),
            max_steps=int(configuration["max_steps"]),
            agent_view_size=int(configuration["agent_view_size"]),
        )
    elif task_spec.controlled:
        if task_spec.size is None:
            raise ValueError("Controlled memory tasks require an explicit size")
        env = ControlledMemoryEnv.build(
            size=task_spec.size,
            random_length=task_spec.random_length,
            max_steps=task_spec.horizon,
        )
    else:
        kwargs = {}
        if task_spec.horizon is not None:
            kwargs["max_steps"] = task_spec.horizon
        env = gym.make(task_spec.env_id, **kwargs)
    env = MemoryInstrumentation(env, task_spec)
    if flatten_for_source:
        env = FlattenMemoryObservation(env)
    return env
