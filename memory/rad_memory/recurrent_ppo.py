"""Shared RecurrentPPO construction and evaluation for source collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .envs import MemoryTaskSpec, make_memory_env


@dataclass(frozen=True)
class RecurrentPPOConfig:
    """The source-learner hyperparameters used to produce collection checkpoints."""

    policy: str = "MlpLstmPolicy"
    n_steps: int = 256
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    n_epochs: int = 10
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_recurrent_ppo(
    env,
    *,
    seed: int,
    config: RecurrentPPOConfig,
    tensorboard_log: str | Path | None,
    verbose: int = 1,
    device: str = "auto",
):
    """Build the exact RecurrentPPO learner used by teacher training."""

    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as error:
        raise RuntimeError(
            "RecurrentPPO requires sb3-contrib; finish the environment setup first"
        ) from error

    return RecurrentPPO(
        config.policy,
        env,
        seed=seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        n_epochs=config.n_epochs,
        clip_range=config.clip_range,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        normalize_advantage=config.normalize_advantage,
        verbose=verbose,
        device=device,
        tensorboard_log=None if tensorboard_log is None else str(tensorboard_log),
    )


def evaluate_recurrent_ppo(
    model,
    spec: MemoryTaskSpec,
    *,
    episodes: int,
    deterministic: bool = True,
) -> dict[str, float | int]:
    """Evaluate with a fresh recurrent state and a fixed seed per episode."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = make_memory_env(spec, flatten_for_source=True)
    successes = 0
    returns: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=spec.seed + episode)
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            episode_return = 0.0
            episode_length = 0
            while True:
                action, recurrent_state = model.predict(
                    observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=deterministic,
                )
                observation, reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                episode_return += float(reward)
                episode_length += 1
                episode_start[:] = terminated or truncated
                if terminated or truncated:
                    successes += int(bool(info["memory_success"] and episode_return > 0))
                    returns.append(episode_return)
                    lengths.append(episode_length)
                    break
    finally:
        env.close()

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }
