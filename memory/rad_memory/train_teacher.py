"""Train a recurrent PPO source learner and save learning checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .envs import MemoryTaskSpec, make_memory_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RecurrentPPO on MiniGrid Memory")
    parser.add_argument("--env-id", default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--size", type=int)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--random-length", action="store_true")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--validation-seed", type=int, default=10000)
    parser.add_argument("--validation-episodes", type=int, default=200)
    parser.add_argument("--minimum-success-rate", type=float, default=0.9)
    args = parser.parse_args()

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as error:
        raise RuntimeError(
            "RecurrentPPO requires sb3-contrib; finish the environment setup first"
        ) from error

    spec = MemoryTaskSpec(
        env_id=args.env_id,
        seed=args.seed,
        split="train",
        horizon=args.horizon,
        controlled=args.controlled,
        size=args.size,
        random_length=args.random_length,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "task_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)

    env = make_memory_env(spec, flatten_for_source=True)
    callback = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=str(run_dir),
        name_prefix="teacher-checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    model.learn(args.total_timesteps, callback=callback, progress_bar=True)
    model.save(run_dir / "teacher-final")
    env.close()

    validation_spec = MemoryTaskSpec(
        env_id=args.env_id,
        seed=args.validation_seed,
        split="validation",
        horizon=args.horizon,
        controlled=args.controlled,
        size=args.size,
        random_length=args.random_length,
    )
    validation_env = make_memory_env(validation_spec, flatten_for_source=True)
    successes = 0
    returns = []
    for episode in range(args.validation_episodes):
        observation, _ = validation_env.reset(seed=args.validation_seed + episode)
        recurrent_state = None
        episode_start = np.ones((1,), dtype=bool)
        episode_return = 0.0
        while True:
            action, recurrent_state = model.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = validation_env.step(
                int(np.asarray(action).item())
            )
            episode_return += float(reward)
            episode_start[:] = terminated or truncated
            if terminated or truncated:
                successes += int(bool(info["memory_success"] and episode_return > 0))
                returns.append(episode_return)
                break
    validation_env.close()
    metrics = {
        "episodes": args.validation_episodes,
        "success_rate": successes / max(args.validation_episodes, 1),
        "mean_return": sum(returns) / max(len(returns), 1),
        "minimum_success_rate": args.minimum_success_rate,
    }
    with (run_dir / "teacher_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    if metrics["success_rate"] < args.minimum_success_rate:
        raise RuntimeError(
            f"Teacher success {metrics['success_rate']:.3f} is below the "
            f"required {args.minimum_success_rate:.3f}; do not collect distillation data"
        )


if __name__ == "__main__":
    main()
