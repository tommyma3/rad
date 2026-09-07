"""Train fixed-task PPO/RecurrentPPO or the legacy recurrent layout-stream learner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .envs import MemoryTaskSpec, make_memory_env
from .recurrent_ppo import (
    RecurrentPPOConfig,
    build_recurrent_ppo,
    evaluate_recurrent_ppo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Train independent fixed-task learners with online histories")
    parser.add_argument("--source-algorithm", choices=("ppo", "recurrent_ppo"), default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="datasets-fixed")
    parser.add_argument("--required-consecutive-evals", type=int, default=3)
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
    if args.manifest:
        from .train_task_pool import train_pool
        from .ppo import PPOConfig
        algorithm = args.source_algorithm or "ppo"
        config_type = PPOConfig if algorithm == "ppo" else RecurrentPPOConfig
        train_pool(args.manifest, [args.seed], args.run_dir, args.output_root,
                   source_algorithm=algorithm, workers=args.workers,
                   torch_threads=args.torch_threads, device=args.device,
                   total_timesteps=args.total_timesteps, evaluation_interval=args.checkpoint_interval,
                   evaluation_episodes=args.validation_episodes, minimum_success_rate=args.minimum_success_rate,
                   required_consecutive_evals=args.required_consecutive_evals,
                   ppo_config=config_type(n_steps=args.n_steps, batch_size=args.batch_size))
        return
    if args.source_algorithm == "ppo":
        parser.error("Feed-forward PPO requires --manifest for fixed-task training")

    try:
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
    ppo_config = RecurrentPPOConfig(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
    )
    with (run_dir / "recurrent_ppo_config.json").open("w", encoding="utf-8") as handle:
        json.dump(ppo_config.to_dict(), handle, indent=2, sort_keys=True)
    model = build_recurrent_ppo(
        env,
        seed=args.seed,
        config=ppo_config,
        tensorboard_log=str(run_dir / "tensorboard"),
        device=args.device or "auto",
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
    metrics = evaluate_recurrent_ppo(
        model,
        validation_spec,
        episodes=args.validation_episodes,
    ) | {"minimum_success_rate": args.minimum_success_rate}
    with (run_dir / "teacher_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    if metrics["success_rate"] < args.minimum_success_rate:
        raise RuntimeError(
            f"Teacher success {metrics['success_rate']:.3f} is below the "
            f"required {args.minimum_success_rate:.3f}; do not collect distillation data"
        )


if __name__ == "__main__":
    main()
