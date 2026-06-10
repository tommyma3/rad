"""
Train PPO agents with the current MetaWorld config and test convergence.

This script is intentionally separate from collect.py: it does not save offline
trajectories, and instead answers whether PPO can solve the configured ML1 task
under the current hyperparameters.

Examples:
    uv run python test_ppo_convergence.py
    uv run python test_ppo_convergence.py --task-indices 0,1,2 --max-timesteps 500000
    uv run python test_ppo_convergence.py --all-train-tasks --success-threshold 0.9
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from algorithm import ALGORITHM
from env import SAMPLE_ENVIRONMENT, make_env
from utils import get_config


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on the current MetaWorld task and test convergence."
    )
    parser.add_argument(
        "--alg-config",
        type=Path,
        default=SCRIPT_DIR / "config" / "algorithm" / "ppo_ml1.yaml",
        help="PPO config to test.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=SCRIPT_DIR / "config" / "env" / "ml1.yaml",
        help="MetaWorld environment config to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "runs" / "ppo_convergence",
        help="Directory for JSON summaries and optional trained models.",
    )
    parser.add_argument(
        "--task-indices",
        default="0",
        help="Comma-separated task indices from the selected split. Defaults to the first train task.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Which ML1 task split to train/evaluate on.",
    )
    parser.add_argument(
        "--all-train-tasks",
        action="store_true",
        help="Run every train task. This can be expensive with the current PPO config.",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Training timesteps per agent. Defaults to total_source_timesteps from the PPO config.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluate after this many training timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.8,
        help="Mean episode success rate required for convergence.",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=None,
        help="Optional mean episode reward required for convergence.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Number of consecutive evaluations meeting thresholds before declaring convergence.",
    )
    parser.add_argument(
        "--n-stream",
        type=int,
        default=None,
        help="Override config n_stream for cheaper smoke tests.",
    )
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="dummy",
        help="Vectorized environment implementation for training.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for PPO.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to alg_seed for this convergence test.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save each final PPO model next to the JSON summary.",
    )
    parser.add_argument(
        "--override",
        default="",
        help="Config overrides in collect.py style, e.g. 'n_stream=8|source_lr=1e-4'.",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], override: str) -> None:
    for option in override.split("|"):
        if not option:
            continue
        address, value = option.split("=", 1)
        keys = address.split(".")
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f"Warning: {address} is not defined in config file.")
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)


def parse_task_indices(raw: str, n_tasks: int) -> list[int]:
    indices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= n_tasks:
            raise ValueError(f"Task index {idx} is out of range [0, {n_tasks - 1}].")
        indices.append(idx)
    if not indices:
        raise ValueError("No task indices were provided.")
    return indices


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))
    apply_overrides(config, args.override)

    if args.n_stream is not None:
        config["n_stream"] = args.n_stream
    if args.max_timesteps is not None:
        config["total_source_timesteps"] = args.max_timesteps

    return config


def make_vec_env(config: dict[str, Any], env_cls: type, task: Any, n_envs: int, kind: str):
    env_fns = [make_env(config, env_cls, task) for _ in range(n_envs)]
    if kind == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def select_device(config: dict[str, Any], requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if config.get("use_gpu", False) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_agent(model, config: dict[str, Any], env_cls: type, task: Any, n_episodes: int) -> dict[str, float]:
    eval_env = DummyVecEnv([make_env(config, env_cls, task)])
    episode_rewards = []
    episode_success = []
    current_reward = 0.0
    current_success = False

    obs = eval_env.reset()
    try:
        while len(episode_rewards) < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)

            current_reward += float(rewards[0])
            current_success = current_success or bool(infos[0].get("success", False))

            if bool(dones[0]):
                episode_rewards.append(current_reward)
                episode_success.append(float(current_success))
                current_reward = 0.0
                current_success = False
    finally:
        eval_env.close()

    rewards = np.asarray(episode_rewards, dtype=float)
    successes = np.asarray(episode_success, dtype=float)
    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "success_rate": float(successes.mean()),
        "episodes": int(len(episode_rewards)),
    }


def meets_convergence(metrics: dict[str, float], success_threshold: float, reward_threshold: float | None) -> bool:
    if metrics["success_rate"] < success_threshold:
        return False
    if reward_threshold is not None and metrics["mean_reward"] < reward_threshold:
        return False
    return True


def train_one_agent(
    config: dict[str, Any],
    env_cls: type,
    task: Any,
    task_index: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    local_config = copy.deepcopy(config)
    n_envs = int(local_config["n_stream"])
    total_timesteps = int(local_config["total_source_timesteps"])
    eval_freq = min(int(args.eval_freq), total_timesteps)
    seed = int(local_config["alg_seed"]) + int(args.seed_offset) + task_index
    device = select_device(local_config, args.device)

    train_env = make_vec_env(local_config, env_cls, task, n_envs, args.vec_env)
    log_dir = str(output_dir / "tensorboard") if local_config.get("use_tensorboard", False) else None

    model = None
    history = []
    consecutive_passes = 0
    converged = False
    trained_timesteps = 0
    start = datetime.now()

    try:
        model = ALGORITHM[local_config["alg"]](local_config, train_env, seed, log_dir, device=device)

        initial_metrics = evaluate_agent(model, local_config, env_cls, task, args.eval_episodes)
        initial_metrics.update({"timesteps": 0, "consecutive_passes": 0})
        history.append(initial_metrics)
        print(format_eval_line(task_index, initial_metrics))

        while trained_timesteps < total_timesteps:
            chunk = min(eval_freq, total_timesteps - trained_timesteps)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
                log_interval=local_config.get("log_interval", 100),
            )
            trained_timesteps = int(model.num_timesteps)

            metrics = evaluate_agent(model, local_config, env_cls, task, args.eval_episodes)
            passed = meets_convergence(metrics, args.success_threshold, args.reward_threshold)
            consecutive_passes = consecutive_passes + 1 if passed else 0
            metrics.update({"timesteps": trained_timesteps, "consecutive_passes": consecutive_passes})
            history.append(metrics)
            print(format_eval_line(task_index, metrics))

            if consecutive_passes >= args.patience:
                converged = True
                break

        if args.save_model:
            model_path = output_dir / f"ppo_task{task_index}_final.zip"
            model.save(model_path)
        else:
            model_path = None

    finally:
        train_env.close()

    elapsed = datetime.now() - start
    result = {
        "task_index": task_index,
        "task_env_name": getattr(task, "env_name", local_config.get("task", "unknown")),
        "seed": seed,
        "converged": converged,
        "trained_timesteps": trained_timesteps,
        "total_timesteps_budget": total_timesteps,
        "elapsed_seconds": elapsed.total_seconds(),
        "success_threshold": args.success_threshold,
        "reward_threshold": args.reward_threshold,
        "patience": args.patience,
        "n_stream": n_envs,
        "history": history,
    }
    if args.save_model and model_path is not None:
        result["model_path"] = str(model_path)
    return result


def format_eval_line(task_index: int, metrics: dict[str, Any]) -> str:
    return (
        f"[task {task_index}] step={metrics['timesteps']:,} "
        f"reward={metrics['mean_reward']:.2f}+/-{metrics['std_reward']:.2f} "
        f"success={metrics['success_rate']:.2f} "
        f"passes={metrics['consecutive_passes']}"
    )


def save_summary(output_dir: Path, config: dict[str, Any], results: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = output_dir / f"ppo_convergence_{config['task']}_{timestamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "n_agents": len(results),
        "n_converged": sum(int(r["converged"]) for r in results),
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")

    config = build_config(args)
    if config["env"] not in SAMPLE_ENVIRONMENT:
        raise ValueError(f"Unsupported environment family: {config['env']}")
    if config["alg"] not in ALGORITHM:
        raise ValueError(f"Unsupported algorithm: {config['alg']}")

    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))

    train_envs, test_envs = SAMPLE_ENVIRONMENT[config["env"]](config)
    task_pool = train_envs if args.split == "train" else test_envs
    if args.all_train_tasks:
        if args.split != "train":
            raise ValueError("--all-train-tasks can only be used with --split train.")
        task_indices = list(range(len(task_pool)))
    else:
        task_indices = parse_task_indices(args.task_indices, len(task_pool))

    print("PPO convergence test")
    print(f"  task: {config['task']} ({args.split} split)")
    print(f"  task indices: {task_indices}")
    print(f"  streams per agent: {config['n_stream']}")
    print(f"  timestep budget per agent: {config['total_source_timesteps']:,}")
    print(f"  eval every: {args.eval_freq:,} steps")
    print(f"  convergence: success >= {args.success_threshold}")
    if args.reward_threshold is not None:
        print(f"  reward >= {args.reward_threshold}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for task_index in task_indices:
        env_cls, task = task_pool[task_index]
        print(f"Starting PPO agent for task index {task_index}: {getattr(task, 'env_name', config['task'])}")
        result = train_one_agent(config, env_cls, task, task_index, args, args.output_dir)
        results.append(result)
        verdict = "CONVERGED" if result["converged"] else "NOT CONVERGED"
        print(f"Finished task {task_index}: {verdict}\n")

    summary_path = save_summary(args.output_dir, config, results)
    n_converged = sum(int(r["converged"]) for r in results)
    print(f"Converged agents: {n_converged}/{len(results)}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
