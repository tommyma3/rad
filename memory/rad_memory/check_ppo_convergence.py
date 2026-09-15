"""Quick probe: train one PPO agent on a single sampled Memory task.

Much lighter than collection training: it records no history artifacts and
runs a single learner, so it finishes in minutes instead of hours.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import random

import torch

from .check_recurrent_ppo_convergence import (
    _make_evaluation_callback,
    _write_json,
    has_converged,
)
from .envs import MemoryTaskSpec, make_memory_env
from .ppo import PPOConfig, build_ppo, evaluate_ppo
from .task_pool import freeze_task, load_pool
from .utils import load_config

DEFAULT_ENV_ID = "MiniGrid-MemoryS13Random-v0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", help="Task pool to sample a fixed task from")
    source.add_argument("--task-spec", help="Saved fixed task_spec.json")
    parser.add_argument("--task-id", help="Pin one manifest task instead of sampling")
    parser.add_argument("--split", choices=("train", "test"),
                        help="Limit manifest sampling to one split")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID,
                        help="Env used to generate a task without --manifest")
    parser.add_argument("--task-seed", type=int,
                        help="Task layout seed; default draws one at random")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for the random task draw")
    parser.add_argument("--horizon", type=int,
                        help="Episode horizon when generating a task (default 30)")
    parser.add_argument("--size", type=int)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--random-length", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="PPO learner seed")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--minimum-success-rate", type=float, default=0.9)
    parser.add_argument("--required-consecutive-evals", type=int, default=2)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--ppo-config",
                        help="Collection source_config.json/.yaml (e.g. config/source/ppo.yaml) "
                             "or raw PPOConfig JSON; flag overrides still win")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/ppo-convergence")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)
    if args.task_id and not args.manifest:
        parser.error("--task-id requires --manifest")
    if args.split and not args.manifest:
        parser.error("--split requires --manifest")
    if (args.manifest or args.task_spec) and any(
        value is not None for value in (args.horizon, args.size)
    ):
        parser.error("Saved tasks cannot be changed with layout arguments")
    if (args.manifest or args.task_spec) and (args.controlled or args.random_length):
        parser.error("Saved tasks cannot be changed with layout arguments")
    if args.task_seed is not None and (args.manifest or args.task_spec):
        parser.error("--task-seed only applies when generating a task")
    if min(args.total_timesteps, args.evaluation_interval, args.evaluation_episodes,
           args.required_consecutive_evals, args.torch_threads) <= 0:
        parser.error("Step, evaluation, and thread counts must be positive")
    if not 0 <= args.minimum_success_rate <= 1:
        parser.error("Success rate must be in [0, 1]")
    return args


def sample_task(args: argparse.Namespace, rng: random.Random) -> MemoryTaskSpec:
    """Pick one fixed task: from a pool, a saved spec, or a fresh freeze."""
    if args.manifest:
        pool = load_pool(args.manifest)
        candidates = pool["tasks"]
        if args.split:
            candidates = [task for task in candidates if task["split"] == args.split]
        if args.task_id:
            candidates = [task for task in candidates if task["task_id"] == args.task_id]
        if not candidates:
            raise ValueError(f"No matching task in {args.manifest}")
        value = candidates[0] if args.task_id else rng.choice(candidates)
        return MemoryTaskSpec.from_dict(value)
    if args.task_spec:
        spec = MemoryTaskSpec.from_dict(
            json.loads(Path(args.task_spec).read_text(encoding="utf-8")))
        if spec.configuration is None:
            raise ValueError("The convergence checker requires a fixed configuration, "
                             "not a changing-layout stream")
        return spec
    seed = args.task_seed if args.task_seed is not None else rng.randrange(2 ** 31)
    return freeze_task(MemoryTaskSpec(
        args.env_id, seed, "train",
        horizon=30 if args.horizon is None else args.horizon, size=args.size,
        controlled=args.controlled, random_length=args.random_length))


def plot_curve(evaluations: list[dict], spec: MemoryTaskSpec, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["timesteps"] for row in evaluations]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for axis, metric in zip(axes, ("success_rate", "mean_return")):
        axis.plot(steps, [row[metric] for row in evaluations], marker=".")
        axis.set(xlabel="Training interactions", ylabel=metric.replace("_", " ").title())
        axis.grid(alpha=0.2)
    axes[0].set_ylim(-0.03, 1.03)
    figure.suptitle(f"PPO on one fixed Memory task ({spec.task_id})")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def collection_config(args: argparse.Namespace) -> PPOConfig:
    config = PPOConfig()
    if args.ppo_config:
        value = load_config(args.ppo_config)
        if value.get("source_algorithm", "ppo") != "ppo":
            raise ValueError("The supplied collection config is not standard PPO")
        config = PPOConfig(**value.get("ppo", value))
    if config.policy != "MlpPolicy":
        raise ValueError("Collection PPO requires MlpPolicy")
    overrides = {key: getattr(args, key)
                 for key in ("n_steps", "batch_size", "n_epochs", "learning_rate")
                 if getattr(args, key) is not None}
    config = replace(config, **overrides)
    if config.n_steps < 2 or config.batch_size < 2:
        raise ValueError("Require n_steps and batch_size >= 2")
    return config


def run_check(args: argparse.Namespace) -> dict:
    rng = random.Random(args.sample_seed)
    spec = sample_task(args, rng)
    config = collection_config(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    _write_json(output / "task_spec.json", spec.to_dict())
    _write_json(output / "ppo_config.json", config.to_dict())

    env = make_memory_env(spec, flatten_for_source=True)
    evaluations: list[dict] = []
    metrics_path = output / "evaluations.json"

    def record(model) -> None:
        record_value = {"timesteps": int(model.num_timesteps),
                        **evaluate_ppo(model, spec, episodes=args.evaluation_episodes)}
        evaluations.append(record_value)
        _write_json(metrics_path, evaluations)
        print(json.dumps(record_value, sort_keys=True), flush=True)

    try:
        model = build_ppo(env, seed=args.seed, config=config, device=args.device,
                          verbose=args.verbose, tensorboard_log=output / "tensorboard")
        record(model)
        callback = _make_evaluation_callback(
            validation_spec=spec,
            evaluation_interval=args.evaluation_interval,
            evaluation_episodes=args.evaluation_episodes,
            evaluations=evaluations,
            metrics_path=metrics_path,
            evaluator=evaluate_ppo,
        )
        model.learn(args.total_timesteps, callback=callback,
                    progress_bar=not args.no_progress_bar)
        if evaluations[-1]["timesteps"] != int(model.num_timesteps):
            record(model)
        if args.save_model:
            model.save(output / "ppo-final")
    finally:
        env.close()

    converged = has_converged(evaluations[1:], minimum_success_rate=args.minimum_success_rate,
                              required_consecutive_evals=args.required_consecutive_evals)
    summary = {
        "status": "complete",
        "passed": converged,
        "task_spec": spec.to_dict(),
        "ppo_config": config.to_dict(),
        "learner_seed": args.seed,
        "criterion": {"minimum_success_rate": args.minimum_success_rate,
                      "required_consecutive_evals": args.required_consecutive_evals},
        "training_budget": args.total_timesteps,
        "evaluation_interval": args.evaluation_interval,
        "evaluation_episodes": args.evaluation_episodes,
        "evaluation_protocol": "Same fixed task, fresh environment, deterministic actions",
        "device": args.device,
        "converged": converged,
        "final_success_rate": evaluations[-1]["success_rate"],
        "peak_success_rate": max(row["success_rate"] for row in evaluations),
        "evaluations": evaluations,
    }
    _write_json(output / "summary.json", summary)
    plot_curve(evaluations, spec, output / "learning-curve.png")
    return summary


def main(argv: list[str] | None = None) -> None:
    summary = run_check(parse_args(argv))
    print("PASS" if summary["passed"] else "FAIL", flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
