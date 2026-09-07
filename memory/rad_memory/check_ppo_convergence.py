"""Check the collection PPO learner on one fixed Memory task."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .check_recurrent_ppo_convergence import _write_json
from .envs import MemoryTaskSpec
from .ppo import PPOConfig
from .task_pool import fingerprint, freeze_task, load_pool
from .train_task_pool import train_task


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", help="Existing task pool; select exactly one task with --task-id")
    source.add_argument("--task-spec", help="Saved fixed task_spec.json")
    source.add_argument("--env-id", help="Generate and freeze one task (default: MiniGrid-MemoryS13Random-v0)")
    parser.add_argument("--task-id")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--random-length", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Independent PPO seeds; layout stays fixed")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--evaluation-interval", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--minimum-success-rate", type=float, default=0.9)
    parser.add_argument("--required-consecutive-evals", type=int, default=3)
    parser.add_argument("--required-seed-fraction", type=float, default=1.0)
    parser.add_argument("--ppo-config", help="Collection source_config.json or a raw PPOConfig JSON")
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/ppo-convergence")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)
    if bool(args.manifest) != bool(args.task_id):
        parser.error("--manifest and --task-id must be supplied together")
    if (args.manifest or args.task_spec) and (
        args.horizon is not None or args.size is not None or args.controlled or args.random_length or args.task_seed != 0
    ):
        parser.error("Saved tasks cannot be changed with layout/horizon arguments")
    if min(args.total_timesteps, args.evaluation_interval, args.evaluation_episodes,
           args.required_consecutive_evals, args.torch_threads) <= 0:
        parser.error("Step, evaluation, and thread counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Source seeds must be distinct nonnegative integers")
    if not 0 <= args.minimum_success_rate <= 1 or not 0 < args.required_seed_fraction <= 1:
        parser.error("Success rate must be in [0, 1] and required seed fraction in (0, 1]")
    if args.horizon is not None and args.horizon <= 0:
        parser.error("horizon must be positive")
    return args


def select_task(args):
    if args.manifest:
        pool = load_pool(args.manifest)
        matches = [task for task in pool["tasks"] if task["task_id"] == args.task_id]
        if len(matches) != 1:
            raise ValueError(f"Task {args.task_id!r} is not in the manifest")
        value = matches[0]
    elif args.task_spec:
        value = json.loads(Path(args.task_spec).read_text(encoding="utf-8"))
    else:
        return freeze_task(MemoryTaskSpec(
            args.env_id or "MiniGrid-MemoryS13Random-v0", args.task_seed, "train",
            horizon=30 if args.horizon is None else args.horizon, size=args.size,
            controlled=args.controlled, random_length=args.random_length))
    spec = MemoryTaskSpec.from_dict(value)
    if spec.configuration is None:
        raise ValueError("The convergence checker requires a fixed configuration, not a changing-layout stream")
    if value.get("task_id", spec.task_id) != spec.task_id:
        raise ValueError("Saved task ID does not match its configuration")
    return spec


def collection_config(args):
    config = PPOConfig()
    if args.ppo_config:
        value = json.loads(Path(args.ppo_config).read_text(encoding="utf-8"))
        if value.get("source_algorithm", "ppo") != "ppo":
            raise ValueError("The supplied collection config is not standard PPO")
        config = PPOConfig(**value.get("ppo", value))
    overrides = {key: getattr(args, key) for key in ("n_steps", "batch_size") if getattr(args, key) is not None}
    config = replace(config, **overrides)
    if config.policy != "MlpPolicy" or config.n_steps < 2 or config.batch_size < 2:
        raise ValueError("Require collection MlpPolicy with n_steps and batch_size >= 2")
    return config


def plot_curves(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for run in runs:
        rows = run["evaluations"]
        steps = [row["timesteps"] for row in rows]
        for axis, metric in zip(axes, ("success_rate", "mean_return")):
            axis.plot(steps, [row[metric] for row in rows], marker=".", label=f"Seed {run['source_seed']}")
    for axis, title in zip(axes, ("Success rate", "Mean episode return")):
        axis.set(xlabel="Training interactions", ylabel=title)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylim(-0.03, 1.03)
    figure.suptitle("Collection PPO on one fixed Memory task")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_check(args):
    original_spec = select_task(args)
    config = collection_config(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.torch_threads)
    # A diagnostic may target a held-out task, but must never masquerade as
    # collection data from its original train/test manifest.
    spec = replace(original_spec, split="train")
    diagnostic_id = fingerprint({"kind": "single-task-convergence-probe", "task": original_spec.to_dict()})
    summary = {
        "status": "running", "passed": False, "task_spec": original_spec.to_dict(),
        "diagnostic_fingerprint": diagnostic_id, "ppo_config": config.to_dict(),
        "criterion": {"minimum_success_rate": args.minimum_success_rate,
                      "required_consecutive_evals": args.required_consecutive_evals,
                      "required_seed_fraction": args.required_seed_fraction},
        "requested_seeds": args.seeds, "runs": [],
        "training_budget": args.total_timesteps, "evaluation_interval": args.evaluation_interval,
        "evaluation_episodes": args.evaluation_episodes, "device": args.device,
        "evaluation_protocol": "Separate evaluation environment, same fixed configuration, deterministic actions",
    }
    _write_json(output / "task_spec.json", original_spec.to_dict())
    _write_json(output / "summary.json", summary)
    try:
        for seed in args.seeds:
            result = train_task(
                spec, diagnostic_id, seed, output / "source", output / "histories",
                source_algorithm="ppo", ppo_config=config, device=args.device, verbose=args.verbose,
                total_timesteps=args.total_timesteps, evaluation_interval=args.evaluation_interval,
                evaluation_episodes=args.evaluation_episodes, minimum_success_rate=args.minimum_success_rate,
                required_consecutive_evals=args.required_consecutive_evals)
            run_dir = output / "source" / Path(result["artifact"]).stem
            evaluations = json.loads((run_dir / "evaluations.json").read_text(encoding="utf-8"))
            summary["runs"].append(result | {"evaluations": evaluations,
                                               "final_success_rate": evaluations[-1]["success_rate"]})
            _write_json(output / "summary.json", summary)
            print(json.dumps({"seed": seed, "converged": result["converged"],
                              "final_success_rate": evaluations[-1]["success_rate"]}), flush=True)
        fraction = sum(run["converged"] for run in summary["runs"]) / len(args.seeds)
        plot_curves(summary["runs"], output / "learning-curves.png")
        summary.update(status="complete", passed=fraction >= args.required_seed_fraction,
                       converged_seed_fraction=fraction)
    except Exception as error:
        summary.update(status="error", error=str(error))
        _write_json(output / "summary.json", summary)
        raise
    _write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    summary = run_check(parse_args(argv))
    print("PASS" if summary["passed"] else "FAIL", flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
