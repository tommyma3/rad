"""Generate or execute the context-window experiment matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def _command(module: str, arguments: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def _argument(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _expected_output(name: str, command: list[str]) -> Path | None:
    if name.startswith("evaluate-"):
        value = _argument(command, "--output")
        return Path(value) if value else None
    run_dir = _argument(command, "--run-dir")
    if run_dir is None:
        return None
    filename = "pretrain-final.pt" if name == "rad-pretrain" else "final.pt"
    return Path(run_dir) / filename


def build_commands(
    horizon: int,
    short_context: int,
    seed: int,
    config: str,
    data_root: str,
    runs_root: str,
    evaluation: dict,
    compute_matched_steps: int,
) -> list[tuple[str, list[str]]]:
    short = int(short_context)
    half = int(math.ceil(0.5 * horizon))
    full = (11 * horizon + 9) // 10
    double = 2 * horizon
    latent_steps = max(1, int(math.ceil(0.25 * short)))
    n_compress_tokens = 3 * latent_steps
    short_keep = max(1, int(math.ceil(0.25 * short)))
    common = [
        "--config", config,
        "--data-root", data_root,
        "--override", f"seed={seed}",
    ]
    if evaluation.get("manifest"):
        common.extend(["--override", "history_scope=task", "--override",
                       f"task_manifest={evaluation['manifest']}"])
    commands: list[tuple[str, list[str]]] = []
    trained_runs: list[tuple[str, Path]] = []
    ad_conditions = (
        ("ad-reactive", 1, None),
        ("ad-short", short, None),
        ("ad-short-extra-compute", short, compute_matched_steps),
        ("ad-half-episode", half, None),
        ("ad-full", full, None),
        ("ad-2x", double, None),
    )
    for name, context, train_steps in ad_conditions:
        run_dir = str(Path(runs_root) / f"{name}-h{horizon}-seed{seed}")
        condition_overrides = ["--override", f"n_transit={context}"]
        if train_steps is not None:
            condition_overrides.extend(["--override", f"train_steps={train_steps}"])
        commands.append(
            (
                name,
                _command(
                    "rad_memory.train_ad",
                    [*common, "--run-dir", run_dir, *condition_overrides],
                ),
            )
        )
        trained_runs.append((name, Path(run_dir)))
    pretrain_dir = Path(runs_root) / f"rad-pretrain-h{horizon}-seed{seed}"
    rad_overrides = [
        "--override", f"n_transit={short}",
        "--override", f"max_context_length={double}",
        "--override", f"n_compress_tokens={n_compress_tokens}",
        "--override", f"short_memory_keep={short_keep}",
    ]
    commands.append(
        (
            "rad-pretrain",
            _command(
                "rad_memory.train_pretrain_compression",
                [*common, "--run-dir", str(pretrain_dir), *rad_overrides],
            ),
        )
    )
    trained_runs.append(("rad-short", Path(runs_root) / f"rad-short-h{horizon}-seed{seed}"))
    commands.append(
        (
            "rad-short",
            _command(
                "rad_memory.train_rad",
                [
                    *common,
                    "--run-dir", str(Path(runs_root) / f"rad-short-h{horizon}-seed{seed}"),
                    "--pretrain-checkpoint", str(pretrain_dir / "pretrain-final.pt"),
                    *rad_overrides,
                ],
            ),
        )
    )
    commands.append(
        (
            "rad-no-pretrain",
            _command(
                "rad_memory.train_rad",
                [
                    *common,
                    "--run-dir", str(Path(runs_root) / f"rad-no-pretrain-h{horizon}-seed{seed}"),
                    *rad_overrides,
                ],
            ),
        )
    )
    trained_runs.append(("rad-no-pretrain", Path(runs_root) / f"rad-no-pretrain-h{horizon}-seed{seed}"))
    for name, run_dir in trained_runs:
        arguments = [
            "--checkpoint", str(run_dir / "final.pt"),
            "--env-id", evaluation["env_id"],
            "--horizon", str(horizon),
            "--seed", str(evaluation["seed"]),
            "--episodes", str(evaluation["episodes"]),
            "--output", str(run_dir / "eval.json"),
        ]
        if evaluation.get("controlled"):
            arguments.append("--controlled")
        if evaluation.get("random_length"):
            arguments.append("--random-length")
        if evaluation.get("size") is not None:
            arguments.extend(["--size", str(evaluation["size"])])
        if evaluation.get("manifest"):
            arguments.extend(["--manifest", evaluation["manifest"], "--trials",
                              str(evaluation.get("trials", 3))])
        commands.append((f"evaluate-{name}", _command("rad_memory.evaluate", arguments)))
        if evaluation.get("manifest"):
            reset_arguments = list(arguments)
            reset_arguments[reset_arguments.index("--output") + 1] = str(run_dir / "eval-reset.json")
            reset_arguments.append("--reset-context-each-episode")
            commands.append((f"evaluate-{name}-reset", _command("rad_memory.evaluate", reset_arguments)))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MiniGrid Memory context sweep")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--runs-root", default="runs/context-sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="*", default=[])
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with Path(args.profile).open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if args.manifest:
        from .task_pool import load_pool
        pool = load_pool(args.manifest)
        if profile.get("manifest_fingerprint") != pool["fingerprint"]:
            raise ValueError("Profile and task manifest do not match")
    horizon = int(profile["recommended_horizon"])
    short_context = int(profile["short_context"])
    profile_spec = profile["task_spec"]
    from .utils import load_config

    base_config = load_config(args.config)
    compute_matched_steps = int(base_config["train_steps"]) + int(base_config["pretrain_steps"])
    evaluation = {
        "manifest": args.manifest,
        "trials": args.trials,
        "env_id": profile_spec["env_id"],
        "controlled": profile_spec.get("controlled", False),
        "random_length": profile_spec.get("random_length", False),
        "size": profile_spec.get("size"),
        "seed": args.eval_seed,
        "episodes": args.eval_episodes,
    }
    matrix = []
    command_index = 0
    for seed in args.seeds:
        for name, command in build_commands(
            horizon,
            short_context,
            seed,
            args.config,
            args.data_root,
            args.runs_root,
            evaluation,
            compute_matched_steps,
        ):
            gpu = args.gpus[command_index % len(args.gpus)] if args.gpus else None
            matrix.append({"name": name, "seed": seed, "gpu": gpu, "command": command})
            command_index += 1
    for item in matrix:
        prefix = f"CUDA_VISIBLE_DEVICES={item['gpu']} " if item["gpu"] is not None else ""
        print(prefix + subprocess.list2cmdline(item["command"]))
    if not args.execute:
        return
    # Commands are intentionally sequential: RAD fine-tuning depends on its
    # pretraining artifact. Parallelism can be added per independent seed.
    for item in matrix:
        expected = _expected_output(item["name"], item["command"])
        if expected is not None and expected.exists() and not args.force:
            print(f"Skipping completed {item['name']}: {expected}")
            continue
        env = os.environ.copy()
        if item["gpu"] is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
        subprocess.run(item["command"], check=True, env=env)


if __name__ == "__main__":
    main()
