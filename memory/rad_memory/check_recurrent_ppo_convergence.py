"""Train and verify that the collection RecurrentPPO converges on Memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .envs import MemoryTaskSpec, make_memory_env
from .recurrent_ppo import (
    RecurrentPPOConfig,
    build_recurrent_ppo,
    evaluate_recurrent_ppo,
)


def has_converged(
    evaluations: list[dict[str, Any]],
    *,
    minimum_success_rate: float,
    required_consecutive_evals: int,
) -> bool:
    """Require the final evaluations to remain above the success threshold."""

    if required_consecutive_evals <= 0:
        raise ValueError("required_consecutive_evals must be positive")
    if len(evaluations) < required_consecutive_evals:
        return False
    tail = evaluations[-required_consecutive_evals:]
    return all(float(item["success_rate"]) >= minimum_success_rate for item in tail)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _make_evaluation_callback(
    *,
    validation_spec: MemoryTaskSpec,
    evaluation_interval: int,
    evaluation_episodes: int,
    evaluations: list[dict[str, Any]],
    metrics_path: Path,
):
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as error:
        raise RuntimeError(
            "The convergence check requires stable-baselines3; finish setup first"
        ) from error

    class _ConvergenceEvaluationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_evaluation = evaluation_interval

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_evaluation:
                return True
            metrics = evaluate_recurrent_ppo(
                self.model,
                validation_spec,
                episodes=evaluation_episodes,
            )
            record = {"timesteps": int(self.num_timesteps), **metrics}
            evaluations.append(record)
            _write_json(metrics_path, evaluations)
            print(json.dumps(record, sort_keys=True), flush=True)
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += evaluation_interval
            return True

    return _ConvergenceEvaluationCallback()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = RecurrentPPOConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Train collection-config RecurrentPPO learners and require stable "
            "held-out MiniGrid Memory success"
        )
    )
    parser.add_argument("--env-id", default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--size", type=int)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--random-length", action="store_true")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--validation-seed", type=int, default=10000)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--evaluation-interval", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--minimum-success-rate", type=float, default=0.9)
    parser.add_argument("--required-consecutive-evals", type=int, default=3)
    parser.add_argument("--required-seed-fraction", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--n-epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--output-dir", default="runs/recurrent-ppo-convergence")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)
    if args.total_timesteps <= 0 or args.evaluation_interval <= 0:
        parser.error("timesteps and evaluation interval must be positive")
    if args.evaluation_episodes <= 0 or args.required_consecutive_evals <= 0:
        parser.error("evaluation counts must be positive")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        parser.error("minimum success rate must be in [0, 1]")
    if not 0.0 < args.required_seed_fraction <= 1.0:
        parser.error("required seed fraction must be in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_config = RecurrentPPOConfig(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        n_epochs=args.n_epochs,
    )
    run_results = []
    for seed in args.seeds:
        seed_dir = output_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_spec = MemoryTaskSpec(
            env_id=args.env_id,
            seed=seed,
            split="train",
            horizon=args.horizon,
            controlled=args.controlled,
            size=args.size,
            random_length=args.random_length,
        )
        validation_spec = MemoryTaskSpec(
            env_id=args.env_id,
            seed=args.validation_seed,
            split="validation",
            horizon=args.horizon,
            controlled=args.controlled,
            size=args.size,
            random_length=args.random_length,
        )
        _write_json(seed_dir / "task_spec.json", train_spec.to_dict())
        _write_json(seed_dir / "recurrent_ppo_config.json", ppo_config.to_dict())
        env = make_memory_env(train_spec, flatten_for_source=True)
        model = build_recurrent_ppo(
            env,
            seed=seed,
            config=ppo_config,
            tensorboard_log=seed_dir / "tensorboard",
        )
        evaluations: list[dict[str, Any]] = []
        initial = evaluate_recurrent_ppo(
            model,
            validation_spec,
            episodes=args.evaluation_episodes,
        )
        evaluations.append({"timesteps": 0, **initial})
        metrics_path = seed_dir / "evaluations.json"
        _write_json(metrics_path, evaluations)
        print(json.dumps(evaluations[-1], sort_keys=True), flush=True)
        callback = _make_evaluation_callback(
            validation_spec=validation_spec,
            evaluation_interval=args.evaluation_interval,
            evaluation_episodes=args.evaluation_episodes,
            evaluations=evaluations,
            metrics_path=metrics_path,
        )
        try:
            model.learn(
                args.total_timesteps,
                callback=callback,
                progress_bar=not args.no_progress_bar,
            )
            if evaluations[-1]["timesteps"] != int(model.num_timesteps):
                final_metrics = evaluate_recurrent_ppo(
                    model,
                    validation_spec,
                    episodes=args.evaluation_episodes,
                )
                evaluations.append({"timesteps": int(model.num_timesteps), **final_metrics})
                _write_json(metrics_path, evaluations)
                print(json.dumps(evaluations[-1], sort_keys=True), flush=True)
            if args.save_models:
                model.save(seed_dir / "teacher-final")
        finally:
            env.close()

        converged = has_converged(
            evaluations,
            minimum_success_rate=args.minimum_success_rate,
            required_consecutive_evals=args.required_consecutive_evals,
        )
        result = {
            "seed": seed,
            "converged": converged,
            "final_success_rate": evaluations[-1]["success_rate"],
            "peak_success_rate": max(item["success_rate"] for item in evaluations),
            "evaluations": evaluations,
        }
        run_results.append(result)
        _write_json(seed_dir / "result.json", result)

    converged_seeds = sum(int(item["converged"]) for item in run_results)
    converged_fraction = converged_seeds / len(run_results)
    passed = converged_fraction >= args.required_seed_fraction
    summary = {
        "passed": passed,
        "environment": {
            "env_id": args.env_id,
            "controlled": args.controlled,
            "random_length": args.random_length,
            "size": args.size,
            "horizon": args.horizon,
            "validation_seed": args.validation_seed,
        },
        "recurrent_ppo_config": ppo_config.to_dict(),
        "criterion": {
            "minimum_success_rate": args.minimum_success_rate,
            "required_consecutive_evals": args.required_consecutive_evals,
            "required_seed_fraction": args.required_seed_fraction,
        },
        "converged_seeds": converged_seeds,
        "total_seeds": len(run_results),
        "converged_seed_fraction": converged_fraction,
        "runs": run_results,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
