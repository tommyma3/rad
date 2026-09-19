"""Delay-sweep evaluation over all training runs, averaging results across runs.

Checkpoints are discovered automatically under ``--runs_dir``: each training
run contributes its max-iteration checkpoint, runs named ``<family>_s<seed>``
share one method label, and the summary averages over the runs in each family.

Rollout results are cached under ``--cache_dir`` per method, evaluation seed,
and delay, so rerunning after plotting-only changes reuses cached rollouts and
regenerates just the summary and figures (pass ``--force`` to reevaluate).
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bandit.evaluation import discover_run_checkpoints, evaluate_suite
from bandit.utils import get_config, project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", "--runs-dir", default="runs",
                        help="Training-runs directory to discover checkpoints under")
    parser.add_argument("--env", default="delayed_adversarial_bandit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--eval_seeds", "--eval-seeds", type=int, default=5,
                        help="Independent evaluation seed runs to average")
    parser.add_argument("--delays", type=int, nargs="+")
    parser.add_argument("--distributions", nargs="+", choices=("uniform",), default=["uniform"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for distilled AD/RAD policies; baselines are CPU-only")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument("--shared_prefix", action="store_true")
    parser.add_argument("--reference_context", type=int, default=50)
    parser.add_argument("--manifest")
    parser.add_argument("--cache_dir", "--cache-dir", default="results/eval_cache",
                        help="Rollout cache directory; reruns reuse cached blocks and only regenerate outputs")
    parser.add_argument("--force", action="store_true", help="Recompute cached evaluations")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.eval_seeds < 1:
        parser.error("eval_seeds must be positive")
    discovered, skipped = discover_run_checkpoints(project_path(args.runs_dir))
    if not discovered:
        parser.error(f"No completed distilled runs found under {project_path(args.runs_dir)}")
    checkpoints, labels = [], []
    for family, runs in sorted(discovered.items()):
        for item in runs:
            print(f"Discovered method={family} run={item['run']} step={item['step']}: "
                  f"{item['checkpoint']}")
            checkpoints.append(item["checkpoint"])
            labels.append(family)
    for run_name in skipped:
        print(f"Skipped run={run_name}: max-iteration checkpoint is not a distilled policy")
    torch.set_num_threads(args.threads)
    config = get_config(f"config/env/{args.env}.yaml")
    config.update(get_config("config/algorithm/ucb.yaml"))
    evaluate_suite(checkpoints, project_path(args.output), config,
                   tasks=args.tasks, seed=args.seed, eval_seeds=args.eval_seeds,
                   delays=args.delays, labels=labels,
                   distributions=args.distributions, device=args.device, sample=not args.greedy,
                   include_baselines=not args.no_baselines, shared_prefix=args.shared_prefix,
                   reference_context=args.reference_context,
                   manifest_path=project_path(args.manifest) if args.manifest else None,
                   cache_dir=project_path(args.cache_dir), force=args.force)


if __name__ == "__main__":
    main()
