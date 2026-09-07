"""Summarize paired evaluation JSON files into CSV and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def _condition(run_name: str) -> str:
    return re.sub(r"-h\d+-seed\d+$", "", run_name)


def _train_seed(run_name: str) -> int:
    match = re.search(r"-seed(\d+)$", run_name)
    if match is None:
        raise ValueError(f"Run directory lacks a training seed: {run_name}")
    return int(match.group(1))


def _record_key(record):
    if "trial" in record:
        return (int(record["train_seed"]), record["task_id"], int(record["trial"]), int(record["episode"]))
    return (int(record["train_seed"]), int(record["seed"]))


def _bootstrap(values: np.ndarray, rng: np.random.Generator, samples: int = 10000):
    if not len(values):
        return None, None
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MiniGrid Memory evaluations")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args()
    runs = []
    for path in sorted(Path(args.input_root).rglob("eval*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload["summary"]
        metrics_path = path.parent / "run_metrics.json"
        run_metrics = {}
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                run_metrics = json.load(handle)
        train_seed = _train_seed(path.parent.name)
        records = [dict(record) | {"train_seed": train_seed} for record in payload["records"]]
        runs.append(
            {
                "run": path.parent.name,
                "condition": _condition(path.parent.name) + ("-reset" if summary.get("reset_context_each_episode") else ""),
                "train_seed": train_seed,
                "model": summary["model"],
                "n_transit": summary["n_transit"],
                "success_rate": summary["success_rate"],
                "out_of_window_success_rate": summary["out_of_window_success_rate"],
                "mean_compressions": summary["mean_compressions"],
                "episodes": summary["episodes"],
                "records": records,
                "run_metrics": run_metrics,
            }
        )
    if not runs:
        raise ValueError(f"No eval*.json files found below {args.input_root}")
    rng = np.random.default_rng(0)
    by_condition = {}
    for run in runs:
        by_condition.setdefault(run["condition"], []).extend(run["records"])
    baseline_by_seed = {
        _record_key(record): float(record["success"])
        for record in by_condition.get("ad-short", [])
    }
    rows = []
    for condition, records in sorted(by_condition.items()):
        success = np.asarray([float(record["success"]) for record in records])
        train_seeds = sorted({int(record["train_seed"]) for record in records})
        seed_means = np.asarray(
            [
                np.mean(
                    [
                        float(record["success"])
                        for record in records
                        if int(record["train_seed"]) == train_seed
                    ]
                )
                for train_seed in train_seeds
            ]
        )
        outside = np.asarray(
            [
                float(record["success"])
                for record in records
                if record["cue_outside_active_context"]
            ]
        )
        low, high = _bootstrap(seed_means, rng)
        paired_by_train_seed = []
        for train_seed in train_seeds:
            differences = [
                float(record["success"])
                - baseline_by_seed[_record_key(record)]
                for record in records
                if int(record["train_seed"]) == train_seed
                and _record_key(record) in baseline_by_seed
            ]
            if differences:
                paired_by_train_seed.append(float(np.mean(differences)))
        paired = np.asarray(paired_by_train_seed)
        condition_runs = [run for run in runs if run["condition"] == condition]
        elapsed = [run["run_metrics"].get("elapsed_seconds") for run in condition_runs]
        elapsed = [value for value in elapsed if value is not None]
        peak = [run["run_metrics"].get("peak_gpu_memory_bytes") for run in condition_runs]
        peak = [value for value in peak if value is not None]
        parameters = [run["run_metrics"].get("trainable_parameters") for run in condition_runs]
        parameters = [value for value in parameters if value is not None]
        delta_low, delta_high = _bootstrap(paired, rng)
        rows.append(
            {
                "condition": condition,
                "train_seeds": len(train_seeds),
                "episodes": len(success),
                "success_rate": float(success.mean()),
                "success_ci_low": low,
                "success_ci_high": high,
                "out_of_window_episodes": len(outside),
                "out_of_window_success_rate": float(outside.mean()) if len(outside) else None,
                "delta_vs_ad_short": float(paired.mean()) if len(paired) else None,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "mean_elapsed_seconds": float(np.mean(elapsed)) if elapsed else None,
                "mean_peak_gpu_memory_bytes": float(np.mean(peak)) if peak else None,
                "trainable_parameters": int(parameters[0]) if parameters else None,
            }
        )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [row["condition"] for row in rows]
    values = [row["success_rate"] for row in rows]
    errors = np.asarray(
        [
            [row["success_rate"] - row["success_ci_low"] for row in rows],
            [row["success_ci_high"] - row["success_rate"] for row in rows],
        ]
    )
    figure_path = Path(args.figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, 0.6 * len(rows)), 5))
    plt.bar(labels, values, yerr=errors, capsize=4)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="binary chance")
    plt.ylim(0, 1)
    plt.ylabel("Success rate")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)


if __name__ == "__main__":
    main()
