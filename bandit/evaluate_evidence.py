"""Paired held-out evidence interventions; no online relearning after the query."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bandit.evidence_experiment import (CONDITIONS, PROTOCOL, EvidenceDataset,
    load_evidence_checkpoint, prepare_data)
from bandit.utils import file_digest, project_path, write_json

METRICS = ("expected_regret", "greedy_regret", "optimal_probability", "optimal_accuracy",
           "teacher_probability", "teacher_accuracy")


def score_probabilities(probabilities, data, indices):
    means = data.arrays["means"][indices]
    target = data.arrays["targets"][indices]
    best = means.argmax(axis=1)
    greedy = probabilities.argmax(axis=1)
    rows = np.arange(len(indices))
    return {"expected_regret": means.max(axis=1) - (means * probabilities).sum(axis=1),
            "greedy_regret": means.max(axis=1) - means[rows, greedy],
            "optimal_probability": probabilities[rows, best],
            "optimal_accuracy": (greedy == best).astype(float),
            "teacher_probability": probabilities[rows, target],
            "teacher_accuracy": (greedy == target).astype(float)}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for stratum in ("all", row["best_block"]):
            groups[(row["method"], row["condition"], stratum)].append(row)
    summaries = []
    for (method, condition, stratum), group in sorted(groups.items()):
        runs = defaultdict(list)
        for row in group:
            runs[row["seed"]].append(row)
        # Seeds are the replication unit for trained models. Baselines appear once.
        unit = "training_seed" if len(runs) > 1 else "test_task"
        summary = {"method": method, "condition": condition, "best_block": stratum,
                   "training_seeds": len(runs) if method not in ("empirical_available", "random") else 0,
                   "tasks_per_run": len(group) // len(runs), "uncertainty_unit": unit}
        for metric in METRICS:
            values = ([np.mean([r[metric] for r in run]) for run in runs.values()] if len(runs) > 1
                      else [r[metric] for r in group])
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_se"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def paired_effects(rows):
    indexed = {(r["method"], r["seed"], r["task"], r["condition"]): r for r in rows}
    differences = []
    for row in rows:
        if row["condition"] != "both":
            continue
        for condition in ("early_only", "recent_only"):
            other = indexed[(row["method"], row["seed"], row["task"], condition)]
            differences.append({**row, "condition": f"{condition}_minus_both",
                                **{metric: other[metric] - row[metric] for metric in METRICS}})
    return summarize(differences)


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [method for method in ("rad", "ad_short", "ad_long") if any(r["method"] == method for r in summary)]
    colors = {"rad": "#009E73", "ad_short": "#E69F00", "ad_long": "#0072B2"}
    labels = {"rad": "RAD", "ad_short": "AD-short", "ad_long": "AD-long"}
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
        width = 0.75 / max(1, len(methods))
        for axis, stratum in zip(axes, ("early", "recent")):
            for i, method in enumerate(methods):
                values = [next(r for r in summary if r["method"] == method and r["condition"] == condition
                               and r["best_block"] == stratum) for condition in CONDITIONS]
                x = np.arange(len(CONDITIONS)) + (i - (len(methods) - 1) / 2) * width
                axis.bar(x, [v["expected_regret"] for v in values], width,
                         yerr=[v["expected_regret_se"] for v in values], capsize=2,
                         color=colors[method], label=labels[method])
            axis.set(title=f"Best arm in {stratum} block", xticks=np.arange(3),
                     xticklabels=["Both", "Early only", "Recent only"])
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        upper = max(r["expected_regret"] + r["expected_regret_se"] for r in summary
                    if r["method"] in methods and r["best_block"] in ("early", "recent"))
        axes[0].set_ylim(0, max(0.01, upper * 1.08))
        axes[0].set_ylabel("First-query expected pseudo-regret")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(Path(output) / "evidence_integration.png", dpi=240)
        fig.savefig(Path(output) / "evidence_integration.pdf")
        plt.close(fig)


def evaluate_runs(checkpoints, data_dir, output, *, device="cpu", batch_size=128):
    if not checkpoints or batch_size < 1:
        raise ValueError("Provide checkpoints and a positive batch size")
    output, data_dir = Path(output), Path(data_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output must be empty: {output}")
    rows, provenance, seen = [], [], set()
    reference = None

    def record(method, seed, condition, indices, probabilities):
        scores = score_probabilities(probabilities, data, indices)
        for position, index in enumerate(indices):
            rows.append({"method": method, "seed": seed, "condition": condition, "task": int(index),
                "best_block": "early" if data.arrays["best_is_early"][index] else "recent",
                **{metric: float(values[position]) for metric, values in scores.items()}})

    for checkpoint in checkpoints:
        model, payload = load_evidence_checkpoint(checkpoint, device)
        if reference is None:
            reference = payload
            manifest = prepare_data(data_dir, payload["study"])
            data = EvidenceDataset(data_dir / "test.npz", payload["study"])
        if payload["data_digests"] != manifest["digests"] or payload["study"] != reference["study"]:
            raise ValueError("All compared checkpoints must share the same study and exact data")
        identity = (payload["method"], payload["seed"])
        if identity in seen:
            raise ValueError("Duplicate method/training-seed checkpoint would bias aggregation")
        seen.add(identity)
        model.eval()
        context = model.context_steps if payload["method"] != "rad" else None
        with torch.inference_mode():
            for condition in CONDITIONS:
                for offset in range(0, data.size, batch_size):
                    indices = np.arange(offset, min(offset + batch_size, data.size))
                    batch = data.batch(indices, condition, context, device)
                    state = model.prefix_state(batch)
                    if payload["method"] == "rad":
                        if (state["compression_count"] != data.spec["compression_count"] or
                                state["recent"].shape[1] != 3 * data.spec["rad_recent_steps"]):
                            raise AssertionError("RAD state violates the evidence geometry")
                    probabilities = model.query_logits(state, batch["query_states"]).double().softmax(-1).cpu().numpy()
                    record(*identity, condition, indices, probabilities)
        source = Path(checkpoint)
        source = source / "model.pt" if source.is_dir() else source
        provenance.append({"path": str(source.resolve()), "sha256": file_digest(source),
                           "method": identity[0], "seed": identity[1], "step": payload["step"]})
        del model
        print(f"Evaluated {identity[0]} seed {identity[1]}", flush=True)

    indices = np.arange(data.size)
    arms = data.study["num_arms"]
    for condition in CONDITIONS:
        probabilities = np.zeros((data.size, arms))
        for index in indices:
            available = np.ones(arms, dtype=bool)
            if condition != "both":
                available[:] = condition == "recent_only"
                available[data.arrays["early_arms"][index]] = condition == "early_only"
            scores = np.where(available, data.arrays["empirical_means"][index], -np.inf)
            probabilities[index, scores.argmax()] = 1
        record("empirical_available", -1, condition, indices, probabilities)
        record("random", -1, condition, indices, np.full((data.size, arms), 1 / arms))
    summary = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(output / "summary.json", summary)
    write_json(output / "paired_effects.json", paired_effects(rows))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot_summary(summary, output)
    write_json(output / "evaluation.json", {"protocol": PROTOCOL, "complete": True,
        "study": reference["study"], "data_digests": manifest["digests"], "checkpoints": provenance,
        "conditions": list(CONDITIONS), "checkpoint_selection": "explicit paths",
        "plot_error_bars": "one standard error; see uncertainty_unit in summary",
        "note": "Models train on both blocks; missing-block conditions are inference interventions."})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    evaluate_runs([project_path(p) for p in args.checkpoint], project_path(args.dataset),
                  project_path(args.output), device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
