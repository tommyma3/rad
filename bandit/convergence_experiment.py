"""Periodic online pseudo-regret for the standard AD/RAD bandit models."""
import csv
import json
from pathlib import Path

import numpy as np

from .dataset import BanditDataset
from .env import BanditTask
from .evaluation import ModelPolicy
from .rollout import generate_history
from .training import train
from .utils import file_digest, write_json

PROTOCOL = "bandit-training-convergence-v1"
METHODS = ("ad_short", "ad_long", "rad")
LABELS = {"ad_short": "AD-short", "ad_long": "AD-long", "rad": "RAD"}


def evaluation_steps(steps, interval):
    if steps < 1 or interval < 1:
        raise ValueError("Steps and interval must be positive")
    return sorted({0, steps, *range(interval, steps + 1, interval)})


def evaluate_model(model, manifest, delays, *, pre_steps=50, post_steps=50, greedy=False):
    """Expected gaps under sampled online actions, not noisy reward differences."""
    rows = []
    for delay in delays:
        for record in manifest["records"]:
            task = BanditTask.from_dict(record["task"])
            policy = ModelPolicy(model, record["policy_seed"], sample=not greedy)
            history, _ = generate_history(task, delay=delay, pre_steps=pre_steps,
                post_steps=post_steps, reward_seed=record["reward_seed"],
                learner_seed=record["learner_seed"], distractor_seed=record["distractor_seed"],
                policy=policy)
            actions = history["actions"][history["loss_mask"]]
            if len(actions) != pre_steps + post_steps:
                raise ValueError("Wrong genuine-pull horizon")
            means = np.asarray(task.means)
            curve = np.cumsum(means.max() - means[actions])
            rows.append({"task_id": task.task_id, "delay": delay,
                "cumulative_expected_regret": float(curve[-1]),
                "cumulative_regret_curve": curve.tolist()})
    return rows


def verify_data(plan, root):
    dataset = Path(plan["dataset"])
    actual = {split: file_digest(dataset / f"{split}.hdf5") for split in ("train", "validation")}
    if actual != plan["data_digests"]:
        raise ValueError("Dataset changed since experiment preparation")
    manifest = json.loads((Path(root) / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if file_digest(Path(root) / "evaluation_manifest.json") != plan["manifest_digest"]:
        raise ValueError("Evaluation manifest changed")
    return manifest


def verify_held_out(dataset, manifest):
    signatures = {tuple(r["task"]["means"]) for r in manifest["records"]}
    for split in ("train", "validation"):
        data = BanditDataset(Path(dataset) / f"{split}.hdf5", expected_split=split)
        if signatures & data.task_signatures:
            raise ValueError("Evaluation tasks overlap the training/validation collection")


def train_worker(root, method, seed, *, cpu=False, resume=False):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan["protocol"] != PROTOCOL or method not in METHODS or seed not in plan["seeds"]:
        raise ValueError("Worker does not match experiment plan")
    manifest = verify_data(plan, root)
    config = {**plan["configs"][method], "seed": seed}
    run = root / f"{method}_s{seed}"
    points = root / "evaluations" / run.name
    steps = evaluation_steps(config["train_steps"], config["eval_interval"])
    checkpoint = None
    if resume and (run / "latest.json").exists():
        latest = json.loads((run / "latest.json").read_text(encoding="utf-8"))
        checkpoint = run / latest["checkpoint"]
        metadata = json.loads((checkpoint / "training.json").read_text(encoding="utf-8"))
        saved_config = {key: value for key, value in metadata["config"].items() if key != "collection_config"}
        if (saved_config != config or metadata["data_digests"] != plan["data_digests"] or
                metadata["step"] != latest["step"] or metadata["world_size"] != 1 or
                metadata["phase"] != "distill" or not (checkpoint / "model.pt").is_file()):
            raise ValueError("Checkpoint does not match the frozen experiment plan")
        # A complete checkpoint must also have every periodic evaluation.
        if latest["step"] == config["train_steps"]:
            if not all((points / f"step-{s:07d}.json").exists() for s in steps):
                raise ValueError("Completed run is missing evaluation points")
            return checkpoint
        if not all((points / f"step-{s:07d}.json").exists() for s in steps if s < latest["step"]):
            raise ValueError("Resumed run is missing earlier evaluation points")
    elif resume and run.exists() and any(run.iterdir()):
        raise ValueError(f"Interrupted before the first checkpoint; use a fresh experiment root: {run}")

    def callback(model, step):
        rows = evaluate_model(model, manifest, plan["delays"], greedy=plan["greedy"])
        write_json(points / f"step-{step:07d}.json", {"protocol": PROTOCOL,
            "method": method, "seed": seed, "step": step, "rows": rows})
        print(json.dumps({"step": step, "online_regret": {
            str(delay): float(np.mean([r["cumulative_expected_regret"] for r in rows if r["delay"] == delay]))
            for delay in plan["delays"]}}), flush=True)

    return train(config, plan["dataset"], run, resume=checkpoint, cpu=cpu,
                 evaluation_callback=callback)


def plot_results(root):
    """Require a complete paired grid; uncertainty is across training-run means."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    root = Path(root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    manifest = verify_data(plan, root)
    task_ids = [r["task"]["task_id"] for r in manifest["records"]]
    config = plan["configs"]["ad_short"]
    steps = evaluation_steps(config["train_steps"], config["eval_interval"])
    curves, summary = {}, []
    for method in METHODS:
        for delay in plan["delays"]:
            per_seed = []
            for seed in plan["seeds"]:
                values = []
                for step in steps:
                    path = root / "evaluations" / f"{method}_s{seed}" / f"step-{step:07d}.json"
                    point = json.loads(path.read_text(encoding="utf-8"))
                    if (point["protocol"], point["method"], point["seed"], point["step"]) != (PROTOCOL, method, seed, step):
                        raise ValueError(f"Mismatched evaluation: {path}")
                    rows = [r for r in point["rows"] if r["delay"] == delay]
                    if [r["task_id"] for r in rows] != task_ids:
                        raise ValueError(f"Unpaired evaluation tasks: {path}")
                    regrets = np.asarray([r["cumulative_expected_regret"] for r in rows])
                    if not np.isfinite(regrets).all() or (regrets < 0).any():
                        raise ValueError(f"Invalid regret: {path}")
                    values.append(float(regrets.mean()))
                per_seed.append(values)
            runs = np.asarray(per_seed)
            mean = runs.mean(axis=0)
            std = runs.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(steps))
            curves[method, delay] = (mean, std)
            summary.extend({"method": method, "delay": delay, "step": step,
                "mean_cumulative_expected_regret": float(mean[i]), "training_seed_std": float(std[i]),
                "training_seeds": len(runs), "evaluation_tasks": len(task_ids), "genuine_pulls": 100}
                for i, step in enumerate(steps))
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    style = {"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        n = len(plan["delays"])
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.05), sharey=True, squeeze=False)
        for ax, delay in zip(axes[0], plan["delays"]):
            for method, color, linestyle in zip(METHODS, ("#0072B2", "#D55E00", "#009E73"), ("--", "-.", "-")):
                mean, std = curves[method, delay]
                ax.plot(steps, mean, label=LABELS[method], color=color, ls=linestyle, lw=1.7)
                if len(plan["seeds"]) > 1:
                    ax.fill_between(steps, np.maximum(0, mean - std), mean + std, color=color, alpha=0.13, linewidth=0)
            ax.set_title(f"Delay = {delay}")
            ax.set_xlabel("Training updates")
            ax.set_xlim(0, steps[-1])
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"))
            ax.grid(axis="y", alpha=0.2, lw=0.5)
        axes[0, 0].set_ylabel("Cumulative expected regret\n(100 genuine pulls)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3,
                   frameon=False, handlelength=2.3, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.8)
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"training_convergence.{suffix}", dpi=300)
        plt.close(fig)
    (output / "caption.txt").write_text(
        "AD-short, AD-long, and RAD training convergence. Each point is mean cumulative "
        "expected regret over 100 genuine pulls (50 before and 50 after the delay), "
        f"on {len(task_ids)} fixed held-out tasks. "
        + (f"Shading shows one sample standard deviation across {len(plan['seeds'])} training-seed means. "
           if len(plan["seeds"]) > 1 else "One training seed; no training-seed uncertainty is shown. ")
        + "Distractor transitions are excluded. RAD is trained from scratch; all methods use the same update budget.\n",
        encoding="utf-8")
    return output
