"""Matched online rollouts, shared-prefix diagnostics, and delay-sweep metrics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .env import BanditTask, make_manifest
from .rollout import generate_history
from .training import load_checkpoint
from .utils import SCHEMA, file_digest, get_config, project_path, write_json


class ModelPolicy:
    def __init__(self, model, seed=0, sample=True):
        self.model = model.eval()
        self.rng = np.random.default_rng(seed)
        self.sample = sample
        self.reset()

    def reset(self):
        self.state = self.model.new_state()

    @property
    def compression_count(self):
        return self.state["compression_count"]

    @property
    def recent_steps(self):
        recent = self.state["recent"]
        return 0 if recent is None else recent.shape[1] // 3

    @torch.inference_mode()
    def action(self, observation):
        observation = torch.tensor([observation], device=self.model.device)
        logits = self.model.query_logits(self.state, observation)[0].float()
        if not self.sample:
            return int(logits.argmax())
        probabilities = logits.softmax(-1).cpu().numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))

    @torch.inference_mode()
    def observe(self, observation, action, reward):
        tensors = [torch.tensor([[value]], device=self.model.device) for value in (observation, action, reward)]
        tokens = self.model.embed_transitions(*tensors)
        self.model.ingest(self.state, tokens)


class RandomPolicy:
    def __init__(self, num_arms, seed):
        self.num_arms, self.rng = num_arms, np.random.default_rng(seed)

    def reset(self):
        pass

    def action(self, observation):
        return int(self.rng.integers(self.num_arms))

    def observe(self, observation, action, reward):
        pass


def make_eval_manifest(seed, tasks, distributions, config):
    if list(distributions) != ["uniform"]:
        raise ValueError("Gaussian tasks use only independently uniform arm means")
    result = []
    for distribution in distributions:
        distribution_id = 0
        root = np.random.SeedSequence([seed, 71933, distribution_id])
        task_root, rollout_root = root.spawn(2)
        manifest = make_manifest(int(task_root.generate_state(1, dtype=np.uint64)[0]), tasks,
                                 distribution, config["num_arms"], f"eval-{distribution}",
                                 reward_std=config["reward_std"])
        for task, rollout_seed in zip(manifest, rollout_root.spawn(tasks)):
            seeds = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in rollout_seed.spawn(4)]
            result.append({"task": task.to_dict(), "reward_seed": seeds[0], "learner_seed": seeds[1],
                           "distractor_seed": seeds[2], "policy_seed": seeds[3]})
    return {"schema": SCHEMA, "seed": seed, "tasks_per_distribution": tasks,
            "distributions": distributions, "num_arms": config["num_arms"],
            "reward_std": config["reward_std"], "records": result}


def rollout_metrics(task, history, pre_steps):
    genuine = history["loss_mask"]
    rewards = history["rewards"][genuine].astype(float)
    actions = history["actions"][genuine]
    means = np.asarray(task.means)
    regret = means.max() - means[actions]
    post = rewards[pre_steps:]
    first_sequence_index = int(np.flatnonzero(genuine)[pre_steps])
    metrics = {"pre_return": float(rewards[:pre_steps].sum()), "post_return": float(post.sum()),
               "post_regret": float(regret[pre_steps:].sum()),
               "first_optimal": float(means[actions[pre_steps]] == means.max()),
               "first_regret": float(regret[pre_steps]),
               "first_post_compressions": int(history["compression_events"][first_sequence_index]),
               "first_post_recent_steps": int(history["recent_steps"][first_sequence_index]),
               "reward_curve": rewards.tolist(), "regret_curve": regret.tolist()}
    for count in (1, 5, 10):
        metrics[f"first_{count}_return"] = float(post[:count].sum())
    return metrics


METRICS = ("pre_return", "post_return", "post_regret", "first_optimal", "first_regret",
           "first_1_return", "first_5_return", "first_10_return",
           "first_post_compressions", "first_post_recent_steps")


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["distribution"], row["delay"])].append(row)
    result = []
    for (method, distribution, delay), group in sorted(grouped.items()):
        item = {"method": method, "distribution": distribution, "delay": delay,
                "rollouts": len(group), "training_runs": len({r["run_id"] for r in group})}
        for metric in METRICS:
            by_run = defaultdict(list)
            for row in group:
                by_run[row["run_id"]].append(row[metric])
            # Multiple checkpoints: uncertainty over run means. Single checkpoint:
            # uncertainty over tasks (does not estimate training-seed variability).
            values = np.asarray([np.mean(v) for v in by_run.values()] if len(by_run) > 1 else next(iter(by_run.values())))
            item[metric] = float(values.mean())
            item[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        item["ci_unit"] = "training_run" if item["training_runs"] > 1 else "task"
        item["reward_curve"] = np.mean([row["reward_curve"] for row in group], axis=0).tolist()
        item["regret_curve"] = np.mean([row["regret_curve"] for row in group], axis=0).tolist()
        result.append(item)
    # Optional normalization is reported alongside raw metrics, with undefined
    # denominators represented explicitly rather than divided by near-zero values.
    baseline = {(r["distribution"], r["delay"], r["method"]): r["post_return"] for r in result}
    for item in result:
        ucb = baseline.get((item["distribution"], item["delay"], "UCB"))
        random = baseline.get((item["distribution"], item["delay"], "Random"))
        item["normalized_post_score"] = ((item["post_return"] - random) / (ucb - random)
                                          if ucb is not None and random is not None and abs(ucb-random) > 1e-8 else None)
    return result


def plot_summary(summary, output, reference_context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distributions = sorted({row["distribution"] for row in summary})
    fig, axes = plt.subplots(2, len(distributions), figsize=(4 * len(distributions), 6), squeeze=False)
    for column, distribution in enumerate(distributions):
        for method in sorted({r["method"] for r in summary}):
            points = sorted((r for r in summary if r["method"] == method and r["distribution"] == distribution),
                            key=lambda r: r["delay"])
            for row_index, metric in enumerate(("post_return", "first_10_return")):
                axes[row_index, column].errorbar([p["delay"] / reference_context for p in points],
                                                [p[metric] for p in points],
                                                yerr=[p[f"{metric}_ci95"] for p in points],
                                                marker="o", label=method, capsize=2)
                axes[row_index, column].set(xlabel=f"Delay / {reference_context}", ylabel=metric.replace("_", " "))
                axes[row_index, column].grid(alpha=0.2)
        axes[0, column].set_title(distribution)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(output) / "delay_sweep.png", dpi=180)
    fig.savefig(Path(output) / "delay_sweep.pdf")
    plt.close(fig)


def evaluate_suite(checkpoints, output, config, *, tasks=100, seed=10000,
                   distributions=("uniform",), delays=None, labels=None,
                   device="cpu", sample=True, include_baselines=True, shared_prefix=False,
                   reference_context=50, manifest_path=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output must be empty: {output}")
    delays = config["eval_delays"] if delays is None else delays
    if not delays or min(delays) < 0 or reference_context <= 0:
        raise ValueError("Delays must be nonnegative and reference_context positive")
    if labels is not None and len(labels) != len(checkpoints):
        raise ValueError("Supply one label per checkpoint")
    if not checkpoints and not include_baselines:
        raise ValueError("No evaluation methods selected")
    if len(set(delays)) != len(delays):
        raise ValueError("Duplicate delays would repeat identical evaluation rollouts")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if (manifest["schema"] != SCHEMA or manifest["num_arms"] != config["num_arms"] or
                manifest.get("reward_std") != config["reward_std"]):
            raise ValueError("Incompatible evaluation manifest")
    else:
        manifest = make_eval_manifest(seed, tasks, list(distributions), config)
    if not manifest["records"]:
        raise ValueError("Evaluation manifest is empty")
    for record in manifest["records"]:
        task = BanditTask.from_dict(record["task"])
        if task.reward_std != config["reward_std"]:
            raise ValueError("Evaluation task noise does not match configuration")
    if len({r["task"]["task_id"] for r in manifest["records"]}) != len(manifest["records"]):
        raise ValueError("Evaluation task IDs must be unique")
    write_json(output / "manifest.json", manifest)
    methods = []
    provenance = []
    method_seeds = set()
    if include_baselines:
        methods.extend([("UCB", "source", None, None), ("Random", "random", None, None)])
    for index, checkpoint in enumerate(checkpoints):
        model, payload = load_checkpoint(checkpoint, device)
        if payload["phase"] != "distill" or model.num_arms != config["num_arms"]:
            raise ValueError("Evaluation requires a distilled policy with matching action space")
        model.requires_grad_(False)
        model_config = payload["config"]
        label = labels[index] if labels else f"{model_config['model']}-K{model_config['context_steps']}"
        if label in ("UCB", "Random") or not label:
            raise ValueError("Policy labels must be nonempty and different from UCB/Random")
        if (label, model_config["seed"]) in method_seeds:
            raise ValueError("Use one checkpoint per training seed per method; label ablations separately")
        method_seeds.add((label, model_config["seed"]))
        file = Path(checkpoint) / "model.pt" if Path(checkpoint).is_dir() else Path(checkpoint)
        run_id = file_digest(file)
        methods.append((label, run_id, model, model_config["seed"]))
        provenance.append({"method": label, "run_id": run_id, "checkpoint": str(file.resolve()),
                           "config": model_config, "step": payload["step"]})
    write_json(output / "evaluation.json", {"config": config, "delays": delays,
               "protocol": "shared_ucb_prefix" if shared_prefix else "online",
               "sample_actions": sample, "reference_context": reference_context,
               "checkpoints": provenance})
    rows = []
    for method, run_id, model, training_seed in methods:
        for delay in delays:
            for record in manifest["records"]:
                task = BanditTask.from_dict(record["task"])
                policy = (ModelPolicy(model, record["policy_seed"], sample) if model is not None else
                          RandomPolicy(config["num_arms"], record["policy_seed"]) if method == "Random" else None)
                kwargs = {key: record[key] for key in ("reward_seed", "learner_seed", "distractor_seed")}
                kwargs.update(delay=delay, pre_steps=config["pre_steps"], post_steps=config["post_steps"],
                              exploration_coefficient=config["exploration_coefficient"])
                prefix = generate_history(task, **kwargs)[0] if shared_prefix and policy is not None else None
                history, _ = generate_history(task, policy=policy, prefix=prefix, **kwargs)
                row = {"method": method, "run_id": run_id, "training_seed": training_seed,
                       "task_id": task.task_id, "distribution": task.distribution, "delay": delay,
                       **rollout_metrics(task, history, config["pre_steps"])}
                rows.append(row)
                with (output / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            print(f"Evaluated {method}: delay={delay}, tasks={len(manifest['records'])}", flush=True)
    summary = summarize(rows)
    write_json(output / "summary.json", summary)
    columns = [key for key in summary[0] if not key.endswith("_curve")]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    plot_summary(summary, output, reference_context)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="*", default=[])
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--env", default="delayed_adversarial_bandit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--delays", type=int, nargs="+")
    parser.add_argument("--distributions", nargs="+", choices=("uniform",), default=["uniform"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument("--shared_prefix", action="store_true")
    parser.add_argument("--reference_context", type=int, default=50)
    parser.add_argument("--manifest")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = get_config(f"config/env/{args.env}.yaml")
    config.update(get_config("config/algorithm/ucb.yaml"))
    evaluate_suite([project_path(p) for p in args.checkpoint], project_path(args.output), config,
                   tasks=args.tasks, seed=args.seed, delays=args.delays, labels=args.labels,
                   distributions=args.distributions, device=args.device, sample=not args.greedy,
                   include_baselines=not args.no_baselines, shared_prefix=args.shared_prefix,
                   reference_context=args.reference_context,
                   manifest_path=project_path(args.manifest) if args.manifest else None)


if __name__ == "__main__":
    main()
