"""Matched online rollouts, shared-prefix diagnostics, and delay-sweep metrics."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

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
                "rollouts": len(group), "training_runs": len({r["run_id"] for r in group}),
                "eval_seeds": len({r["eval_seed"] for r in group})}
        run_ids = {r["run_id"] for r in group}
        seeds = {r["eval_seed"] for r in group}
        # Uncertainty hierarchy: training-run means when several checkpoints are
        # present, else evaluation-seed means, else task variability for a single
        # run and seed (the unit is labeled explicitly below).
        if len(run_ids) > 1:
            bucketed = defaultdict(list)
            for row in group:
                bucketed[row["run_id"]].append(row)
            ci_unit = "training_run"
        elif len(seeds) > 1:
            bucketed = defaultdict(list)
            for row in group:
                bucketed[row["eval_seed"]].append(row)
            ci_unit = "eval_seed"
        else:
            bucketed = {None: list(group)}
            ci_unit = "task"
        replications = list(bucketed.values())
        for metric in METRICS:
            values = np.asarray([np.mean([row[metric] for row in rep]) for rep in replications])
            item[metric] = float(values.mean())
            item[f"{metric}_ci95"] = (float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                                      if len(values) > 1 else 0.0)
        item["ci_unit"] = ci_unit
        item["reward_curve"] = np.mean([row["reward_curve"] for row in group], axis=0).tolist()
        replication_curves = np.asarray([np.mean([row["regret_curve"] for row in rep], axis=0)
                                         for rep in replications])
        item["regret_curve"] = replication_curves.mean(axis=0).tolist()
        item["regret_curve_std"] = (replication_curves.std(axis=0, ddof=1) if len(replications) > 1
                                    else np.zeros(replication_curves.shape[1])).tolist()
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


METHOD_STYLES = {
    "ucb": {"color": "#999999", "linestyle": "--", "marker": ""},
    "random": {"color": "#666666", "linestyle": ":", "marker": ""},
    "adlong": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "adshort": {"color": "#E69F00", "linestyle": "-", "marker": "^"},
    "rad": {"color": "#009E73", "linestyle": "-", "marker": "s"},
}
STYLE_FALLBACK_COLORS = ("#D55E00", "#CC79A7", "#56B4E9", "#F0E442")
STYLE_FALLBACK_MARKERS = ("D", "v", "P", "X")

PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def method_styles(methods):
    """Colorblind-safe styles; baselines are muted, and line/marker shapes keep
    methods distinguishable in grayscale and for unknown label spellings."""
    styles = {}
    fallback = 0
    for method in methods:
        key = method.lower().replace("-", "").replace("_", "")
        if key in METHOD_STYLES:
            styles[method] = dict(METHOD_STYLES[key])
        else:
            styles[method] = {"color": STYLE_FALLBACK_COLORS[fallback % len(STYLE_FALLBACK_COLORS)],
                              "linestyle": "-",
                              "marker": STYLE_FALLBACK_MARKERS[fallback % len(STYLE_FALLBACK_MARKERS)]}
            fallback += 1
    return styles


def method_plot_order(methods):
    """Draw baselines first so distilled-policy lines stay on top."""
    return sorted(methods, key=lambda m: (m.lower() in ("ucb", "random"), m.lower()))


def plot_summary(summary, output, reference_context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distributions = sorted({row["distribution"] for row in summary})
    methods = sorted({r["method"] for r in summary})
    styles = method_styles(methods)
    order = method_plot_order(methods)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, len(distributions), figsize=(3.5 * len(distributions), 5.0),
                                 squeeze=False, layout="constrained")
        handles = {}
        for column, distribution in enumerate(distributions):
            for row_index, metric in enumerate(("post_return", "first_10_return")):
                axis = axes[row_index, column]
                series = []
                for method in order:
                    points = sorted((r for r in summary if r["method"] == method
                                     and r["distribution"] == distribution), key=lambda r: r["delay"])
                    if not points:
                        continue
                    x = np.asarray([p["delay"] / reference_context for p in points])
                    y = np.asarray([p[metric] for p in points])
                    ci = np.asarray([p[f"{metric}_ci95"] for p in points])
                    series.append((method, x, y, ci))
                for method, x, y, ci in series:
                    if ci.any():
                        axis.fill_between(x, y - ci, y + ci, color=styles[method]["color"],
                                          alpha=0.15, linewidth=0)
                for method, x, y, ci in series:
                    (line,) = axis.plot(x, y, **styles[method], markersize=4)
                    handles[method] = line
                axis.set(xlabel=f"Delay / {reference_context}", ylabel=metric.replace("_", " "))
                axis.grid(True)
                axis.margins(x=0.08, y=0.18)
            axes[0, column].set_title(distribution)
        legend_methods = [m for m in order if m in handles]
        fig.legend([handles[m] for m in legend_methods], legend_methods,
                   loc="outside lower center", ncol=max(2, int(np.ceil(len(legend_methods) / 2))),
                   frameon=False)
        fig.savefig(Path(output) / "delay_sweep.png", dpi=300)
        fig.savefig(Path(output) / "delay_sweep.pdf")
    plt.close(fig)


def plot_cumulative_regret(summary, output, pre_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    distributions = sorted({row["distribution"] for row in summary})
    delays = sorted({row["delay"] for row in summary})
    methods = sorted({row["method"] for row in summary})
    styles = method_styles(methods)
    order = method_plot_order(methods)
    total_pulls = max(len(row["regret_curve"]) for row in summary)
    panels = {}
    ymax = 0.0
    for distribution in distributions:
        for delay in delays:
            curves = {}
            for method in order:
                row = next((r for r in summary if r["method"] == method
                            and r["distribution"] == distribution and r["delay"] == delay), None)
                if row is None:
                    continue
                cumulative = np.concatenate(([0.0], np.cumsum(row["regret_curve"])))
                curves[method] = cumulative
                ymax = max(ymax, float(cumulative.max()))
            if curves:
                panels[(distribution, delay)] = curves
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(distributions), len(delays),
                                 figsize=(1.9 * len(delays) + 0.9, 1.9 * len(distributions) + 0.6),
                                 squeeze=False, layout="constrained")
        handles = {}
        for column, delay in enumerate(delays):
            for row_index, distribution in enumerate(distributions):
                axis = axes[row_index, column]
                curves = panels.get((distribution, delay), {})
                for method in order:
                    if method not in curves:
                        continue
                    cumulative = curves[method]
                    (line,) = axis.plot(np.arange(len(cumulative)), cumulative,
                                        **{**styles[method], "marker": ""})
                    handles[method] = line
                axis.axvline(pre_steps, color="black", linestyle="--", linewidth=0.8)
                axis.set(xlabel="Arm pulls", title=f"delay {delay}", xlim=(0, total_pulls),
                         ylim=(0.0, ymax * 1.05))
                if column == 0:
                    axis.set(ylabel="Cumulative expected regret")
                axis.grid(True)
                label_transform = blended_transform_factory(axis.transData, axis.transAxes)
                axis.text(pre_steps / 2, 0.96, "before delay", transform=label_transform,
                          ha="center", va="top", fontsize=7)
                axis.text((pre_steps + total_pulls) / 2, 0.96, "after delay", transform=label_transform,
                          ha="center", va="top", fontsize=7)
        legend_entries = [m for m in order if m in handles]
        fig.legend([handles[m] for m in legend_entries], legend_entries,
                   loc="outside lower center", ncol=len(legend_entries), frameon=False)
        fig.savefig(Path(output) / "cumulative_regret.png", dpi=300)
        fig.savefig(Path(output) / "cumulative_regret.pdf")
    plt.close(fig)


RUN_FAMILY_PATTERN = re.compile(r"^(?P<family>.+)_s(?P<seed>\d+)$")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(?P<step>\d+)$")


def discover_run_checkpoints(runs_dir):
    """Group training runs by family and select each run's max-iteration checkpoint.

    Run directories named ``<family>_s<seed>`` belong to one method; giving every
    run in a family the same label makes the summary average over training runs.
    Runs without a completed distilled checkpoint (e.g. pretraining only) are
    returned in ``skipped`` and excluded from ``discovered``.
    """
    discovered = defaultdict(list)
    skipped = []
    for run_dir in sorted(path for path in Path(runs_dir).iterdir() if path.is_dir()):
        candidates = []
        for child in run_dir.iterdir():
            match = CHECKPOINT_PATTERN.match(child.name)
            if match and child.is_dir() and (child / "model.pt").exists():
                candidates.append((int(match.group("step")), child))
        if not candidates:
            continue
        step, checkpoint_dir = max(candidates)
        training = json.loads((checkpoint_dir / "training.json").read_text(encoding="utf-8"))
        if training.get("phase") != "distill":
            skipped.append(run_dir.name)
            continue
        match = RUN_FAMILY_PATTERN.match(run_dir.name)
        family = match.group("family") if match else run_dir.name
        seed = int(match.group("seed")) if match else None
        discovered[family].append({"run": run_dir.name, "seed": seed, "step": step,
                                   "checkpoint": checkpoint_dir})
    for runs in discovered.values():
        runs.sort(key=lambda item: (item["seed"] is None, item["seed"] if item["seed"] is not None else 0,
                                    item["run"]))
    return dict(discovered), skipped


def _cache_protocol(config, manifest, delay, sample, shared_prefix):
    """Everything that determines rollout outcomes for one cache block.

    Plotting-only settings (e.g. reference_context) stay out so restyling
    figures does not invalidate cached evaluations.
    """
    return {"schema": SCHEMA, "num_arms": manifest["num_arms"], "reward_std": manifest["reward_std"],
            "tasks": manifest["tasks_per_distribution"], "distributions": list(manifest["distributions"]),
            "manifest_seed": manifest["seed"], "delay": delay,
            "pre_steps": config["pre_steps"], "post_steps": config["post_steps"],
            "exploration_coefficient": config["exploration_coefficient"],
            "sample": sample, "shared_prefix": shared_prefix}


def _cache_file_name(method, run_id, eval_seed, delay, protocol):
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    safe_label = re.sub(r"[^0-9A-Za-z_.-]+", "_", method)
    key = run_id[:12] if run_id else "baseline"
    return f"{safe_label}__{key}__seed{eval_seed}__delay{delay}__{digest}.json"


def evaluate_suite(checkpoints, output, config, *, tasks=100, seed=10000, eval_seeds=1,
                   distributions=("uniform",), delays=None, labels=None,
                   device="cpu", sample=True, include_baselines=True, shared_prefix=False,
                   reference_context=50, manifest_path=None, cache_dir=None, force=False):
    output = Path(output)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if output.exists() and any(output.iterdir()):
        if cache_dir is None or not (output / "evaluation.json").exists():
            raise FileExistsError(f"Evaluation output must be empty: {output}")
        reuse_output = True
    else:
        reuse_output = False
    delays = config["eval_delays"] if delays is None else delays
    if not delays or min(delays) < 0 or reference_context <= 0:
        raise ValueError("Delays must be nonnegative and reference_context positive")
    if eval_seeds < 1:
        raise ValueError("eval_seeds must be positive")
    if labels is not None and len(labels) != len(checkpoints):
        raise ValueError("Supply one label per checkpoint")
    if not checkpoints and not include_baselines:
        raise ValueError("No evaluation methods selected")
    if len(set(delays)) != len(delays):
        raise ValueError("Duplicate delays would repeat identical evaluation rollouts")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path:
        # An external manifest pins a single evaluation (its own seed); seed
        # averaging requires generated manifests, so it is disabled on this path.
        eval_seeds = 1
        manifests = [json.loads(Path(manifest_path).read_text(encoding="utf-8"))]
    else:
        manifests = [make_eval_manifest(seed + offset, tasks, list(distributions), config)
                     for offset in range(eval_seeds)]
    for manifest in manifests:
        if (manifest["schema"] != SCHEMA or manifest["num_arms"] != config["num_arms"] or
                manifest.get("reward_std") != config["reward_std"]):
            raise ValueError("Incompatible evaluation manifest")
        if not manifest["records"]:
            raise ValueError("Evaluation manifest is empty")
        for record in manifest["records"]:
            task = BanditTask.from_dict(record["task"])
            if task.reward_std != config["reward_std"]:
                raise ValueError("Evaluation task noise does not match configuration")
        if len({r["task"]["task_id"] for r in manifest["records"]}) != len(manifest["records"]):
            raise ValueError("Evaluation task IDs must be unique")
    write_json(output / "manifest.json", manifests[0])
    if len(manifests) > 1:
        write_json(output / "manifests.json", {"schema": SCHEMA,
                                               "eval_seeds": [m["seed"] for m in manifests],
                                               "manifests": manifests})
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
    evaluation = {"config": config, "delays": delays,
                  "protocol": "shared_ucb_prefix" if shared_prefix else "online",
                  "sample_actions": sample, "reference_context": reference_context,
                  "eval_seeds": [m["seed"] for m in manifests],
                  "checkpoints": provenance}
    if reuse_output:
        previous = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        # reference_context only rescales the delay-sweep axes; rollouts and
        # summaries are unaffected, so replotting may reuse the output.
        plotting_keys = ("reference_context",)
        compatible = {key: value for key, value in evaluation.items() if key not in plotting_keys}
        previous_compatible = {key: value for key, value in previous.items() if key not in plotting_keys}
        if previous_compatible != compatible:
            raise FileExistsError(f"Evaluation output {output} holds a different evaluation;"
                                  " choose a new --output")
        (output / "rollouts.jsonl").unlink(missing_ok=True)
        (output / "manifests.json").unlink(missing_ok=True)
    write_json(output / "evaluation.json", evaluation)
    rows = []
    for manifest in manifests:
        eval_seed = manifest["seed"]
        for method, run_id, model, training_seed in methods:
            for delay in delays:
                cache_path = None
                if cache_dir is not None:
                    protocol = _cache_protocol(config, manifest, delay, sample, shared_prefix)
                    cache_path = cache_dir / _cache_file_name(method, run_id, eval_seed, delay, protocol)
                    if cache_path.exists() and not force:
                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                        if cached["protocol"] != protocol:
                            raise ValueError(f"Cache entry does not match its protocol: {cache_path}")
                        rows.extend(cached["rows"])
                        with (output / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                            for row in cached["rows"]:
                                handle.write(json.dumps(row) + "\n")
                        print(f"Loaded {method}: eval_seed={eval_seed}, delay={delay} from cache",
                              flush=True)
                        continue
                block_rows = []
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
                           "eval_seed": eval_seed, "task_id": task.task_id,
                           "distribution": task.distribution, "delay": delay,
                           **rollout_metrics(task, history, config["pre_steps"])}
                    rows.append(row)
                    block_rows.append(row)
                    with (output / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(cache_path, {"protocol": protocol, "rows": block_rows})
                seed_note = f" eval_seed={eval_seed}," if len(manifests) > 1 else ""
                print(f"Evaluated {method}:{seed_note} delay={delay}, tasks={len(manifest['records'])}",
                      flush=True)
    summary = summarize(rows)
    write_json(output / "summary.json", summary)
    columns = [key for key in summary[0] if not key.endswith(("_curve", "_curve_std"))]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    plot_summary(summary, output, reference_context)
    plot_cumulative_regret(summary, output, config["pre_steps"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="*", default=[])
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--env", default="delayed_adversarial_bandit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--eval_seeds", "--eval-seeds", type=int, default=5,
                        help="Independent evaluation seed runs to average")
    parser.add_argument("--delays", type=int, nargs="+")
    parser.add_argument("--distributions", nargs="+", choices=("uniform",), default=["uniform"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument("--shared_prefix", action="store_true")
    parser.add_argument("--reference_context", type=int, default=50)
    parser.add_argument("--manifest")
    parser.add_argument("--cache_dir", "--cache-dir", default=None,
                        help="Directory caching evaluation rollouts per method, eval seed, and delay")
    parser.add_argument("--force", action="store_true", help="Recompute cached evaluations")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.eval_seeds < 1:
        parser.error("eval_seeds must be positive")
    torch.set_num_threads(args.threads)
    config = get_config(f"config/env/{args.env}.yaml")
    config.update(get_config("config/algorithm/ucb.yaml"))
    evaluate_suite([project_path(p) for p in args.checkpoint], project_path(args.output), config,
                   tasks=args.tasks, seed=args.seed, eval_seeds=args.eval_seeds,
                   delays=args.delays, labels=args.labels,
                   distributions=args.distributions, device=args.device, sample=not args.greedy,
                   include_baselines=not args.no_baselines, shared_prefix=args.shared_prefix,
                   reference_context=args.reference_context,
                   manifest_path=project_path(args.manifest) if args.manifest else None,
                   cache_dir=project_path(args.cache_dir) if args.cache_dir else None,
                   force=args.force)


if __name__ == "__main__":
    main()
