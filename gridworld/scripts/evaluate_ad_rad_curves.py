"""
Evaluate RAD, AD, DPT, IDT and plot them alongside source RL learning curves.

Examples:
    uv run python scripts/evaluate_ad_rad_curves.py --dry-run
    uv run python scripts/evaluate_ad_rad_curves.py --eval-seeds 0 1 2 --eval-episodes 10
    uv run python scripts/evaluate_ad_rad_curves.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from env import SAMPLE_ENVIRONMENT, make_env
from baseline_dataset import selected_group_ids
from model import MODEL
from utils import get_traj_file_name, normalize_compiled_state_dict


ENV_LABELS = {
    "darkroom": "Darkroom",
    "dktd": "Dark Key-to-Door",
}

METHOD_LABELS = {
    "AD": "AD",
    "RAD": "RAD",
    "DPT": "DPT",
    "IDT": "IDT",
    "SOURCE": "Source RL (training)",
}

METHOD_COLORS = {
    "AD": "#0072B2",
    "RAD": "#D55E00",
    "DPT": "#009E73",
    "IDT": "#CC79A7",
    "SOURCE": "#666666",
}



@dataclass(frozen=True)
class CheckpointSpec:
    env: str
    method: str
    train_seed: int
    run_dir: Path
    ckpt_path: Path
    ckpt_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AD/DPT/IDT checkpoints and RAD best checkpoints, then plot "
            "episode rewards alongside source RL training histories."
        )
    )
    parser.add_argument("--runs-root", default="./runs", help="Run directory root, relative to gridworld_test.")
    parser.add_argument("--output-dir", default="./runs/eval_curves", help="Output directory.")
    parser.add_argument("--envs", nargs="+", default=["darkroom", "dktd"], choices=["darkroom", "dktd"])
    parser.add_argument("--methods", nargs="+", default=["RAD", "AD", "DPT", "IDT"], choices=list(METHOD_LABELS))
    parser.add_argument("--source-history", action="append", default=[], metavar="ENV=PATH",
                        help="Override source HDF5 path per environment; otherwise use checkpoint config and --datasets-root")
    parser.add_argument("--datasets-root", default="./datasets")
    parser.add_argument("--collection-env-split-seed", type=int, help="Seed used by collect.py to number source tasks")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=None, help="Train seeds to include.")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=None, help="Eval seeds. Defaults to 0..19.")
    parser.add_argument("--num-eval-seeds", type=int, default=20, help="Used when --eval-seeds is omitted.")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-timesteps", type=int, default=None, help="Overrides --eval-episodes.")
    parser.add_argument("--window", type=int, default=5, help="Centered moving-average smoothing window.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Figure formats to save.")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--force", action="store_true", help="Recompute cached evaluations.")
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered checkpoint/eval work.")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax actions instead of the evaluators' default stochastic sampling.",
    )
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    resolved = Path(raw_path)
    if not resolved.is_absolute():
        resolved = PROJECT_DIR / resolved
    return resolved


def train_seed_from_run_name(run_name: str) -> int | None:
    match = re.search(r"-seed(\d+)(?:$|-)", run_name)
    return int(match.group(1)) if match else None


def step_from_checkpoint_name(path: Path) -> int | None:
    match = re.search(r"ckpt-(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(
    runs_root: Path,
    envs: Iterable[str],
    methods: Iterable[str],
    train_seeds: set[int] | None,
) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    for env in envs:
        for method in methods:
            for run_dir in sorted(runs_root.glob(f"{method}-{env}-seed*")):
                seed = train_seed_from_run_name(run_dir.name)
                if seed is None or (train_seeds is not None and seed not in train_seeds):
                    continue

                if method == "RAD":
                    ckpt_path = run_dir / "best-model.pt"
                    if ckpt_path.exists():
                        specs.append(
                            CheckpointSpec(
                                env=env,
                                method=method,
                                train_seed=seed,
                                run_dir=run_dir,
                                ckpt_path=ckpt_path,
                                ckpt_label="best",
                            )
                        )
                    continue

                for ckpt_path in sorted(run_dir.glob("ckpt-*.pt"), key=lambda p: step_from_checkpoint_name(p) or -1):
                    specs.append(
                        CheckpointSpec(
                            env=env,
                            method=method,
                            train_seed=seed,
                            run_dir=run_dir,
                            ckpt_path=ckpt_path,
                            ckpt_label=ckpt_path.stem,
                        )
                    )
    return specs


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_eval_envs(config: dict) -> DummyVecEnv:
    env_name = config["env"]
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    if env_name == "darkroom":
        env_fns = [make_env(config, goal=arg) for arg in test_env_args]
    elif env_name == "dktd":
        env_fns = [make_env(config, key=arg[:2], goal=arg[2:]) for arg in test_env_args]
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return DummyVecEnv(env_fns)


def load_model(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    config["device"] = device
    model = MODEL[config["model"]](config).to(device)
    strict = config["model"] != "RAD" or bool(config.get('compressor_type'))
    load_result = model.load_state_dict(
        normalize_compiled_state_dict(checkpoint["model"]),
        strict=strict,
    )
    if not strict:
        if load_result.missing_keys:
            print(f"Missing RAD keys initialized from config: {load_result.missing_keys}", flush=True)
        if load_result.unexpected_keys:
            print(f"Unexpected RAD keys ignored: {load_result.unexpected_keys}", flush=True)
    model.eval()
    return model, config, checkpoint


def cache_path_for(output_dir: Path, spec: CheckpointSpec, eval_seed: int, protocol: str = "") -> Path:
    identity = hashlib.sha256(f'{spec.ckpt_path.resolve()}|{protocol}'.encode()).hexdigest()[:12]
    return (
        output_dir
        / "cache"
        / spec.env
        / spec.method
        / f"train_seed{spec.train_seed}"
        / f"{spec.ckpt_label}_{identity}_eval_seed{eval_seed}.npz"
    )


def evaluate_checkpoint(
    spec: CheckpointSpec,
    eval_seeds: list[int],
    eval_episodes: int,
    eval_timesteps_override: int | None,
    device: torch.device,
    output_dir: Path,
    force: bool,
    sample: bool,
) -> list[dict]:
    model = None
    config = None
    checkpoint = None
    rows = []

    stat = spec.ckpt_path.stat()
    protocol = json.dumps([2, stat.st_mtime_ns, stat.st_size, eval_episodes, eval_timesteps_override, sample])

    for eval_seed in eval_seeds:
        result_path = cache_path_for(output_dir, spec, eval_seed, protocol)
        if result_path.exists() and not force:
            cached = np.load(result_path, allow_pickle=False)
            rows.append(
                {
                    "env": spec.env,
                    "method": spec.method,
                    "checkpoint": spec.ckpt_label,
                    "train_seed": spec.train_seed,
                    "eval_seed": eval_seed,
                    "step": int(cached["step"]),
                    "reward_mean": float(cached["reward_mean"]),
                    "reward_std_over_env_episodes": float(cached["reward_std_over_env_episodes"]),
                    "num_test_envs": int(cached["num_test_envs"]),
                    "num_eval_episodes": int(cached["num_eval_episodes"]),
                    "ckpt_path": str(spec.ckpt_path),
                    "cache_path": str(result_path),
                }
            )
            continue

        if model is None or config is None or checkpoint is None:
            model, config, checkpoint = load_model(spec.ckpt_path, device)

        set_eval_seed(eval_seed)
        eval_timesteps = eval_timesteps_override
        if eval_timesteps is None:
            eval_timesteps = int(config["horizon"]) * eval_episodes

        envs = build_eval_envs(config)
        try:
            with torch.inference_mode():
                rewards = model.evaluate_in_context(
                    vec_env=envs,
                    eval_timesteps=eval_timesteps,
                    sample=sample,
                )["reward_episode"]
        finally:
            envs.close()

        step = int(checkpoint.get("step") or step_from_checkpoint_name(spec.ckpt_path) or -1)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            result_path,
            reward_episode=rewards,
            reward_mean=float(rewards.mean()),
            reward_std_over_env_episodes=float(rewards.std()),
            step=step,
            train_seed=spec.train_seed,
            eval_seed=eval_seed,
            num_test_envs=rewards.shape[0] if rewards.ndim >= 1 else 0,
            num_eval_episodes=rewards.shape[1] if rewards.ndim >= 2 else 0,
        )

        rows.append(
            {
                "env": spec.env,
                "method": spec.method,
                "checkpoint": spec.ckpt_label,
                "train_seed": spec.train_seed,
                "eval_seed": eval_seed,
                "step": step,
                "reward_mean": float(rewards.mean()),
                "reward_std_over_env_episodes": float(rewards.std()),
                "num_test_envs": int(rewards.shape[0]) if rewards.ndim >= 1 else 0,
                "num_eval_episodes": int(rewards.shape[1]) if rewards.ndim >= 2 else 0,
                "ckpt_path": str(spec.ckpt_path),
                "cache_path": str(result_path),
            }
        )

    return rows


def source_history_rows(specs, overrides, datasets_root, output_dir, eval_episodes,
                        eval_timesteps=None, collection_seed=None):
    """Read true PPO training rewards for the tasks used by build_eval_envs.

    collect.py stores sequential group IDs in shuffled task order. Invert that
    order to find evaluation tasks; a group ID is not a coordinate encoding.
    Each stream is one source learner trajectory, not a new evaluation seed.
    """
    rows, seen = [], set()
    for spec in specs:
        checkpoint = torch.load(spec.ckpt_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        path = overrides.get(spec.env, datasets_root / f'{get_traj_file_name(config)}.hdf5')
        split_seed = config['env_split_seed']
        source_seed = collection_seed if collection_seed is not None else config.get('collection_env_split_seed', split_seed)
        horizon = int(config['horizon'])
        episode_limit = eval_episodes if eval_timesteps is None else eval_timesteps // horizon
        if episode_limit < 1:
            raise ValueError('Evaluation must cover at least one complete episode')
        groups = selected_group_ids({**config, 'collection_env_split_seed': source_seed}, 'test')
        identity = (str(path.resolve()), tuple(groups), horizon, episode_limit, config['alg'])
        if identity in seen:
            continue
        seen.add(identity)
        if not path.exists():
            raise FileNotFoundError(f'Source history missing: {path}. Set --source-history {spec.env}=PATH or omit SOURCE from --methods.')
        trials = []
        with h5py.File(path, 'r') as history:
            for group_id in groups:
                if str(group_id) not in history:
                    raise ValueError(f'{path}: missing held-out task group {group_id}; collect all evaluation tasks')
                group = history[str(group_id)]
                reward = np.asarray(group['rewards'][:episode_limit * horizon], dtype=float)
                count = reward.shape[0] // horizon
                if count < 1:
                    raise ValueError(f'{path}: group {group_id} has no complete episodes')
                if 'dones' in group:
                    done = group['dones'][:count * horizon].astype(bool)
                    expected = np.broadcast_to((np.arange(count * horizon) % horizon == horizon - 1)[:, None], done.shape)
                    if not np.array_equal(done, expected):
                        raise ValueError(f'{path}: episode boundaries disagree with checkpoint horizon={horizon}')
                trials.append(reward[:count * horizon].reshape(count, horizon, -1).sum(1).T)
        count = min(trial.shape[1] for trial in trials)
        rewards = np.concatenate([trial[:, :count] for trial in trials], 0)
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:12]
        cache_path = output_dir / 'cache' / spec.env / 'SOURCE' / f'{digest}.npz'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, reward_episode=rewards, group_ids=groups, history_path=str(path))
        rows.append(dict(env=spec.env, method='SOURCE', checkpoint='source-history', train_seed=split_seed,
                         eval_seed=int(config.get('alg_seed', 0)), step=0, reward_mean=float(rewards.mean()),
                         reward_std_over_env_episodes=float(rewards.std()), num_test_envs=len(groups),
                         num_eval_episodes=count, ckpt_path=str(path), cache_path=str(cache_path),
                         source_algorithm=config['alg']))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregation_checkpoint(row: dict) -> str:
    if row["method"] == "RAD":
        return "best"
    return str(row["step"])


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["env"], row["method"], aggregation_checkpoint(row))].append(row)

    summary = []
    for (env, method, checkpoint), group in sorted(grouped.items()):
        values = np.array([row["reward_mean"] for row in group], dtype=np.float64)
        steps = np.array([row["step"] for row in group], dtype=np.float64)
        train_seeds = sorted({int(row["train_seed"]) for row in group})
        eval_seeds = sorted({int(row["eval_seed"]) for row in group})
        summary.append(
            {
                "env": env,
                "method": method,
                "checkpoint": checkpoint,
                "step_mean": float(steps.mean()),
                "step_std": float(steps.std(ddof=0)),
                "step_min": int(steps.min()),
                "step_max": int(steps.max()),
                "reward_mean": float(values.mean()),
                "reward_std": float(values.std(ddof=0)),
                "reward_sem": float(values.std(ddof=0) / np.sqrt(values.size)),
                "num_points": int(values.size),
                "num_train_seeds": len(train_seeds),
                "num_eval_seeds": len(eval_seeds),
                "train_seeds": " ".join(map(str, train_seeds)),
                "eval_seeds": " ".join(map(str, eval_seeds)),
            }
        )
    return summary


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 4.0),
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )



def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    # Centered moving average that preserves the original series length.
    if window <= 1:
        return values.astype(float, copy=True)

    window = min(window, values.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values.astype(float), (left, right), mode='edge')
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode='valid')


def load_plot_rewards(rows: list[dict], env: str, method: str) -> np.ndarray | None:
    trials = []
    episode_count = None
    relevant = [row for row in rows if row['env'] == env and row['method'] == method]
    # Different training checkpoints are not independent evaluation trials.
    # Retain all in CSV, but draw the latest checkpoint from each training run.
    latest = {}
    for row in relevant:
        run = str(Path(row['ckpt_path']).parent)
        latest[run] = max(latest.get(run, -1), row['step'])
    for row in relevant:
        if row['step'] != latest[str(Path(row['ckpt_path']).parent)]:
            continue
        if row['env'] != env or row['method'] != method:
            continue
        cache_path = Path(row['cache_path'])
        cached = np.load(cache_path, allow_pickle=False)
        rewards = np.asarray(cached['reward_episode'], dtype=float)
        if rewards.ndim == 1:
            rewards = rewards[None, :]
        elif rewards.ndim != 2:
            raise ValueError(f'{cache_path} reward_episode must be 1D or 2D, got {rewards.shape}.')

        if rewards.shape[1] == 0:
            raise ValueError(f'{cache_path}: no completed evaluation episodes')
        if episode_count is None:
            episode_count = rewards.shape[1]
        elif rewards.shape[1] != episode_count:
            if method == 'SOURCE':
                episode_count = min(episode_count, rewards.shape[1])
                trials = [trial[:, :episode_count] for trial in trials]
                rewards = rewards[:, :episode_count]
                trials.append(rewards)
                continue
            raise ValueError(
                f'Episode count mismatch for {method} on {env}: {cache_path} has '
                f'{rewards.shape[1]} episodes, expected {episode_count}.'
            )
        trials.append(rewards)

    if not trials:
        return None
    return np.concatenate(trials, axis=0)


def summarize_rewards(rewards: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    smooth_mean = moving_average(mean, window)
    smooth_std = moving_average(std, window)
    episodes = np.arange(1, mean.size + 1)

    return episodes, smooth_mean, smooth_std


def plot_environment(
    env: str | list[str],
    detail_rows: list[dict],
    output_dir: Path,
    window: int,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    configure_plot_style()

    envs = [env] if isinstance(env, str) else list(dict.fromkeys(env))
    envs = [name for name in envs if any(row['env'] == name for row in detail_rows)]
    if not envs:
        return []

    fig, axes = plt.subplots(1, len(envs), figsize=(5.0 * len(envs), 4.6), squeeze=False)
    saved_paths = []
    legend_handles = {}

    for ax, env in zip(axes[0], envs):
        max_upper = 0.0
        max_episodes = 1
        for method in ('RAD', 'AD', 'DPT', 'IDT', 'SOURCE'):
            rewards = load_plot_rewards(detail_rows, env, method)
            if rewards is None:
                continue

            episodes, mean, std = summarize_rewards(rewards, window)
            max_episodes = max(max_episodes, int(episodes[-1]))
            color = METHOD_COLORS[method]
            label = METHOD_LABELS[method]
            if method == 'SOURCE':
                algorithms = sorted({row.get('source_algorithm', 'RL') for row in detail_rows if row['env'] in envs and row['method'] == method})
                label = f"Source {'/'.join(algorithms)} (training)"
            lower = np.maximum(mean - std, 0.0)
            upper = mean + std
            max_upper = max(max_upper, float(np.nanmax(upper)))

            line, = ax.plot(episodes, mean, color=color, label=label, linestyle='--' if method == 'SOURCE' else '-')
            ax.fill_between(episodes, lower, upper, color=color, alpha=0.1, linewidth=0.0)
            legend_handles.setdefault(method, line)

        env_label = ENV_LABELS[env]
        ax.set_title(f'{env_label}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average episode reward')
        ax.set_xlim(1, max(2, max_episodes))
        ax.set_ylim(0, max_upper * 1.08 if max_upper > 0 else 1)
        ax.margins(x=0.01)

    if not legend_handles:
        plt.close(fig)
        return saved_paths

    handles = [legend_handles[method] for method in ('RAD', 'AD', 'DPT', 'IDT', 'SOURCE') if method in legend_handles]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ad_rad_{'_'.join(envs)}_eval_rewards"
    for fmt in formats:
        suffix = fmt.lstrip('.')
        output_path = output_dir / f'{stem}.{suffix}'
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        saved_paths.append(output_path)

    plt.close(fig)
    return saved_paths


def print_discovery(specs: list[CheckpointSpec], eval_seeds: list[int]) -> None:
    print(f"Discovered {len(specs)} checkpoints/directories to evaluate.")
    print(f"Eval seeds: {eval_seeds}")
    for spec in specs:
        print(
            f"  {spec.env:8s} {spec.method:3s} train_seed={spec.train_seed:<3d} "
            f"{spec.ckpt_path}"
        )


def main() -> None:
    args = parse_args()
    runs_root = resolve_project_path(args.runs_root)
    output_dir = resolve_project_path(args.output_dir)
    eval_seeds = args.eval_seeds if args.eval_seeds is not None else list(range(args.num_eval_seeds))
    train_seeds = set(args.train_seeds) if args.train_seeds is not None else None

    specs = discover_checkpoints(runs_root, args.envs, args.methods, train_seeds)
    print_discovery(specs, eval_seeds)
    if args.dry_run:
        if 'SOURCE' in args.methods:
            print('Source RL: read held-out task learning histories from checkpoint configs; overrides:', args.source_history)
        return
    if not specs:
        raise RuntimeError(f"No checkpoints found under {runs_root}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(args.device)
    sample = not args.greedy
    rows = []

    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] Evaluating {spec.env} {spec.method} "
            f"train_seed={spec.train_seed} checkpoint={spec.ckpt_path.name}",
            flush=True,
        )
        rows.extend(
            evaluate_checkpoint(
                spec=spec,
                eval_seeds=eval_seeds,
                eval_episodes=args.eval_episodes,
                eval_timesteps_override=args.eval_timesteps,
                device=device,
                output_dir=output_dir,
                force=args.force,
                sample=sample,
            )
        )

    if 'SOURCE' in args.methods:
        overrides = {}
        for item in args.source_history:
            env, separator, raw_path = item.partition('=')
            if not separator or env not in ENV_LABELS:
                raise ValueError('--source-history must be darkroom=PATH or dktd=PATH')
            overrides[env] = resolve_project_path(raw_path)
        rows.extend(source_history_rows(specs, overrides, resolve_project_path(args.datasets_root), output_dir,
                                        args.eval_episodes, args.eval_timesteps, args.collection_env_split_seed))

    detail_fields = [
        "env",
        "method",
        "checkpoint",
        "train_seed",
        "eval_seed",
        "step",
        "reward_mean",
        "reward_std_over_env_episodes",
        "num_test_envs",
        "num_eval_episodes",
        "ckpt_path",
        "cache_path",
        "source_algorithm",
    ]
    write_csv(output_dir / "detail.csv", rows, detail_fields)

    summary_rows = aggregate_rows(rows)
    summary_fields = [
        "env",
        "method",
        "checkpoint",
        "step_mean",
        "step_std",
        "step_min",
        "step_max",
        "reward_mean",
        "reward_std",
        "reward_sem",
        "num_points",
        "num_train_seeds",
        "num_eval_seeds",
        "train_seeds",
        "eval_seeds",
    ]
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)

    saved_figures = plot_environment(args.envs, rows, output_dir, args.window, args.formats, args.dpi)

    print(f"Wrote {output_dir / 'detail.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    for figure_path in saved_figures:
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
