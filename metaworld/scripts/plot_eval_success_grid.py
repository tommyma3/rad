"""
Plot AD and RAD Meta-world evaluation success rates (solved-within-k metric)
for several tasks in a single 1xN figure with near-square subplots.

Examples:
    uv run python scripts/plot_eval_success_grid.py
    uv run python scripts/plot_eval_success_grid.py --tasks reach-v3 push-v3 --window 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "AD": "AD",
    "RAD": "RAD",
}

METHOD_COLORS = {
    "AD": "#0072B2",
    "RAD": "#D55E00",
}

DEFAULT_TASKS = ["reach-v3", "push-v3", "coffee-button-v3", "drawer-close-v3"]

SCRIPT_DIR = Path(__file__).resolve().parents[1]

RUN_NAME_PATTERN = re.compile(r"^(?:AD|RAD)-ml1-(.+?)(?:-seed\d+)?$")


def normalize_task(value: str) -> str:
    task = value.strip().lower().replace("_", "-")
    if "-v" not in task:
        task = f"{task}-v3"
    return task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AD/RAD solved-within-k success curves for several ML1 tasks in one figure."
    )
    parser.add_argument(
        "--tasks",
        type=normalize_task,
        nargs="+",
        default=DEFAULT_TASKS,
        help="Meta-world tasks to plot, one subplot each, in order.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=SCRIPT_DIR / "runs",
        help="Directory containing AD/RAD run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Centered moving-average smoothing window in episodes.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats, e.g. pdf png svg.",
    )
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average that preserves the original series length."""
    if window <= 1:
        return values.astype(float, copy=True)

    window = min(window, values.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values.astype(float), (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def available_tasks(runs_dir: Path) -> list[str]:
    tasks = set()
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        match = RUN_NAME_PATTERN.match(run_dir.name)
        if match:
            tasks.add(match.group(1))
    return sorted(tasks)


def load_success_file(result_path: Path) -> np.ndarray:
    success = np.asarray(np.load(result_path), dtype=float)
    if success.ndim == 1:
        success = success[None, :]
    elif success.ndim != 2:
        raise ValueError(f"{result_path} must contain a 1D or 2D array, got {success.shape}.")

    return success


def load_method_success(runs_dir: Path, method: str, task: str) -> tuple[np.ndarray, list[Path]]:
    pattern = f"{method}-ml1-{task}*/eval_success.npy"
    result_paths = sorted(runs_dir.glob(pattern))

    if not result_paths:
        expected = runs_dir / f"{method}-ml1-{task}-varTrue" / "eval_success.npy"
        tasks = available_tasks(runs_dir)
        task_hint = f" Available tasks: {', '.join(tasks)}." if tasks else ""
        raise FileNotFoundError(
            f"No success evaluation results found for {method} on {task}. "
            f"Expected files like {expected}. Run metaworld/evaluate.py or "
            f"metaworld/evaluate_rad.py again to create eval_success.npy.{task_hint}"
        )

    trials = []
    episode_count = None
    for result_path in result_paths:
        success = load_success_file(result_path)

        if episode_count is None:
            episode_count = success.shape[1]
        elif success.shape[1] != episode_count:
            raise ValueError(
                f"Episode count mismatch for {method}: {result_path} has "
                f"{success.shape[1]} episodes, expected {episode_count}."
            )

        trials.append(success)

    return np.concatenate(trials, axis=0), result_paths


def summarize_best_within_k(success: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solved = np.maximum.accumulate(success.astype(float), axis=1)

    mean = solved.mean(axis=0)
    std = solved.std(axis=0)

    smooth_mean = moving_average(mean, window)
    smooth_std = moving_average(std, window)
    episodes = np.arange(1, mean.size + 1)

    return episodes, smooth_mean, smooth_std


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (11.0, 3.0),
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "lines.linewidth": 1.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def task_label(task: str) -> str:
    name = re.sub(r"-v\d+$", "", task)
    return "-".join(word.capitalize() for word in name.split("-"))


def episode_ticks(episode_count: int) -> list[int]:
    return sorted({1, episode_count // 2, episode_count})


def plot_grid(
    tasks: list[str],
    runs_dir: Path,
    output_dir: Path,
    window: int,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    configure_style()

    fig, axes = plt.subplots(1, len(tasks), sharey=True)
    axes = np.atleast_1d(axes)
    loaded_paths: dict[str, dict[str, list[Path]]] = {}

    for ax, task in zip(axes, tasks):
        loaded_paths[task] = {}
        for method in ("AD", "RAD"):
            success, paths = load_method_success(runs_dir, method, task)
            loaded_paths[task][method] = paths
            episodes, mean, std = summarize_best_within_k(success, window)

            color = METHOD_COLORS[method]
            label = METHOD_LABELS[method]
            lower = np.maximum(mean - std, 0.0)
            upper = np.minimum(mean + std, 1.0)

            ax.plot(episodes, mean, color=color, label=label)
            ax.fill_between(episodes, lower, upper, color=color, alpha=0.18, linewidth=0.0)

        ax.set_title(task_label(task))
        ax.set_xlabel("Episode")
        ax.set_xlim(1, episodes[-1])
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(episode_ticks(episodes.size))
        ax.margins(x=0.01)

    axes[0].set_ylabel("Success Rate")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=1.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    stem = "ad_rad_ml1_grid_eval_success_best_within_k"
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt.lstrip('.')}"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
        saved_paths.append(output_path)

    plt.close(fig)

    print(f"Plotted solved-within-k success curves for: {', '.join(tasks)}.")
    for task in tasks:
        for method, paths in loaded_paths[task].items():
            joined = ", ".join(str(p) for p in paths)
            print(f"  {task} {method}: {joined}")
    for output_path in saved_paths:
        print(f"Saved {output_path}")

    return saved_paths


def main() -> None:
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window must be at least 1.")
    if not args.tasks:
        raise ValueError("--tasks must contain at least one task.")

    plot_grid(
        tasks=args.tasks,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        window=args.window,
        formats=args.formats,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
