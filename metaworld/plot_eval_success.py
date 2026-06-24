"""
Plot AD and RAD Meta-world evaluation success rates for a selected task.

Examples:
    uv run python plot_eval_success.py --task reach-v3
    uv run python plot_eval_success.py --task door-close-v3 --window 5
"""

from __future__ import annotations

import argparse
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

SCRIPT_DIR = Path(__file__).resolve().parent


def normalize_task(value: str) -> str:
    task = value.strip().lower().replace("_", "-")
    if "-v" not in task:
        task = f"{task}-v3"
    return task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-episode evaluation success rates for AD and RAD on a Meta-world ML1 task."
    )
    parser.add_argument(
        "--task",
        type=normalize_task,
        required=True,
        help="Meta-world task to plot, e.g. reach-v3, door-close-v3, window-open-v3.",
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
    for run_dir in runs_dir.glob("*-ml1-*"):
        name = run_dir.name
        if name.startswith("RAD-pretrain-"):
            continue
        for prefix in ("AD-ml1-", "RAD-ml1-"):
            if name.startswith(prefix):
                task = name[len(prefix) :].split("-var", 1)[0]
                tasks.add(task)
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


def summarize_success(success: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = success.mean(axis=0)
    std = success.std(axis=0)

    smooth_mean = moving_average(mean, window)
    smooth_std = moving_average(std, window)
    episodes = np.arange(1, mean.size + 1)

    return episodes, smooth_mean, smooth_std


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.6, 4.2),
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def task_label(task: str) -> str:
    return task.replace("-", " ").title().replace(" V3", "-v3")


def plot_success(
    task: str,
    runs_dir: Path,
    output_dir: Path,
    window: int,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    configure_style()

    fig, ax = plt.subplots()
    loaded_paths: dict[str, list[Path]] = {}
    episodes = None

    for method in ("AD", "RAD"):
        success, paths = load_method_success(runs_dir, method, task)
        loaded_paths[method] = paths
        episodes, mean, std = summarize_success(success, window)

        color = METHOD_COLORS[method]
        label = f"{METHOD_LABELS[method]} (n={success.shape[0]})"
        lower = np.maximum(mean - std, 0.0)
        upper = np.minimum(mean + std, 1.0)

        ax.plot(episodes, mean, color=color, label=label)
        ax.fill_between(episodes, lower, upper, color=color, alpha=0.18, linewidth=0.0)

    if episodes is None:
        raise RuntimeError("No success data loaded.")

    ax.set_title(f"{task_label(task)} Evaluation Success")
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Success rate")
    ax.set_xlim(1, episodes[-1])
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.01)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    stem = f"ad_rad_ml1_{task}_eval_success"
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt.lstrip('.')}"
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        saved_paths.append(output_path)

    plt.close(fig)

    print(f"Plotted {task_label(task)} evaluation success rates.")
    for method, paths in loaded_paths.items():
        joined = ", ".join(str(p) for p in paths)
        print(f"  {method}: {joined}")
    for output_path in saved_paths:
        print(f"Saved {output_path}")

    return saved_paths


def main() -> None:
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window must be at least 1.")

    plot_success(
        task=args.task,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        window=args.window,
        formats=args.formats,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
