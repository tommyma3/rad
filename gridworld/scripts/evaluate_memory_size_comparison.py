"""End-to-end Darkroom memory-size pipeline: multi-GPU training, best-model evaluation, one figure.

Stages:
    train      Run train_pretrain_compression.py and train_rad.py for every
               size/seed, chained per seed and distributed over --gpus
               (one job per GPU at a time, seeds round-robin across GPUs).
    evaluate   Run evaluate_rad.py --use_best on every trained run, writing
               eval_result.npy into each run directory.
    plot       Aggregate eval_result.npy across training seeds and draw a
               single near-square paper figure (PDF + PNG) with mean episode
               return and +-1 SEM over training seeds per memory size.
    all        train -> evaluate -> plot.

Examples:
    python scripts/evaluate_memory_size_comparison.py --gpus 0 1 2
    python scripts/evaluate_memory_size_comparison.py --stage plot
    python scripts/evaluate_memory_size_comparison.py --gpus 0 --steps 100 \
        --batch-size 8 --episodes 10   # end-to-end smoke test
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from memory_size_experiment import PROTOCOL, SIZES, run_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'plot'], default='all')
    parser.add_argument('--sizes', nargs='+', type=int, default=list(SIZES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--gpus', nargs='+', default=None,
                        help='GPU indices for round-robin scheduling; defaults to all visible GPUs')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Evaluation episodes passed to evaluate_rad.py --eval_episodes')
    parser.add_argument('--runs-root', type=Path, default=PROJECT / 'runs/memory_size_darkroom')
    parser.add_argument('--traj-dir', type=Path, default=PROJECT / 'datasets')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Defaults to <runs-root>/comparison')
    parser.add_argument('--config', default='rad_dr_memory_size')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--steps', type=int,
                        help='Training budget override for both phases (smoke tests/pilots)')
    parser.add_argument('--batch-size', type=int,
                        help='Per-process batch size override for both phases (smoke tests/pilots)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip pretraining with existing pretrain-final.pt and policy training '
                             'with existing best-model.pt (treated as completed)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    return parser.parse_args()


def validate_args(args):
    for values in (args.sizes, args.seeds):
        if len(set(values)) != len(values):
            raise SystemExit('Size and seed lists must not contain duplicates')
    if any(size <= 0 or size % 3 for size in args.sizes):
        raise SystemExit('Sizes must be positive multiples of three')
    if args.episodes < 1 or (args.steps is not None and args.steps < 1) \
            or (args.batch_size is not None and args.batch_size < 1):
        raise SystemExit('Budgets must be positive')
    if args.num_workers is not None and args.num_workers < 0:
        raise SystemExit('--num-workers must be nonnegative')
    if not args.gpus:
        args.gpus = [str(index) for index in range(torch.cuda.device_count())] or ['0']


def training_command(args, size, seed, pretrain):
    command = [args.python, str(PROJECT / ('train_pretrain_compression.py' if pretrain else 'train_rad.py')),
               '--env', 'darkroom', '--config', args.config, '--n_latents', str(size),
               '--seed', str(seed), '--env_split_seed', '0',
               '--runs_root', str(args.runs_root), '--traj_dir', str(args.traj_dir),
               '--run_name', run_name(size, seed, pretrain)]
    if args.num_workers is not None:
        command += ['--num_workers', str(args.num_workers)]
    if not pretrain:
        command += ['--pretrain_ckpt',
                    str(args.runs_root / run_name(size, seed, True) / 'pretrain-final.pt')]
    if args.steps is not None:
        command += ['--steps', str(args.steps)]
    if args.batch_size is not None:
        command += ['--batch_size', str(args.batch_size)]
    return command


def training_artifact(args, size, seed, pretrain):
    name = 'pretrain-final.pt' if pretrain else 'best-model.pt'
    return args.runs_root / run_name(size, seed, pretrain) / name


def build_training_jobs(args):
    """Chain pretrain->train per seed; assign whole seed chains to GPUs round-robin."""
    jobs_by_gpu = {gpu: [] for gpu in args.gpus}
    for seed_index, seed in enumerate(args.seeds):
        gpu = args.gpus[seed_index % len(args.gpus)]
        for size in args.sizes:
            for pretrain in (True, False):
                if args.skip_existing and training_artifact(args, size, seed, pretrain).is_file():
                    print(f'Skip completed {"pretrain" if pretrain else "train"}: '
                          f'memory{size} seed{seed}', flush=True)
                    continue
                label = f'{"pretrain" if pretrain else "train"}-darkroom-memory{size}-train{seed}'
                jobs_by_gpu[gpu].append((label, training_command(args, size, seed, pretrain)))
    return jobs_by_gpu


def build_evaluation_jobs(args):
    jobs = []
    for seed in args.seeds:
        for size in args.sizes:
            run_dir = args.runs_root / run_name(size, seed)
            if not (run_dir / 'best-model.pt').is_file():
                if args.dry_run:
                    print(f'[dry-run] best-model.pt not present yet: {run_dir}', flush=True)
                else:
                    raise FileNotFoundError(
                        f'Missing best-model.pt in {run_dir}; run --stage train first '
                        f'(or check that training saved a best model)')
            label = f'evaluate-darkroom-memory{size}-train{seed}'
            command = [args.python, str(PROJECT / 'evaluate_rad.py'), '--ckpt_dir', str(run_dir),
                       '--use_best', '--eval_episodes', str(args.episodes)]
            jobs.append((label, command))
    jobs_by_gpu = {gpu: [] for gpu in args.gpus}
    for index, job in enumerate(jobs):
        jobs_by_gpu[args.gpus[index % len(args.gpus)]].append(job)
    return jobs_by_gpu


def execute_jobs(jobs_by_gpu, dry_run):
    """Run each GPU's queue serially; queues run in parallel across GPUs."""
    results, failures = [], []
    lock = threading.Lock()

    def worker(gpu, jobs):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        for label, command in jobs:
            print(f'[GPU {gpu}] {label}: {subprocess.list2cmdline(command)}', flush=True)
            if dry_run:
                continue
            start = time.monotonic()
            completed = subprocess.run(command, cwd=PROJECT, env=environment)
            entry = dict(gpu=str(gpu), label=label, command=command,
                         returncode=completed.returncode,
                         seconds=round(time.monotonic() - start, 3))
            with lock:
                results.append(entry)
            status = 'done' if completed.returncode == 0 else f'FAILED({completed.returncode})'
            print(f'[GPU {gpu}] {label}: {status} in {entry["seconds"]:.1f}s', flush=True)
            if completed.returncode != 0:
                with lock:
                    failures.append(entry)
                return

    threads = [threading.Thread(target=worker, args=(gpu, jobs), daemon=True)
               for gpu, jobs in jobs_by_gpu.items() if jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, failures


def collect_curves(args):
    """Return {size: {seed: rewards[goal, episode]}} from evaluate_rad.py outputs."""
    curves, missing = {}, []
    for seed in args.seeds:
        for size in args.sizes:
            result_path = args.runs_root / run_name(size, seed) / 'eval_result.npy'
            if not result_path.is_file():
                missing.append(str(result_path))
                continue
            rewards = np.load(result_path)
            if rewards.ndim != 2:
                raise ValueError(f'{result_path} must be a 2D [goal, episode] array, '
                                 f'got shape {rewards.shape}')
            curves.setdefault(size, {})[seed] = rewards
    if missing:
        raise FileNotFoundError('Missing evaluation results (run --stage evaluate first):\n'
                                + '\n'.join(missing))
    episode_counts = {rewards.shape[1] for seeds in curves.values() for rewards in seeds.values()}
    goal_counts = {rewards.shape[0] for seeds in curves.values() for rewards in seeds.values()}
    if len(episode_counts) != 1 or len(goal_counts) != 1:
        raise ValueError(f'Inconsistent evaluation shapes: episodes={episode_counts}, goals={goal_counts}')
    return curves


def write_csv(filename, rows):
    if rows:
        with filename.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_plot(curves, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.22,
        'grid.linewidth': 0.7,
        'lines.linewidth': 1.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    sizes = sorted(curves)
    colors = plt.get_cmap('viridis')(np.linspace(0.05, 0.9, len(sizes)))
    fig, axis = plt.subplots(figsize=(4.6, 4.2))

    curve_rows, seed_rows, summary_rows = [], [], []
    for size, color in zip(sizes, colors):
        # Average held-out goals within each training seed, then across seeds.
        seed_curves = np.stack([curves[size][seed].mean(axis=0) for seed in sorted(curves[size])])
        mean = seed_curves.mean(axis=0)
        sem = (seed_curves.std(axis=0, ddof=1) / np.sqrt(len(seed_curves))
               if len(seed_curves) > 1 else np.zeros_like(mean))
        episodes = np.arange(1, len(mean) + 1)

        axis.plot(episodes, mean, color=color, label=f'{size}')
        if len(seed_curves) > 1:
            axis.fill_between(episodes, mean - sem, mean + sem, color=color, alpha=0.18,
                              linewidth=0.0)

        per_seed_returns = seed_curves.mean(axis=1)
        seed_rows.extend(dict(n_latents=size, train_seed=int(seed), mean_return=float(value))
                         for seed, value in zip(sorted(curves[size]), per_seed_returns))
        summary_rows.append(dict(n_latents=size, n_training_seeds=len(seed_curves),
                                 mean_return=float(per_seed_returns.mean()),
                                 sem_over_training_seeds=(float(per_seed_returns.std(ddof=1)
                                                                / np.sqrt(len(seed_curves)))
                                                          if len(seed_curves) > 1 else None)))
        curve_rows.extend(dict(n_latents=size, episode=int(episode), mean=float(value),
                               sem=float(band) if len(seed_curves) > 1 else None)
                          for episode, value, band in zip(episodes, mean, sem))

    axis.set(xlim=(1, len(mean)), ylim=(0, 20), xlabel='Episode', ylabel='Episode return',
             title='Darkroom: long-term memory size')
    axis.legend(title='Latent tokens', loc='upper left', ncol=2, frameon=False,
                handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
    axis.margins(x=0.01)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    for extension, kwargs in (('pdf', {}), ('png', {'dpi': 400})):
        figure_path = output_dir / f'memory_size_comparison.{extension}'
        fig.savefig(figure_path, bbox_inches='tight', **kwargs)
        figure_paths.append(figure_path)
    plt.close(fig)

    write_csv(output_dir / 'per_training_seed.csv', seed_rows)
    write_csv(output_dir / 'summary.csv', summary_rows)
    write_csv(output_dir / 'curves.csv', curve_rows)
    return figure_paths


def main():
    args = parse_args()
    validate_args(args)
    args.runs_root = args.runs_root.resolve()
    args.traj_dir = args.traj_dir.resolve()
    args.output_dir = (args.output_dir or args.runs_root / 'comparison').resolve()
    if Path(args.config).suffix in ('.yaml', '.yml'):
        args.config = str(Path(args.config).resolve())

    manifest = dict(protocol=PROTOCOL, stage=args.stage, sizes=args.sizes, train_seeds=args.seeds,
                    gpus=args.gpus, episodes=args.episodes, steps=args.steps,
                    batch_size=args.batch_size, skip_existing=args.skip_existing,
                    runs_root=str(args.runs_root), traj_dir=str(args.traj_dir),
                    config=args.config, jobs=[], failures=[])
    stages = ('train', 'evaluate', 'plot') if args.stage == 'all' else (args.stage,)
    failures = []
    try:
        if 'train' in stages:
            jobs_by_gpu = build_training_jobs(args)
            manifest['jobs'], failures = execute_jobs(jobs_by_gpu, args.dry_run)
            if failures:
                raise RuntimeError(f'{len(failures)} training job(s) failed; see manifest')
        if 'evaluate' in stages:
            jobs_by_gpu = build_evaluation_jobs(args)
            results, failures = execute_jobs(jobs_by_gpu, args.dry_run)
            manifest['jobs'] += results
            if failures:
                raise RuntimeError(f'{len(failures)} evaluation job(s) failed; see manifest')
        if 'plot' in stages:
            figure_paths = save_plot(collect_curves(args), args.output_dir)
            for figure_path in figure_paths:
                print(f'Saved {figure_path}', flush=True)
    finally:
        manifest['failures'] = [failure['label'] for failure in failures]
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / 'pipeline-manifest.json').write_text(
                json.dumps(manifest, indent=2))

    if not args.dry_run:
        protocol = dict(protocol=PROTOCOL, checkpoint_selection='best-model.pt',
                        evaluator='evaluate_rad.py --use_best', episodes=args.episodes,
                        sizes=args.sizes, train_seeds=args.seeds,
                        curve_aggregation='mean over held-out goals per training seed; '
                                          'band is +-1 SEM across training seeds')
        (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        print(f'Artifacts written to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
