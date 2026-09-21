"""
Summarize RAD latent-update results and plot mean reward by evaluation episode.

Example:
    python scripts/summarize_rad_latent_update_comparison.py --output runs/rad_latent_update_summary.csv
"""

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np
import torch
import yaml


class SummaryConfigLoader(yaml.SafeLoader):
    pass


def construct_torch_device(loader, node):
    values = loader.construct_sequence(node)
    return str(values[0]) if values else None


SummaryConfigLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:torch.device',
    construct_torch_device,
)


DEFAULT_RUN_NAMES = {
    'replace': 'RAD-dktd-seed0-replace',
    'residual': 'RAD-dktd-seed0-residual',
    'multiplicative_gate': 'RAD-dktd-seed0-multiplicative_gate',
    'gru_gate': 'RAD-dktd-seed0-gru_gate',
}


VARIANT_STYLES = {
    'replace': ('Replace', '#0072B2', '-'),
    'residual': ('Residual', '#D55E00', '--'),
    'multiplicative_gate': ('Multiplicative gate', '#009E73', '-.'),
    'gru_gate': ('GRU gate', '#CC79A7', ':'),
}


def plot_eval_curves(runs_root, variants, output_path):
    """Plot unsmoothed per-episode means over evaluation environments.

    Each eval_result.npy has shape (environments, episodes). Curves retain
    their own episode counts; absent results are reported and skipped.
    output_path is a filename stem, optionally ending in .pdf or .png.
    """
    curves = {}
    for variant in dict.fromkeys(variants):
        result_path = runs_root / DEFAULT_RUN_NAMES[variant] / 'eval_result.npy'
        if not result_path.exists():
            warnings.warn(f'Skipping {variant}: no evaluation results at {result_path}.')
            continue
        rewards = np.load(result_path, allow_pickle=False)
        if rewards.ndim != 2 or 0 in rewards.shape:
            raise ValueError(
                f'{result_path}: expected a nonempty (environments, episodes) '
                f'array, got {rewards.shape}.'
            )
        if not np.isfinite(rewards).all():
            raise ValueError(f'{result_path}: evaluation rewards must all be finite.')
        curves[variant] = rewards.mean(axis=0, dtype=np.float64)

    if not curves:
        warnings.warn('No evaluation curves available; no figure written.')
        return []

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_path = Path(output_path)
    if output_path.suffix.lower() in {'.pdf', '.png'}:
        output_path = output_path.with_suffix('')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = {
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
    saved_paths = []
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        try:
            for variant, mean in curves.items():
                label, color, linestyle = VARIANT_STYLES[variant]
                ax.plot(
                    np.arange(1, mean.size + 1), mean,
                    label=label, color=color, linestyle=linestyle, linewidth=1.6,
                    marker='o' if mean.size == 1 else None, markersize=3,
                )
            ax.set_xlabel('Evaluation episode')
            ax.set_ylabel('Average episode reward')
            max_episodes = max(mean.size for mean in curves.values())
            ax.set_xlim((1, max_episodes) if max_episodes > 1 else (0.5, 1.5))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='0.88', linewidth=0.5)
            ax.margins(y=0.08)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),
                ncol=min(2, len(curves)), frameon=False,
                handlelength=2.5, columnspacing=1.2,
            )
            fig.tight_layout(rect=(0, 0.17, 1, 1), pad=0.6)
            for extension in ('pdf', 'png'):
                destination = output_path.parent / f'{output_path.name}.{extension}'
                fig.savefig(destination, dpi=400, bbox_inches='tight', pad_inches=0.04)
                saved_paths.append(destination)
        finally:
            plt.close(fig)
    return saved_paths


def load_yaml(path):
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return yaml.load(f, Loader=SummaryConfigLoader) or {}


def load_checkpoint_metadata(path):
    if not path.exists():
        return {}
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'best_step': checkpoint.get('step'),
        'best_eval_reward': checkpoint.get('eval_reward'),
    }


def load_eval_result(path):
    if not path.exists():
        return {}
    rewards = np.load(path)
    return {
        'eval_mean_reward': float(rewards.mean()),
        'eval_std_reward': float(rewards.std()),
        'eval_num_envs': int(rewards.shape[0]) if rewards.ndim >= 1 else 0,
        'eval_num_episodes': int(rewards.shape[1]) if rewards.ndim >= 2 else 0,
    }


def summarize_run(runs_root, variant, run_name):
    run_dir = runs_root / run_name
    config = load_yaml(run_dir / 'config.yaml')
    row = {
        'variant': variant,
        'run_name': run_name,
        'run_dir': str(run_dir),
        'exists': run_dir.exists(),
        'latent_update_mode': config.get('latent_update_mode', variant),
        'n_transit': config.get('n_transit'),
        'n_compress_tokens': config.get('n_compress_tokens'),
        'train_timesteps': config.get('train_timesteps'),
    }
    row.update(load_checkpoint_metadata(run_dir / 'best-model.pt'))
    row.update(load_eval_result(run_dir / 'eval_result.npy'))
    return row


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize RAD latent-update comparison runs.')
    parser.add_argument('--runs_root', default='./runs')
    parser.add_argument('--variants', nargs='+', default=list(DEFAULT_RUN_NAMES), choices=list(DEFAULT_RUN_NAMES))
    parser.add_argument('--output', default=None, help='Optional CSV output path.')
    parser.add_argument(
        '--plot_output', default=None,
        help='Figure filename stem (or .pdf/.png path), relative to gridworld. '
             'Defaults to <runs_root>/rad_latent_update_comparison; writes PDF and PNG.',
    )
    parser.add_argument('--no_plot', action='store_true', help='Only summarize; skip the figure.')
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = project_dir / runs_root

    rows = [
        summarize_run(runs_root, variant, DEFAULT_RUN_NAMES[variant])
        for variant in args.variants
    ]
    fieldnames = [
        'variant',
        'latent_update_mode',
        'exists',
        'best_step',
        'best_eval_reward',
        'eval_mean_reward',
        'eval_std_reward',
        'eval_num_envs',
        'eval_num_episodes',
        'n_transit',
        'n_compress_tokens',
        'train_timesteps',
        'run_name',
        'run_dir',
    ]

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {output_path}')

    print(','.join(fieldnames))
    for row in rows:
        print(','.join('' if row.get(name) is None else str(row.get(name)) for name in fieldnames))

    if not args.no_plot:
        plot_path = Path(args.plot_output) if args.plot_output else runs_root / 'rad_latent_update_comparison'
        if not plot_path.is_absolute():
            plot_path = project_dir / plot_path
        for saved_path in plot_eval_curves(runs_root, args.variants, plot_path):
            print(f'Wrote {saved_path}')


if __name__ == '__main__':
    main()
