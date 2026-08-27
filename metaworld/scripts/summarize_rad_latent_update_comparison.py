"""
Summarize RAD latent-update comparison results.

Example:
    python scripts/summarize_rad_latent_update_comparison.py --output runs/rad_latent_update_summary.csv
"""

import argparse
import csv
from pathlib import Path

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
    'replace': 'RAD-ml1-window-open-v3-seed0-replace',
    'residual': 'RAD-ml1-window-open-v3-seed0-residual',
    'multiplicative_gate': 'RAD-ml1-window-open-v3-seed0-multiplicative_gate',
    'gru_gate': 'RAD-ml1-window-open-v3-seed0-gru_gate',
}


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
        'best_final_success_rate': checkpoint.get('final_success_rate'),
    }


def load_eval_result(path):
    if not path.exists():
        return {}
    rewards = np.load(path)
    result = {
        'eval_mean_reward': float(rewards.mean()),
        'eval_std_reward': float(rewards.std()),
        'eval_num_envs': int(rewards.shape[0]) if rewards.ndim >= 1 else 0,
        'eval_num_episodes': int(rewards.shape[1]) if rewards.ndim >= 2 else 0,
    }
    success_path = path.with_name('eval_success.npy')
    if success_path.exists():
        result['eval_mean_success'] = float(np.load(success_path).mean())
    return result


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
        'best_final_success_rate',
        'eval_mean_reward',
        'eval_std_reward',
        'eval_mean_success',
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


if __name__ == '__main__':
    main()

