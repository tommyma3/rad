"""Compare explicit AD, RAD, AD_DPT and RAD_DPT checkpoints on Darkroom.

Example (run from gridworld): python scripts/evaluate_tokenization.py
  --checkpoint AD=runs/AD-darkroom-seed0 --checkpoint RAD=runs/RAD-darkroom-seed0
  --checkpoint AD_DPT=runs/tokenization/AD_DPT-darkroom-seed42
  --checkpoint RAD_DPT=runs/tokenization/RAD_DPT-darkroom-seed42

Directories select the largest numeric ckpt-N.pt. Pass best-model.pt explicitly
to evaluate it. Legacy model evaluators are called without behavioral changes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_dataset import collection_task_ids
from evaluate_ad_rad_curves import build_eval_envs, load_model, set_eval_seed
from model.transition_ad import TOKENIZED_MODELS
from utils import normalize_compiled_state_dict


LABELS = {'AD': 'AD', 'RAD': 'RAD', 'AD_DPT': 'AD (DPT-tokenized)', 'RAD_DPT': 'RAD (DPT-tokenized)'}


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        candidates = [(int(match[1]), candidate) for candidate in path.glob('ckpt-*.pt')
                      if (match := re.fullmatch(r'ckpt-(\d+)\.pt', candidate.name))]
        if not candidates:
            raise FileNotFoundError(f'No numeric training checkpoints in {path}')
        return max(candidates)[1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def split_audit(config, collection_seed):
    """Report legacy group selection accurately without migrating any datasets."""
    count = config['grid_size'] ** 2
    ids = list(range(count))
    random.Random(config['env_split_seed']).shuffle(ids)
    split = round(count * config['train_env_ratio'])
    order = collection_task_ids(config['grid_size'], 2, collection_seed)
    train_tasks = {order[group] for group in ids[:split]}
    test_tasks = set(ids[split:])
    return {'collection_env_split_seed': collection_seed,
            'training_goal_ids': sorted(train_tasks), 'evaluation_goal_ids': sorted(test_tasks),
            'overlap_goal_ids': sorted(train_tasks & test_tasks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='METHOD=PATH')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/tokenization/comparison'))
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--collection-env-split-seed', type=int, default=0)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--allow-partial', action='store_true', help='Allow fewer than four methods for smoke tests')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    entries = []
    for value in args.checkpoint:
        method, path = value.split('=', 1)
        if method not in LABELS:
            parser.error(f'Unknown method {method}')
        entries.append((method, resolve_checkpoint(path)))
    if not args.allow_partial and {method for method, _ in entries} != set(LABELS):
        parser.error('Supply all four methods, or use --allow-partial for smoke tests')
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, curves, seen = [], {}, set()
    environment = None
    for method, path in entries:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = dict(checkpoint['config'])
        if config['model'] != method or config['env'] != 'darkroom' or checkpoint.get('phase', 'train') != 'train':
            raise ValueError(f'{path} is not a {method} Darkroom policy-training checkpoint')
        signature = tuple(config[key] for key in ('grid_size', 'horizon', 'env_split_seed', 'train_env_ratio'))
        if environment is not None and signature != environment:
            raise ValueError('All checkpoints must use the same Darkroom task split and horizon')
        environment = signature
        seed_key = (method, config.get('seed', 42), config['env_split_seed'])
        if seed_key in seen:
            raise ValueError('Multiple checkpoints from the same method/training seed are not independent replicates')
        seen.add(seed_key)
        if method in TOKENIZED_MODELS:
            config['device'] = device
            model = TOKENIZED_MODELS[method](config).to(device)
            model.load_state_dict(normalize_compiled_state_dict(checkpoint['model']), strict=True)
            model.eval()
        else:
            model, config, checkpoint = load_model(path, device)
        audit = split_audit(config, args.collection_env_split_seed)
        if audit['overlap_goal_ids']:
            print(f"{method}: legacy source-group selection overlaps {len(audit['overlap_goal_ids'])} evaluation goals; see metrics.json", flush=True)
        samples, elapsed, compression_counts = [], [], []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for seed in args.eval_seeds:
            set_eval_seed(seed)
            envs = build_eval_envs(config)
            try:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = model.evaluate_in_context(envs, args.episodes * config['horizon'], sample=not args.greedy)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed.append(time.perf_counter() - start)
                samples.append(output['reward_episode'])
                compression_counts.append(output.get('total_compressions', 0))
            finally:
                envs.close()
        rewards = np.stack(samples)
        per_episode = rewards.mean(axis=(0, 1))
        curves.setdefault(method, []).append(per_episode)
        identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        artifact = args.output_dir / f'{method}-{identity}.npz'
        np.savez_compressed(artifact, reward_episode=rewards, eval_seeds=args.eval_seeds)
        result = {'method': method, 'checkpoint': str(path.resolve()), 'training_seed': config.get('seed', 42),
                  'step': checkpoint.get('step'), 'artifact': str(artifact.resolve()),
                  'mean_episode_return': float(rewards.mean()),
                  'cumulative_reward': float(rewards.sum(-1).mean()),
                  'final_10_episode_return': float(rewards[..., -10:].mean()),
                  'seconds_per_vector_step': float(np.mean(elapsed) / (args.episodes * config['horizon'])),
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  'mean_compressions': float(np.mean(compression_counts)), 'split_audit': audit,
                  'policy_token_budget': config.get('policy_token_budget', 3 * config['n_transit']),
                  'initialization': config.get('initialization', 'legacy-checkpoint'),
                  'train_source_timesteps': config.get('train_source_timesteps'),
                  'train_batch_size': config.get('train_batch_size')}
        results.append(result)
        print(f"{method}: mean episode return {result['mean_episode_return']:.3f}", flush=True)
        del model, checkpoint
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    figure, axis = plt.subplots(figsize=(7, 4))
    aggregates = {}
    for method in LABELS:
        if method not in curves:
            continue
        values = np.stack(curves[method])
        mean = values.mean(0)
        half_width = 1.96 * values.std(0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
        episodes = np.arange(1, len(mean) + 1)
        line, = axis.plot(episodes, mean, label=LABELS[method])
        if len(values) > 1:
            axis.fill_between(episodes, mean - half_width, mean + half_width, color=line.get_color(), alpha=.15)
        aggregates[method] = {'training_seeds': len(values), 'episode_mean': mean.tolist(),
                              'episode_95pct_half_width': half_width.tolist()}
    axis.set(xlabel='Episode', ylabel='Episode return', title='Darkroom tokenization ablation')
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / 'comparison.png', dpi=200)
    plt.close(figure)
    (args.output_dir / 'metrics.json').write_text(json.dumps({'runs': results, 'aggregates': aggregates}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
