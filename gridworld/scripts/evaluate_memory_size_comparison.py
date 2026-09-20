"""Evaluate final Darkroom memory-size checkpoints with paired task/seed trials."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from memory_size_experiment import PROTOCOL, SIZES, BASELINE_SIZE, run_name, validate_memory_size_config
from memory_capacity import policy_token_capacity
from compressor_experiment import validate_checkpoint_config
from model.compressed_ad import RAD
from env import SAMPLE_ENVIRONMENT, make_env
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import normalize_compiled_state_dict
from evaluate_compressor_comparison import metrics, write_csv, synchronize


def comparison_signature(config):
    """Compare all resolved settings except the independent variable and provenance."""
    excluded = {'n_compress_tokens', 'seed', 'data_seed', 'device', 'log_dir', 'run_name',
                'runs_root', 'traj_dir', 'dataset_audit', 'pretrain_checkpoint',
                'pretrain_provenance', 'amp_retries'}
    return {key: value for key, value in config.items() if key not in excluded}


def aggregate(rows):
    result = []
    for size in sorted({row['n_latents'] for row in rows}):
        subset = [row for row in rows if row['n_latents'] == size]
        if len({row['train_seed'] for row in subset}) != len(subset):
            raise ValueError('Duplicate training-seed replicate')
        for key in ('mean_return', 'early_1_10', 'first_50', 'late_last_20', 'after_50'):
            values = np.array([row[key] for row in subset if key in row])
            if not len(values):
                continue
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            result.append(dict(n_latents=size, metric=key, n_training_seeds=len(values),
                               mean=float(values.mean()), std_over_training_seeds=std,
                               sem=std / np.sqrt(len(values)) if std is not None else None))
    return result


def paired_differences(rows):
    paired = []
    metric_keys = ('mean_return', 'early_1_10', 'first_50', 'late_last_20', 'after_50')
    for row in rows:
        baseline = next((r for r in rows if r['n_latents'] == BASELINE_SIZE
                         and r['train_seed'] == row['train_seed']), None)
        if baseline is not None and row['n_latents'] != BASELINE_SIZE:
            paired.append(dict(n_latents=row['n_latents'], train_seed=row['train_seed'],
                               **{key: row[key] - baseline[key] for key in metric_keys if key in row}))
    return paired


@torch.inference_mode()
def benchmark_compression(model, device, repeats):
    results = {}
    for recurrent in (False, True):
        length = model._recent_capacity(recurrent) + 1 - model.short_memory_keep_tokens
        if recurrent:
            length += model.n_compress_tokens
        generator = torch.Generator(device=device).manual_seed(12345)
        context = torch.randn(8, length, model.config['tf_n_embd'], device=device, generator=generator)
        old = torch.zeros(8, model.n_compress_tokens, model.config['tf_n_embd'], device=device) if recurrent else None
        for _ in range(3):
            model._compress_sequence(context, False, old)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model._compress_sequence(context, False, old)
        synchronize(device)
        results['recurrent_compression_ms' if recurrent else 'first_compression_ms'] = (
            1000 * (time.perf_counter() - start) / repeats)
    return results


def save_outputs(output, rows, curves):
    write_csv(output / 'per_training_seed.csv', rows)
    write_csv(output / 'summary.csv', aggregate(rows))
    paired = paired_differences(rows)
    write_csv(output / 'paired_vs_15.csv', paired)
    write_csv(output / 'paired_summary_vs_15.csv', aggregate(paired))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 4))
    curve_rows = []
    for size in sorted({key[0] for key in curves}):
        values = np.stack([curve for (n, _), curve in curves.items() if n == size])
        mean = values.mean(0)
        sem = values.std(0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
        episodes = np.arange(1, len(mean) + 1)
        line, = axis.plot(episodes, mean, label=f'{size} latents')
        if len(values) > 1:
            axis.fill_between(episodes, mean - sem, mean + sem, color=line.get_color(), alpha=0.15)
        curve_rows.extend(dict(n_latents=size, episode=int(e), mean=float(m),
                               sem=float(s) if len(values) > 1 else None, n_training_seeds=len(values))
                          for e, m, s in zip(episodes, mean, sem))
    axis.set(xlabel='Episode', ylabel='Episode return', ylim=(0, 20), title='Darkroom long-term memory size')
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'adaptation.png', dpi=180)
    fig.savefig(output / 'adaptation.pdf')
    plt.close(fig)
    write_csv(output / 'adaptation.csv', curve_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=PROJECT / 'runs/memory_size_darkroom')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, default=list(SIZES))
    parser.add_argument('--train-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=list(range(20)))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--checkpoint-step', type=int, default=100000)
    parser.add_argument('--pretrain-steps', type=int, default=40000)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--benchmark-repeats', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.episodes, args.checkpoint_step, args.pretrain_steps, args.benchmark_repeats, args.threads) < 1:
        parser.error('Budgets must be positive')
    for values in (args.sizes, args.train_seeds, args.eval_seeds):
        if len(set(values)) != len(values):
            parser.error('Size and seed lists must not contain duplicates')
    if any(size <= 0 or size % 3 for size in args.sizes):
        parser.error('Sizes must be positive multiples of three')
    if not args.pilot and (args.checkpoint_step != 100000 or args.pretrain_steps != 40000):
        parser.error('Reduced training budgets require --pilot')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output = args.output_dir or args.runs_root / 'comparison'
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(protocol=PROTOCOL, pilot=args.pilot, sizes=args.sizes, train_seeds=args.train_seeds,
                    eval_seeds=args.eval_seeds, episodes=args.episodes, checkpoint_step=args.checkpoint_step,
                    pretrain_steps=args.pretrain_steps, action_sampling=True, device=str(device),
                    torch_version=torch.__version__, benchmark_repeats=args.benchmark_repeats,
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    reference_audit = reference_signature = reference_pretraining = reference_events = None
    rows, curves = [], {}
    for seed in args.train_seeds:
        for size in args.sizes:
            run = args.runs_root / run_name(size, seed)
            checkpoint_path = run / f'ckpt-{args.checkpoint_step}.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            config = dict(checkpoint['config'])
            validate_memory_size_config(config)
            if (config['n_compress_tokens'] != size or config['seed'] != seed or config['data_seed'] != seed
                    or checkpoint['step'] != args.checkpoint_step or config['train_timesteps'] != args.checkpoint_step):
                raise ValueError(f'Checkpoint size/seed/budget mismatch: {checkpoint_path}')
            audit = config.get('dataset_audit', {})
            if (len(audit.get('train_groups', [])) != 73 or len(audit.get('test_groups', [])) != 8
                    or set(audit['train_groups']) & set(audit['test_groups'])):
                raise ValueError('Missing verified 73/8 Darkroom goal split')
            identity = {key: audit[key] for key in ('data_sha256', 'train_groups', 'test_groups',
                                                   'group_goals', 'collection_env_split_seed')}
            signature = comparison_signature(config)
            if reference_audit is not None and (identity != reference_audit or signature != reference_signature):
                raise ValueError('Dataset or shared model/training settings differ across sizes/seeds')
            reference_audit, reference_signature = identity, signature
            pretrain_run = args.runs_root / run_name(size, seed, True)
            pretrain_path = pretrain_run / 'pretrain-final.pt'
            source = torch.load(pretrain_path, map_location='cpu', weights_only=False)
            validate_checkpoint_config(config, source['config'])
            provenance = config.get('pretrain_provenance', {})
            with pretrain_path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if (provenance.get('sha256') != digest or provenance.get('step') != args.pretrain_steps
                    or source['step'] != args.pretrain_steps
                    or source['config']['pretrain_timesteps'] != args.pretrain_steps
                    or source['config']['data_seed'] != seed):
                raise ValueError('Pretraining checkpoint provenance/budget mismatch')
            if any(source['config'].get(key) != value for key, value in provenance['settings'].items()):
                raise ValueError('Recorded pretraining settings disagree with the source checkpoint')
            settings = comparison_signature(source['config'])
            if reference_pretraining is not None and settings != reference_pretraining:
                raise ValueError('Pretraining settings differ across sizes/seeds')
            reference_pretraining = settings
            del source
            config.update(device=device, torch_compile=False)
            model = RAD(config).to(device).eval()
            model.load_state_dict(normalize_compiled_state_dict(checkpoint['model']), strict=True)
            del checkpoint
            model.set_curriculum(None)
            _, goals = SAMPLE_ENVIRONMENT['darkroom'](config)
            rewards, compressions, events = [], [], []
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()
            for eval_seed in args.eval_seeds:
                envs = DummyVecEnv([make_env(config, goal=goal) for goal in goals])
                try:
                    envs.seed(eval_seed)
                    result = model.evaluate_in_context(envs, config['horizon'] * args.episodes, action_seed=eval_seed)
                finally:
                    envs.close()
                rewards.append(result['reward_episode'])
                compressions.append(result['total_compressions'])
                events.append(result['compression_events'])
            synchronize(device)
            elapsed = time.perf_counter() - start
            rewards = np.stack(rewards)
            if rewards.shape != (len(args.eval_seeds), 8, args.episodes):
                raise ValueError(f'Unexpected reward shape: {rewards.shape}')
            if reference_events is not None and events != reference_events:
                raise ValueError('Compression boundaries differ across evaluation runs')
            reference_events = events
            np.savez_compressed(output / f'memory{size}-train{seed}.npz', rewards=rewards,
                                goals=np.asarray(goals), eval_seeds=args.eval_seeds, n_latents=size,
                                compressions=compressions, compression_events=np.asarray(events),
                                checkpoint=str(checkpoint_path.resolve()))
            row = dict(n_latents=size, train_seed=seed, **metrics(rewards),
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       compressor_parameter_count=sum(p.numel() for p in model.compression_transformer.parameters()),
                       latent_values=size * config['tf_n_embd'], policy_token_capacity=policy_token_capacity(config),
                       total_compressions=compressions[0], eval_seconds=elapsed,
                       eval_peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                       **benchmark_compression(model, device, args.benchmark_repeats))
            for stage, directory in (('pretrain', pretrain_run), ('train', run)):
                stage_metrics = json.loads((directory / f'{stage}-metrics.json').read_text())
                row[f'{stage}_seconds'] = stage_metrics['elapsed_seconds']
                row[f'{stage}_peak_gpu_bytes'] = stage_metrics['peak_gpu_bytes']
                if stage == 'pretrain':
                    row['pretrain_reconstruction_mse'] = stage_metrics['loss_recon']
            rows.append(row)
            curves[(size, seed)] = rewards.mean(axis=(0, 1))
            print(json.dumps(row), flush=True)
            del model
    save_outputs(output, rows, curves)


if __name__ == '__main__':
    main()
