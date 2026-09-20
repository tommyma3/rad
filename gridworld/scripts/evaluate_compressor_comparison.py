"""Evaluate final Darkroom AE/VAE/VQ-VAE checkpoints with paired seeds."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from compressor_experiment import PROTOCOL, VARIANTS
from env import make_env, SAMPLE_ENVIRONMENT
from model.compressed_ad import RAD
from utils import normalize_compiled_state_dict


def metrics(rewards):
    """Input: evaluation seed x goal x episode; each training seed is one replicate."""
    curve = rewards.mean(axis=(0, 1))
    result = dict(mean_return=float(curve.mean()), early_1_10=float(curve[:10].mean()),
                  first_50=float(curve[:50].mean()), late_last_20=float(curve[-20:].mean()))
    if len(curve) > 50:
        result['after_50'] = float(curve[50:].mean())
    return result


def write_csv(filename, rows):
    if rows:
        with filename.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_compression(model, device, batch_size, repeats):
    # Match the input length at the first capacity crossing in online RAD.
    length = model.max_seq_length + 1 - model.short_memory_keep_tokens
    generator = torch.Generator(device=device).manual_seed(12345)
    context = torch.randn(batch_size, length, model.config['tf_n_embd'], device=device, generator=generator)
    old = torch.zeros(batch_size, model.n_compress_tokens, model.config['tf_n_embd'], device=device)
    for _ in range(3):
        model._compress_sequence(context, False, old)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model._compress_sequence(context, False, old)
    synchronize(device)
    return 1000 * (time.perf_counter() - start) / repeats


def aggregate(rows):
    result = []
    for variant, mode in sorted({(row['variant'], row['latent_mode']) for row in rows}):
        subset = [row for row in rows if (row['variant'], row['latent_mode']) == (variant, mode)]
        for metric in ('mean_return', 'early_1_10', 'first_50', 'late_last_20', 'after_50'):
            values = np.array([row[metric] for row in subset if metric in row])
            if not len(values):
                continue
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result.append(dict(variant=variant, latent_mode=mode, metric=metric,
                               n_training_seeds=len(values), mean=float(values.mean()),
                               std_over_training_seeds=std, sem=std / np.sqrt(len(values))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=PROJECT / 'runs/compressor_darkroom')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--train-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=list(range(20)))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--checkpoint-step', type=int, default=100000, help='Change only for pilot evaluation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--benchmark-repeats', type=int, default=50)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.episodes, args.checkpoint_step, args.benchmark_repeats, args.threads) < 1:
        parser.error('Budgets and threads must be positive')
    for values in (args.variants, args.train_seeds, args.eval_seeds):
        if len(set(values)) != len(values):
            parser.error('Variants and seed lists must not contain duplicates')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output = args.output_dir or args.runs_root / 'comparison'
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(protocol=PROTOCOL, checkpoint_step=args.checkpoint_step,
                    train_seeds=args.train_seeds, eval_seeds=args.eval_seeds,
                    episodes=args.episodes, action_sampling=True, device=str(device),
                    benchmark_repeats=args.benchmark_repeats, torch_version=torch.__version__)
    protocol['device_name'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, curves = [], {}
    reference_audit = None
    reference_config = None
    reference_pretraining = None
    for seed in args.train_seeds:
        for variant in args.variants:
            run = args.runs_root / f'RAD-darkroom-{variant}-split0-train{seed}'
            checkpoint_path = run / f'ckpt-{args.checkpoint_step}.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            config = dict(checkpoint['config'])
            if (config.get('compressor_comparison') != PROTOCOL or config.get('compressor_type') != variant
                    or config.get('seed') != seed or config.get('env_split_seed') != 0
                    or config.get('env') != 'darkroom' or config.get('grid_size') != 9 or config.get('horizon') != 20
                    or checkpoint['step'] != args.checkpoint_step
                    or config['train_timesteps'] != args.checkpoint_step):
                raise ValueError(f'Checkpoint protocol mismatch: {checkpoint_path}')
            audit = config.get('dataset_audit')
            if not audit or len(audit['test_groups']) != 8:
                raise ValueError('Missing verified eight-goal Darkroom split')
            identity = {key: audit[key] for key in ('data_sha256', 'train_groups',
                                                   'test_groups', 'group_goals', 'collection_env_split_seed')}
            if reference_audit is not None and identity != reference_audit:
                raise ValueError('Checkpoints were trained against different dataset identities')
            reference_audit = identity
            shared_keys = ('tf_n_embd', 'tf_n_layer', 'tf_n_head', 'tf_dim_feedforward', 'compress_n_layers',
                           'compress_n_heads', 'n_compress_tokens', 'n_transit', 'short_memory_keep',
                           'latent_update_mode', 'always_use_latent_prefix', 'max_gradient_rounds',
                           'train_batch_size', 'train_source_timesteps', 'train_n_stream', 'curriculum_schedule',
                           'ad_lr', 'compression_lr', 'latent_lr', 'torch_compile', 'num_workers',
                           'mixed_precision', 'min_context_length', 'max_context_length', 'rad_batching_strategy',
                           'tf_dropout', 'tf_attn_dropout', 'num_warmup_steps', 'stage_warmup_steps',
                           'min_lr_ratio', 'beta1', 'beta2', 'weight_decay', 'gradient_accumulation_steps')
            shared = {key: config.get(key) for key in shared_keys}
            if reference_config is not None and shared != reference_config:
                raise ValueError('Shared model/training settings differ across comparison checkpoints')
            reference_config = shared
            provenance = config.get('pretrain_provenance')
            if not provenance or provenance['step'] != provenance['settings']['pretrain_timesteps']:
                raise ValueError('Missing completed-pretraining provenance')
            if reference_pretraining is not None and provenance['settings'] != reference_pretraining:
                raise ValueError('Pretraining settings/budgets differ across checkpoints')
            reference_pretraining = provenance['settings']
            config.update(device=device, torch_compile=False)
            model = RAD(config).to(device).eval()
            model.load_state_dict(normalize_compiled_state_dict(checkpoint['model']), strict=True)
            model.set_curriculum(None)
            _, goals = SAMPLE_ENVIRONMENT['darkroom'](config)
            for mode in (['mean', 'sample'] if variant == 'vae' else ['deterministic']):
                model.vae_eval_mode = mode if variant == 'vae' else 'mean'
                rewards, compressions = [], []
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                for eval_seed in args.eval_seeds:
                    model.reset_latent_rng(eval_seed)
                    envs = DummyVecEnv([make_env(config, goal=goal) for goal in goals])
                    try:
                        envs.seed(eval_seed)
                        result = model.evaluate_in_context(envs, config['horizon'] * args.episodes,
                                                           action_seed=eval_seed)
                    finally:
                        envs.close()
                    rewards.append(result['reward_episode'])
                    compressions.append(result['total_compressions'])
                rewards = np.stack(rewards)
                if rewards.shape != (len(args.eval_seeds), 8, args.episodes):
                    raise ValueError(f'Unexpected reward shape: {rewards.shape}')
                np.savez_compressed(output / f'{variant}-train{seed}-{mode}.npz', rewards=rewards,
                                    goals=np.asarray(goals), eval_seeds=args.eval_seeds,
                                    compressions=compressions, checkpoint=str(checkpoint_path))
                row = dict(variant=variant, latent_mode=mode, train_seed=seed, **metrics(rewards),
                           parameter_count=sum(p.numel() for p in model.parameters()),
                           compression_ms=benchmark_compression(model, device, 8, args.benchmark_repeats),
                           peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
                train_metrics_path = run / 'train-metrics.json'
                pretrain_run = args.runs_root / f'RAD-pretrain-darkroom-{variant}-split0-train{seed}'
                pretrain_metrics_path = pretrain_run / 'pretrain-metrics.json'
                train_metrics = json.loads(train_metrics_path.read_text()) if train_metrics_path.exists() else {}
                pretrain_metrics = json.loads(pretrain_metrics_path.read_text()) if pretrain_metrics_path.exists() else {}
                row.update(pretrain_reconstruction_mse=pretrain_metrics.get('loss_recon'),
                           train_kl=train_metrics.get('loss_kl'),
                           train_posterior_variance=train_metrics.get('posterior_variance'),
                           train_codebook_usage=train_metrics.get('codebook_usage'),
                           train_codebook_perplexity=train_metrics.get('codebook_perplexity'),
                           peak_training_gpu_bytes=train_metrics.get('peak_gpu_bytes'))
                rows.append(row)
                curves[(variant, mode, seed)] = rewards.mean(axis=(0, 1))
                print(json.dumps(row), flush=True)
            del model
    write_csv(output / 'per_training_seed.csv', rows)
    write_csv(output / 'summary.csv', aggregate(rows))
    paired = []
    for row in rows:
        baseline = next((r for r in rows if r['variant'] == 'ae' and r['train_seed'] == row['train_seed']), None)
        if baseline and row['variant'] != 'ae':
            paired.append(dict(variant=row['variant'], latent_mode=row['latent_mode'], train_seed=row['train_seed'],
                               **{key: row[key] - baseline[key] for key in metrics(np.zeros((1, 1, args.episodes)))}))
    write_csv(output / 'paired_vs_ae.csv', paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    curve_rows = []
    for variant, mode in sorted({key[:2] for key in curves}):
        values = np.stack([value for key, value in curves.items() if key[:2] == (variant, mode)])
        mean = values.mean(0)
        sem = values.std(0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
        episodes = np.arange(1, len(mean) + 1)
        label = f'{variant} ({mode})' if variant == 'vae' else variant
        line, = ax.plot(episodes, mean, label=label, linestyle='--' if mode == 'sample' else '-')
        ax.fill_between(episodes, mean - sem, mean + sem, color=line.get_color(), alpha=0.15)
        curve_rows.extend(dict(variant=variant, latent_mode=mode, episode=int(e), mean=float(m), sem=float(s))
                          for e, m, s in zip(episodes, mean, sem))
    ax.set(xlabel='Episode', ylabel='Episode return', ylim=(0, 20), title='Darkroom compressor comparison')
    if args.episodes > 50:
        ax.axvline(50, color='gray', linestyle=':', label='1,000 environment steps')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'adaptation.png', dpi=180)
    fig.savefig(output / 'adaptation.pdf')
    plt.close(fig)
    write_csv(output / 'adaptation.csv', curve_rows)


if __name__ == '__main__':
    main()
