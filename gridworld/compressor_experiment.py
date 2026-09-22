"""Shared contract for the Darkroom compressor comparison."""

from pathlib import Path
import hashlib
import json
import random


VARIANTS = ('ae', 'vae', 'vq_vae')
PROTOCOL = 'darkroom-compressor-legacy-v1'


def is_comparison(config):
    """Shared reproducibility safeguards, with separate protocol identities."""
    return bool(config.get('compressor_comparison') or config.get('memory_size_comparison'))


def model_config_path(name):
    return name if Path(name).suffix in ('.yaml', '.yml') else f'./config/model/{name}.yaml'


def write_run_metrics(model, config, output, elapsed_seconds, pretrain=False):
    import torch
    metrics = {key: value.detach().float().item() if torch.is_tensor(value) else value
               for key, value in output.items()}
    metrics.update(elapsed_seconds=elapsed_seconds,
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   compressor_parameter_count=sum(p.numel() for p in model.compression_transformer.parameters()),
                   amp_retries=config.get('amp_retries', 0),
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(config['device']) if config['device'].type == 'cuda' else 0)
    stage = 'pretrain' if pretrain else 'train'
    (Path(config['log_dir']) / f'{stage}-metrics.json').write_text(json.dumps(metrics, indent=2))


def comparison_optimizer_step(model, batch, optimizer, accelerator, config, **forward_kwargs):
    """Count actual updates; retry AMP overflow on the same batch and RNG state."""
    import torch
    unwrapped = accelerator.unwrap_model(model)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    latent_states = {key: generator.get_state() for key, generator in unwrapped._latent_generators.items()}
    for attempt in range(10):
        if attempt:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            for key in list(unwrapped._latent_generators):
                if key in latent_states:
                    unwrapped._latent_generators[key].set_state(latent_states[key])
                else:
                    del unwrapped._latent_generators[key]
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(batch, **forward_kwargs)
        loss = output['loss_total']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite comparison loss')
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not accelerator.optimizer_step_was_skipped:
            config['amp_retries'] = config.get('amp_retries', 0) + attempt
            return output
    raise FloatingPointError('AMP overflow persisted for 10 attempts on one comparison batch')


def validate_checkpoint_config(expected, actual, same_stage=False):
    """Never silently transplant a different bottleneck or experiment's state."""
    keys = ['compressor_type', 'tf_n_embd', 'compress_n_layers', 'compress_n_heads',
            'n_compress_tokens']
    if expected.get('compressor_type', 'ae') == 'vq_vae':
        keys.append('vq_codebook_size')
    if is_comparison(expected):
        keys += ['compressor_comparison', 'seed', 'env_split_seed', 'collection_env_split_seed',
                 'dataset_task_mapping', 'vae_kl_weight', 'vq_commitment_weight',
                 'train_n_stream', 'train_source_timesteps']
    if same_stage:
        keys += ['n_transit', 'always_use_latent_prefix', 'latent_update_mode']
        keys += ['first_recent_capacity', 'recurrent_recent_capacity']
    if expected.get('memory_size_comparison') or actual.get('memory_size_comparison'):
        keys += ['memory_size_comparison', 'always_use_latent_prefix', 'latent_update_mode',
                 'seed', 'env_split_seed', 'collection_env_split_seed', 'dataset_task_mapping',
                 'train_n_stream', 'train_source_timesteps']
        from memory_size_experiment import validate_memory_size_config
        validate_memory_size_config(actual, pretrain=not same_stage or 'pretrain_timesteps' in expected)
    for key in keys:
        default = 'ae' if key == 'compressor_type' else None
        if expected.get(key, default) != actual.get(key, default):
            raise ValueError(f'Checkpoint mismatch for {key}: expected {expected.get(key, default)!r}, '
                             f'got {actual.get(key, default)!r}')
    if is_comparison(expected) and 'dataset_audit' in expected:
        for key in ('data_sha256', 'train_groups', 'test_groups', 'group_goals'):
            if expected['dataset_audit'].get(key) != actual.get('dataset_audit', {}).get(key):
                raise ValueError(f'Checkpoint dataset mismatch for {key}')


def add_experiment_arguments(parser):
    parser.add_argument('--seed', type=int, help='Model/training seed; does not change the goal split')
    parser.add_argument('--runs_root', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--traj_dir', type=str)
    parser.add_argument('--steps', type=int, help='Update budget override (pilot only)')
    parser.add_argument('--batch_size', type=int, help='Per-process batch size override (pilot only)')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--no_compile', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='CPU smoke checks only')
    parser.add_argument('--n_latents', type=int, help='Latent tokens for the memory-size protocol only')


def apply_experiment_arguments(config, args, pretrain=False):
    if getattr(args, 'n_latents', None) is not None:
        if not config.get('memory_size_comparison'):
            raise ValueError('--n_latents requires a memory-size experiment config')
        config['n_compress_tokens'] = args.n_latents
    for key in ('seed', 'runs_root', 'run_name', 'traj_dir', 'num_workers'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError('--steps must be positive')
        config['pretrain_timesteps' if pretrain else 'train_timesteps'] = args.steps
        if config.get('memory_size_comparison') and not pretrain:
            # Short pilots must run online evaluation to produce best-model.pt.
            config['gen_interval'] = min(config.get('gen_interval', 10000), args.steps)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError('--batch_size must be positive')
        config['pretrain_batch_size' if pretrain else 'train_batch_size'] = args.batch_size
    if args.no_compile or args.cpu:
        config['torch_compile'] = False
    if args.cpu:
        config['mixed_precision'] = 'no'
    if config.get('memory_size_comparison'):
        from memory_size_experiment import validate_memory_size_config, run_name
        validate_memory_size_config(config, pretrain=pretrain)
        config.setdefault('run_name', run_name(config['n_compress_tokens'], config.get('seed', 0), pretrain))
    if is_comparison(config):
        if config['env'] != 'darkroom':
            raise ValueError('The compressor comparison protocol is Darkroom-only')
        config['data_seed'] = int(config.get('seed', 42))
        stage = 'RAD-pretrain' if pretrain else 'RAD'
        config.setdefault('run_name', f"{stage}-darkroom-{config['compressor_type']}-split{config['env_split_seed']}-train{config.get('seed', 42)}")
        if config.get('gradient_accumulation_steps', 1) != 1:
            raise ValueError('Comparison update budgets currently require gradient_accumulation_steps=1')


def make_data_generator(config):
    import torch
    if not is_comparison(config):
        return None
    return torch.Generator().manual_seed(int(config['data_seed']))


def seed_data_worker(worker_id):
    import torch
    info = torch.utils.data.get_worker_info()
    if hasattr(info.dataset, 'rng'):
        info.dataset.rng = random.Random(info.seed)


def select_dataset_groups(config, mode):
    if config.get('dataset_task_mapping') == 'collection_order':
        if 'collection_env_split_seed' not in config:
            raise ValueError('collection_order requires the original collection_env_split_seed')
        from baseline_dataset import selected_group_ids
        return selected_group_ids(config, mode)
    if config.get('dataset_task_mapping', 'legacy') != 'legacy':
        raise ValueError('Unknown dataset_task_mapping')
    count = config['grid_size'] ** {'darkroom': 2, 'dktd': 4}[config['env']]
    ids = list(range(count))
    random.seed(config['env_split_seed'])
    random.shuffle(ids)
    split = round(count * config['train_env_ratio'])
    if mode == 'train':
        return ids[:split]
    if mode == 'test':
        return ids[split:]
    if mode == 'all':
        return ids
    raise ValueError(f'Invalid dataset mode: {mode}')


def validate_darkroom_group(config, group_id, states, actions, rewards):
    """Check actual transitions against the proposed group-to-goal mapping."""
    import numpy as np
    from baseline_dataset import task_for_group, transition_next_states
    goal = task_for_group(config, int(group_id))
    next_states = transition_next_states(states, actions.astype(np.int64), config['grid_size'])
    expected = np.all(next_states == goal, axis=-1).astype(np.float32)
    if not np.allclose(rewards, expected, rtol=0, atol=1e-6):
        raise ValueError(f'Group {group_id}: rewards disagree with goal {goal.tolist()}; '
                         'verify collection_env_split_seed and source dataset')


def audit_darkroom_dataset(config, traj_dir):
    import h5py
    import numpy as np
    from baseline_dataset import task_for_group
    from utils import get_traj_file_name
    source = Path(traj_dir) / (get_traj_file_name(config) + '.hdf5')
    train = select_dataset_groups(config, 'train')
    test = select_dataset_groups(config, 'test')
    if set(train) & set(test):
        raise ValueError('Train/test source groups overlap')
    goals = {}
    digest = hashlib.sha256()
    with h5py.File(source, 'r') as history:
        for group_id in train + test:
            group = history[str(group_id)]  # Missing tasks must not silently disappear.
            limit = int(config['train_source_timesteps'])
            streams = int(config['train_n_stream'])
            if group['states'].shape[0] < limit or group['states'].shape[1] < streams:
                raise ValueError(f'Group {group_id} has insufficient source steps or streams')
            states = group['states'][:limit, :streams]
            actions = group['actions'][:limit, :streams]
            rewards = group['rewards'][:limit, :streams]
            validate_darkroom_group(config, group_id, states, actions, rewards)
            if not np.any(rewards > 0):
                raise ValueError(f'Group {group_id}: no positive reward to verify goal identity')
            goals[str(group_id)] = task_for_group(config, group_id).tolist()
            digest.update(str(group_id).encode())
            for values in (states, actions, rewards):
                digest.update(str((values.shape, values.dtype.str)).encode())
                digest.update(values.tobytes())
    stat = source.stat()
    return dict(source=str(source.resolve()), source_bytes=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns, train_groups=train, test_groups=test,
                group_goals=goals, data_sha256=digest.hexdigest(),
                collection_env_split_seed=config['collection_env_split_seed'])
