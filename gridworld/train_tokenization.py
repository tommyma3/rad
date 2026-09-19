"""Train isolated AD_DPT/RAD_DPT variants; legacy entrypoints are unchanged.

Run from gridworld: accelerate launch train_tokenization.py --config rad_dpt_dr
Compression pretraining: add --phase pretrain. Fine-tuning accepts an explicit
--pretrain-ckpt; it never searches legacy run directories for initialization.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from model.transition_ad import TOKENIZED_MODELS, validate_config
from optimizer_utils import build_rad_optimizer_param_groups
from transition_dataset import TransitionDataset, EndpointBatchSampler, exact_length_collate, curriculum_stage
from utils import checkpoint_state_dict, get_config, get_curriculum_aware_scheduler, maybe_compile_model, normalize_compiled_state_dict


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def load_pretraining(model, checkpoint):
    source = checkpoint['config']
    validate_config(source)
    if source['model'] != 'RAD_DPT' or checkpoint.get('phase') != 'pretrain':
        raise ValueError('Expected a RAD_DPT compression-pretraining checkpoint')
    for key in ('env', 'grid_size', 'num_actions', 'dim_states', 'tf_n_embd', 'n_compress_tokens',
                'compress_n_heads', 'compress_n_layers', 'tf_dim_feedforward'):
        if source.get(key) != model.config.get(key):
            raise ValueError(f'Pretraining configuration mismatch: {key}')
    state = normalize_compiled_state_dict(checkpoint['model'])
    for name in ('embed_context', 'compression_transformer', 'reconstruction_decoder'):
        prefix = name + '.'
        getattr(model, name).load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                             if key.startswith(prefix)}, strict=True)


def load_experiment_config(preset=None, env=None):
    """Select environment defaults before overlaying the isolated model preset."""
    preset = preset or ('rad_dpt_dktd' if env == 'dktd' else 'rad_dpt_dr')
    path = Path(preset) if Path(preset).suffix in ('.yaml', '.yml') else Path(f'config/model/{preset}.yaml')
    model_config = get_config(path)
    if env is not None and model_config.get('env', env) != env:
        raise ValueError('--env disagrees with the model preset environment')
    env = env or model_config.get('env', 'darkroom')
    if env not in ('darkroom', 'dktd'):
        raise ValueError(f'Unsupported tokenization-ablation environment: {env}')
    config = get_config(f'config/env/{env}.yaml')
    config.update(get_config(f'config/algorithm/ppo_{env}.yaml'))
    config.update(model_config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Preset name or YAML path; defaults to the environment RAD_DPT preset')
    parser.add_argument('--env', choices=['darkroom', 'dktd'], help='Defaults to the preset environment, or Darkroom')
    parser.add_argument('--phase', choices=['train', 'pretrain'])
    parser.add_argument('--traj-dir', help='Source histories; resume defaults to the saved directory')
    parser.add_argument('--runs-root', default='./runs/tokenization')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--pretrain-ckpt', type=Path)
    parser.add_argument('--updates', type=int, help='Override update budget for a fresh run')
    parser.add_argument('--mixed-precision', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--cpu', action='store_true', help='Run a small validation job on CPU')
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        config = dict(checkpoint['config'])
        phase = checkpoint['phase']
        if args.phase not in (None, phase) or args.seed not in (None, config['seed']):
            raise ValueError('Resume cannot change phase or seed')
        if args.env not in (None, config['env']):
            raise ValueError('Resume cannot change environment')
        if args.pretrain_ckpt or args.updates is not None:
            raise ValueError('Resume restores the saved update budget and initialization')
    else:
        config = load_experiment_config(args.config, args.env)
        config['seed'] = args.seed if args.seed is not None else config.get('seed', 42)
        phase = args.phase or 'train'
    validate_config(config)
    method = config['model']
    if method not in TOKENIZED_MODELS or (phase == 'pretrain' and method != 'RAD_DPT'):
        raise ValueError('Only AD_DPT/RAD_DPT training and RAD_DPT pretraining are supported')
    if args.pretrain_ckpt and method != 'RAD_DPT':
        raise ValueError('Only RAD_DPT accepts compression pretraining')
    pretrain = phase == 'pretrain'
    if pretrain and args.pretrain_ckpt:
        raise ValueError('--pretrain-ckpt initializes policy training; use --resume to continue pretraining')
    settings = {**config, **config.get('pretrain', {})} if pretrain else config
    total = int(settings['pretrain_timesteps'] if pretrain else config['train_timesteps'])
    if args.updates is not None:
        total = args.updates
    if checkpoint:
        total = checkpoint['total_updates']
    if total < 1:
        raise ValueError('Update budget must be positive')
    precision = args.mixed_precision or config.get('mixed_precision', 'fp16')
    if checkpoint and precision != config['mixed_precision']:
        raise ValueError('Resume cannot change mixed precision')
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=precision,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=method == 'RAD_DPT')])
    traj_dir = args.traj_dir or config.get('traj_dir', './datasets')
    config.update(device=accelerator.device, mixed_precision=precision, traj_dir=str(Path(traj_dir).resolve()),
                  effective_total_updates=total)
    set_seed(config['seed'])
    run_dir = args.resume.parent if checkpoint else Path(args.runs_root) / f"{method}{'-pretrain' if pretrain else ''}-{config['env']}-seed{config['seed']}"
    if not checkpoint and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} is nonempty; use --resume or another --runs-root')
    dataset = TransitionDataset(config, traj_dir, 'train', settings['train_n_stream'], config['train_source_timesteps'])
    test_dataset = TransitionDataset(config, traj_dir, 'test', 1, config['train_source_timesteps'])
    # Legacy readers seed Python's global RNG. Restore the requested model seed.
    set_seed(config['seed'])
    model = TOKENIZED_MODELS[method](config)
    if method == 'RAD_DPT':
        model.configure_phase(pretrain)
    if args.pretrain_ckpt:
        load_pretraining(model, torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False))
        config['pretrain_checkpoint'] = str(args.pretrain_ckpt.resolve())
    config['initialization'] = config.get('initialization', 'pretrained_compression' if args.pretrain_ckpt else 'scratch')
    if pretrain:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        lr = settings['pretrain_lr']
    elif method == 'RAD_DPT':
        parameters = build_rad_optimizer_param_groups(model, config)
        lr = config['lr']
    else:
        parameters, lr = model.parameters(), config['lr']
    optimizer = AdamW(parameters, lr=lr, betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    if not pretrain and method == 'RAD_DPT' and config.get('curriculum_schedule'):
        curriculum = [(stage['step'], stage['max_compressions'], stage['length_distribution']) for stage in config['curriculum_schedule']]
        scheduler = get_curriculum_aware_scheduler(optimizer, curriculum, total, config['num_warmup_steps'],
                                                  config.get('stage_warmup_steps', 1000), config.get('min_lr_ratio', .1))
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, settings['pretrain_warmup_steps'] if pretrain else config['num_warmup_steps'], total)
    step = 0
    if checkpoint:
        model.load_state_dict(normalize_compiled_state_dict(checkpoint['model']), strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        step = checkpoint['step']
        if len(checkpoint['rng']) != accelerator.num_processes:
            raise ValueError('Exact resume requires the same distributed world size')
    compile_config = {**config, 'torch_compile_modules': settings.get('torch_compile_modules', ['ad_transformer'])}
    model = maybe_compile_model(model, compile_config, accelerator.is_main_process,
                                default_modules=['ad_transformer'])
    model, optimizer = accelerator.prepare(model, optimizer)
    if checkpoint and checkpoint.get('scaler') is not None:
        accelerator.scaler.load_state_dict(checkpoint['scaler'])
    accumulation = int(config.get('gradient_accumulation_steps', 1))
    batch_size = int(settings['pretrain_batch_size'] if pretrain else config['train_batch_size']) * accumulation
    sampler = EndpointBatchSampler(dataset, config, batch_size, step, total, accelerator.process_index, pretrain)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=exact_length_collate,
                        num_workers=settings.get('num_workers', 0),
                        persistent_workers=settings.get('num_workers', 0) > 0,
                        generator=torch.Generator().manual_seed(config['seed']))
    writer = None
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.yaml').write_text(yaml.safe_dump({**config, 'device': str(config['device'])}), encoding='utf-8')
        writer = SummaryWriter(str(run_dir))
    accelerator.wait_for_everyone()
    if checkpoint:
        restore_rng(checkpoint['rng'][accelerator.process_index])

    def save():
        states = gather_object([rng_state()])
        if accelerator.is_main_process:
            target = run_dir / f'ckpt-{step}.pt'
            temporary = target.with_suffix('.pt.tmp')
            torch.save({'config': config, 'phase': phase, 'step': step, 'total_updates': total,
                        'model': checkpoint_state_dict(accelerator.unwrap_model(model)),
                        'optimizer': optimizer.state_dict(), 'lr_sched': scheduler.state_dict(), 'rng': states,
                        'scaler': accelerator.scaler.state_dict() if accelerator.scaler else None}, temporary)
            temporary.replace(target)
        accelerator.wait_for_everyone()

    try:
        model.train()
        for microbatches in loader:
            raw_model = accelerator.unwrap_model(model)
            if method == 'RAD_DPT' and not pretrain:
                raw_model.max_compressions = curriculum_stage(config, step)['max_compressions']
            count = sum(len(batch['states']) for batch in microbatches)
            retry_rng = rng_state()
            for attempt in range(16):
                optimizer.zero_grad(set_to_none=True)
                metrics = {}
                for index, batch in enumerate(microbatches):
                    weight = len(batch['states']) / count
                    sync = nullcontext() if index == len(microbatches) - 1 else accelerator.no_sync(model)
                    with sync, accelerator.autocast():
                        output = model(batch, pretrain=pretrain)
                        loss = output['loss_recon' if pretrain else 'loss_action'] * weight
                        accelerator.backward(loss)
                    for key, value in output.items():
                        metrics[key] = metrics.get(key, 0.) + value.detach() * weight
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    break
                # Retry the same data/dropout draws with the reduced AMP scale;
                # an overflow must not advance the curriculum or sampler index.
                restore_rng(retry_rng)
            else:
                raise FloatingPointError('AMP overflow persisted for 16 attempts at one update')
            scheduler.step()
            step += 1
            if step % config.get('progress_interval', 50) == 0 and accelerator.is_main_process:
                print(f'{method} {phase} {step}/{total}: ' + ', '.join(f'{key}={value.item():.4f}' for key, value in metrics.items()), flush=True)
            if step % config['summary_interval'] == 0:
                for key, value in metrics.items():
                    value = accelerator.reduce(value, reduction='mean')
                    if writer:
                        writer.add_scalar(f'train/{key}', value.item(), step)
                if writer:
                    for group in optimizer.param_groups:
                        writer.add_scalar(f"lr/{group.get('group_name', 'all')}", group['lr'], step)
                    lengths = np.concatenate([np.full(len(batch['states']), batch['states'].shape[1]) for batch in microbatches])
                    writer.add_histogram('train/history_lengths', lengths, step)
                    if method == 'RAD_DPT' and not pretrain:
                        memory_states = np.asarray([raw_model.schedule.state_after(int(length)) for length in lengths])
                        writer.add_histogram('train/compression_counts', memory_states[:, 0], step)
                        writer.add_histogram('train/recent_lengths', memory_states[:, 1], step)
            if step % config['eval_interval'] == 0:
                saved_rng = rng_state()
                model.eval()
                validation = EndpointBatchSampler(test_dataset, config, config['test_batch_size'], 0, 1,
                                                   accelerator.process_index, pretrain)
                # Fixed seed and fixed endpoint identities within each curriculum stage.
                stage_start = curriculum_stage(config, step - 1).get('step', 0) if not pretrain else 0
                batches = exact_length_collate([test_dataset[index] for index in validation.batch_at(stage_start)])
                val = 0.
                with torch.inference_mode(), accelerator.autocast():
                    for batch in batches:
                        result = model(batch, pretrain=pretrain)
                        val = val + result['loss_recon' if pretrain else 'loss_action'] * len(batch['states']) / config['test_batch_size']
                val = accelerator.reduce(val, reduction='mean')
                if writer:
                    writer.add_scalar('validation/loss', val.item(), step)
                model.train()
                restore_rng(saved_rng)
            if step % config['ckpt_interval'] == 0 or step == total:
                save()
    finally:
        if writer:
            writer.close()
        accelerator.end_training()


if __name__ == '__main__':
    main()
