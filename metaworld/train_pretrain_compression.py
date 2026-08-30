"""
Pre-training script for Compression Transformer.

This script trains the compression transformer using reconstruction loss
before fine-tuning the full RAD system.

Usage:
    accelerate launch train_pretrain_compression.py
    
For multi-GPU:
    accelerate launch --multi_gpu --num_processes=N train_pretrain_compression.py
"""

from datetime import datetime
import os
import os.path as path
from glob import glob
import argparse

from accelerate import Accelerator
from accelerate.utils import set_seed

import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import CompressionBucketBatchSampler, CompressionPretrainDataset
from model import MODEL
from optimizer_utils import build_compression_pretraining_parameters
from utils import (
    checkpoint_state_dict,
    configure_torch_runtime,
    get_config,
    get_data_loader,
    maybe_compile_model,
    next_dataloader,
    normalize_compiled_state_dict,
)
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm

CHECKPOINT_FORMAT = 'metaworld-sar-v1'
COMPRESSION_PRETRAIN_CONTRACT = 'recurrent-transition-v1'


def apply_overrides(config, override):
    for option in override.split('|'):
        if not option:
            continue
        address, value = option.split('=', 1)
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)


def pretrain_collate_fn(batch):
    """Collate function for pre-training dataset (no query states/targets)."""
    import numpy as np
    
    res = {}
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    return res


def get_pretrain_data_loader(dataset, batch_size, config, shuffle=True):
    """Data loader grouped by exact recurrent compression count."""
    from torch.utils.data import DataLoader
    sampler = CompressionBucketBatchSampler(dataset, batch_size, shuffle=shuffle, drop_last=False)
    num_workers = config.get('pretrain_num_workers', config['num_workers'])
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=pretrain_collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='rad_ml1',
                       help='Model config name (without .yaml extension)')
    parser.add_argument('--override', '-o', default='',
                       help='Override config entries, e.g. "task=push-v3|train_source_timesteps=10000"')
    args = parser.parse_args()
    
    config = get_config('./config/env/ml1.yaml')
    config.update(get_config('./config/algorithm/ppo_ml1.yaml'))
    config.update(get_config(f'./config/model/{args.config}.yaml'))
    apply_overrides(config, args.override)

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    runs_root = config.get('runs_root', './runs')
    default_run_name = f"RAD-pretrain-ml1-{config['task']}"
    run_name = config.get('pretrain_run_name', default_run_name)
    log_dir = path.join(runs_root, run_name)
    
    # Check if already exists
    config_save_path = path.join(log_dir, 'config.yaml')
    try:
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists:
        print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)
    
    config['log_dir'] = log_dir
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = config.get('mixed_precision', 'bf16')
    configure_torch_runtime(config)

    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
    )
    
    config['device'] = accelerator.device
    
    # Only main process prints and logs
    is_main = accelerator.is_main_process
    
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, flush_secs=15)
        print(f'Using Device: {config["device"]}')
        print(f'Number of processes: {accelerator.num_processes}')

    # Create model
    model = MODEL[config['model']](config)

    if is_main:
        load_start_time = datetime.now()
        print(f'Data loading started at {load_start_time}')

    # Create pre-training dataset
    train_dataset = CompressionPretrainDataset(
        config, 
        config['traj_dir'], 
        'train', 
        config['train_n_stream'], 
        config['train_source_timesteps'],
        n_seed=config.get('train_n_seed'),
    )
    
    train_dataloader = get_pretrain_data_loader(
        train_dataset, 
        batch_size=config['pretrain_batch_size'], 
        config=config, 
        shuffle=True
    )
    train_dataloader = next_dataloader(train_dataloader)

    if is_main:
        load_end_time = datetime.now()
        print(f'Data loading ended at {load_end_time}')
        print(f'Elapsed time: {load_end_time - load_start_time}')

    compression_params = build_compression_pretraining_parameters(model)
    
    optimizer = AdamW(
        compression_params, 
        lr=config['pretrain_lr'], 
        betas=(config['beta1'], config['beta2']), 
        weight_decay=config['weight_decay']
    )
    
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer, 
        config['pretrain_warmup_steps'], 
        config['pretrain_timesteps']
    )
    
    step = 0

    # Load checkpoint if exists
    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'pretrain-ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=config['device'])
        if ckpt.get('format') != CHECKPOINT_FORMAT:
            raise ValueError(
                f'{ckpt_path} uses the legacy packed-transition checkpoint format; '
                'start a new pretraining run'
            )
        if ckpt.get('compression_pretrain_contract') != COMPRESSION_PRETRAIN_CONTRACT:
            raise ValueError(f'{ckpt_path} does not use {COMPRESSION_PRETRAIN_CONTRACT}; start a new pretraining run')
        load_result = model.load_state_dict(normalize_compiled_state_dict(ckpt['model']), strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        if is_main:
            print(f'Checkpoint loaded from {ckpt_path}')
            if load_result.missing_keys:
                print(f'Missing model keys initialized from current config: {load_result.missing_keys}')
            if load_result.unexpected_keys:
                print(f'Unexpected model keys ignored: {load_result.unexpected_keys}')

    pretrain_compile_config = dict(config)
    pretrain_compile_config['torch_compile_modules'] = config.get(
        'pretrain_torch_compile_modules',
        ['compression_transformer', 'reconstruction_decoder'],
    )
    model = maybe_compile_model(
        model,
        pretrain_compile_config,
        is_main,
        default_modules=['compression_transformer', 'reconstruction_decoder'],
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )

    if is_main:
        start_time = datetime.now()
        print(f'Pre-training started at {start_time}')

    # Training loop
    with tqdm(total=config['pretrain_timesteps'], position=0, leave=True, disable=not is_main) as pbar:
        pbar.update(step)

        while step < config['pretrain_timesteps']:
            batch = next(train_dataloader)
            
            step += 1
            
            with accelerator.autocast():
                output = model(batch, pretrain_compression=True)
            
            loss = output['loss_recon']

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(loss_recon=loss.item())

            # Logging
            if is_main and step % config['summary_interval'] == 0:
                writer.add_scalar('pretrain/loss_recon', loss.item(), step)
                writer.add_scalar('pretrain/lr', lr_sched.get_last_lr()[0], step)

            # Save checkpoint
            if is_main and step % config['ckpt_interval'] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'pretrain-ckpt-*.pt')))
                for old_ckpt_path in ckpt_paths:
                    os.remove(old_ckpt_path)

                new_ckpt_path = path.join(config['log_dir'], f'pretrain-ckpt-{step}.pt')
                
                # Get unwrapped model state dict
                unwrapped_model = accelerator.unwrap_model(model)
                
                torch.save({
                    'format': CHECKPOINT_FORMAT,
                    'compression_pretrain_contract': COMPRESSION_PRETRAIN_CONTRACT,
                    'step': step,
                    'config': config,
                    'model': checkpoint_state_dict(unwrapped_model),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')

            pbar.update(1)

    # Save final model
    if is_main:
        final_path = path.join(config['log_dir'], 'pretrain-final.pt')
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'format': CHECKPOINT_FORMAT,
            'compression_pretrain_contract': COMPRESSION_PRETRAIN_CONTRACT,
            'step': step,
            'config': config,
            'model': checkpoint_state_dict(unwrapped_model),
        }, final_path)
        print(f'\nFinal model saved to {final_path}')

        writer.flush()
        
        end_time = datetime.now()
        print(f'\nPre-training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
