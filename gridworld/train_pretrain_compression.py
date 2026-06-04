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

from dataset import CompressionPretrainDataset
from model import MODEL
from utils import get_config, get_data_loader, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm


def pretrain_collate_fn(batch, grid_size):
    """Collate function for pre-training dataset (no query states/targets)."""
    import numpy as np
    import torch.nn.functional as F
    
    res = {}
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = F.one_hot(torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.long), num_classes=5)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    return res


def get_pretrain_data_loader(dataset, batch_size, config, shuffle=True):
    """Data loader for pre-training with custom collate function."""
    from torch.utils.data import DataLoader
    from functools import partial
    
    collate_fn = partial(pretrain_collate_fn, grid_size=config['grid_size'])
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        num_workers=config['num_workers'], 
        persistent_workers=True
    )


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='rad_pretrain',
                       help='Model config name (without .yaml extension)')
    parser.add_argument('--env', type=str, default='darkroom',
                       help='Environment name: darkroom or dark_key_to_door')
    args = parser.parse_args()
    
    # Determine config files based on environment
    env_config_map = {
        'darkroom': ('darkroom', 'ppo_darkroom'),
        'dark_key_to_door': ('dark_key_to_door', 'ppo_dark_key_to_door'),
    }
    if args.env not in env_config_map:
        raise ValueError(f'Unknown environment: {args.env}')
    env_cfg, alg_cfg = env_config_map[args.env]
    
    # Load configs
    config = get_config(f'./config/env/{env_cfg}.yaml')
    config.update(get_config(f'./config/algorithm/{alg_cfg}.yaml'))
    config.update(get_config(f'./config/model/{args.config}.yaml'))

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    log_dir = path.join('./runs', f"RAD-pretrain-{config['env']}-seed{config['env_split_seed']}")
    
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
    config['mixed_precision'] = 'fp16'

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
        config['train_source_timesteps']
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

    # Optimizer - only for compression-related parameters
    compression_params = list(model.compression_transformer.parameters()) + \
                        list(model.reconstruction_decoder.parameters()) + \
                        list(model.embed_context.parameters())
    
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
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        if is_main:
            print(f'Checkpoint loaded from {ckpt_path}')

    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )

    if is_main:
        start_time = datetime.now()
        print(f'Pre-training started at {start_time}')

    # Get unwrapped model for calling custom methods (DDP wrapping hides them)
    unwrapped_model = accelerator.unwrap_model(model)

    # Training loop
    with tqdm(total=config['pretrain_timesteps'], position=0, leave=True, disable=not is_main) as pbar:
        pbar.update(step)

        while step < config['pretrain_timesteps']:
            batch = next(train_dataloader)
            
            step += 1
            
            with accelerator.autocast():
                output = unwrapped_model.forward_pretrain_compression(batch)
            
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
                    'step': step,
                    'config': config,
                    'model': unwrapped_model.state_dict(),
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
            'step': step,
            'config': config,
            'model': unwrapped_model.state_dict(),
        }, final_path)
        print(f'\nFinal model saved to {final_path}')

        writer.flush()
        
        end_time = datetime.now()
        print(f'\nPre-training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
