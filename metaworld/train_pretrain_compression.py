"""
Pre-training script for Compression Transformer on Meta-world.

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
from modulefinder import ModuleFinder
from glob import glob
import shutil
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


def pretrain_collate_fn(batch, dim_actions):
    """Collate function for pre-training dataset (no query states/targets)."""
    import numpy as np
    
    res = {}
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.float)  # Continuous actions
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    return res


def get_pretrain_data_loader(dataset, batch_size, config, shuffle=True):
    """Data loader for pre-training with custom collate function."""
    from torch.utils.data import DataLoader
    from functools import partial
    
    collate_fn = partial(pretrain_collate_fn, dim_actions=config['dim_actions'])
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        num_workers=config['num_workers'], 
        persistent_workers=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=False, default='./config/algorithm/ppo_ml1.yaml', help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=False, default='./config/env/ml1.yaml', help="Environment config")
    parser.add_argument('--model-config', '-mc', required=False, default='./config/model/rad_pretrain_ml1.yaml', help="Model config")
    parser.add_argument('--log-dir', '-l', required=False, default='./runs', help="Log directory")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Trajectory directory")
    parser.add_argument('--no-backup', '-nb', required=False, default=False, help="Save code", action='store_true')
    parser.add_argument('--override', '-o', default='')
    parser.add_argument('--resume', required=False, default=False, help="Resume train", action='store_true')
    parser.add_argument('--mixed-precision', '-m', required=False, default='bf16')
    parser.add_argument('--disable-tqdm', '-d', required=False, default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))
    config.update(get_config(args.model_config))

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    log_dir = path.join(args.log_dir, f"RAD-pretrain-ml1-{config['task']}")
    
    config['log_dir'] = log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    
    traj_dir = path.join(args.traj_dir, config['task'])
    config['traj_dir'] = traj_dir
    config['mixed_precision'] = args.mixed_precision

    # Initialize accelerator for multi-GPU support
    if args.mixed_precision == 'bf16' or args.mixed_precision == 'fp16':
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
    elif args.mixed_precision == 'fp32':
        accelerator = Accelerator(mixed_precision='no')
    else:
        raise ValueError(f'Unsupported mixed precision: {args.mixed_precision}')

    config['device'] = accelerator.device
    
    # Only main process handles logging and checkpointing
    is_main = accelerator.is_main_process

    # Prevent overwriting
    try:
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)
    
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, flush_secs=15)
        print(f'Using Device: {config["device"]}')
        print(f'Number of processes: {accelerator.num_processes}')

        # Save config
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        print(f'Config saved to {config_save_path}')

        # Save code
        if not args.no_backup:
            code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
            mf = ModuleFinder([os.getcwd()])
            mf.run_script(__file__)
            for name, module in mf.modules.items():
                if module.__file__ is None:
                    continue
                rel_path = path.relpath(module.__file__)
                new_path = path.join(code_dir, rel_path)
                new_dirname = path.dirname(new_path)
                os.makedirs(new_dirname, mode=0o750, exist_ok=True)
                shutil.copy2(rel_path, new_path)
            print(f'Code saved to {code_dir}')

    # Create model
    model = MODEL[config['model']](config)

    if is_main:
        load_start_time = datetime.now()
        print(f'Data loading started at {load_start_time}')

    # Create pre-training dataset
    train_dataset = CompressionPretrainDataset(
        config, 
        traj_dir, 
        'train', 
        config['train_n_seed'],
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

    # Resume checkpoint
    if args.resume:
        ckpt_paths = sorted(glob(path.join(config['log_dir'], 'pretrain-ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
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

    # Get unwrapped model to call custom methods (DDP hides them)
    unwrapped_model = accelerator.unwrap_model(model)

    # Training loop
    with tqdm(total=config['pretrain_timesteps'], position=0, leave=True, disable=not is_main or args.disable_tqdm) as pbar:
        pbar.update(step)

        while step < config['pretrain_timesteps']:
            with accelerator.autocast():
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
