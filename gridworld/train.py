"""
Training script for Algorithm Distillation (AD).

This script trains the original AD model with decoder-only transformer.
Supports multi-GPU training via Hugging Face Accelerate.

Usage:
    accelerate launch train.py
    
For multi-GPU:
    accelerate launch --multi_gpu --num_processes=N train.py
"""

from datetime import datetime
import os
import os.path as path
from modulefinder import ModuleFinder
from glob import glob
import shutil

from accelerate import Accelerator
from accelerate.utils import set_seed

import argparse
import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset
from env import SAMPLE_ENVIRONMENT
from model import MODEL
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import make_env

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='ad_dr',
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
    
    config = get_config(f'./config/env/{env_cfg}.yaml')
    config.update(get_config(f'./config/algorithm/{alg_cfg}.yaml'))
    config.update(get_config(f'./config/model/{args.config}.yaml'))

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    log_dir = path.join('./runs', f"{config['model']}-{config['env']}-seed{config['env_split_seed']}")
    
    config['log_dir'] = log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'fp16'  # 'no' for fp32, or 'fp16'/'bf16' for mixed precision

    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
    )
    config['device'] = accelerator.device
    
    # Only main process handles logging and checkpointing
    is_main = accelerator.is_main_process
    
    if is_main:
        try:
            # Try to open config file to bypass NFS cache
            with open(config_save_path, 'r') as f:
                f.read(1)
                config_exists = True
        except FileNotFoundError:
            config_exists = False

        if config_exists:
            print(f'WARNING: {log_dir} already exists. Skipping...')
            exit(0)
        
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, flush_secs=15)
        print(f'Using Device: {config["device"]}')
        print(f'Number of processes: {accelerator.num_processes}')

    model_name = config['model']
    model = MODEL[model_name](config)

    if is_main:
        load_start_time = datetime.now()
        print(f'Data loading started at {load_start_time}')

    train_dataset = ADDataset(config, config['traj_dir'], 'train', config['train_n_stream'], config['train_source_timesteps'])
    test_dataset = ADDataset(config, config['traj_dir'], 'test', 1, config['train_source_timesteps'])

    train_dataloader = get_data_loader(train_dataset, batch_size=config['train_batch_size'], config=config, shuffle=True)
    train_dataloader = next_dataloader(train_dataloader)

    test_dataloader = get_data_loader(test_dataset, batch_size=config['test_batch_size'], config=config, shuffle=False)
    
    if is_main:
        load_end_time = datetime.now()
        print()
        print(f'Data loading ended at {load_end_time}')
        print(f'Elapsed time: {load_end_time - load_start_time}')

    optimizer = AdamW(model.parameters(), lr = config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    lr_sched = get_cosine_schedule_with_warmup(optimizer, config['num_warmup_steps'], config['train_timesteps'])
    step = 0

    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=config['device'])
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        if is_main:
            print(f'Checkpoint loaded from {ckpt_path}')

    env_name = config['env']
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    train_env_args = train_env_args[:10]
    test_env_args = test_env_args[:10]
    env_args = train_env_args + test_env_args    

    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
    elif env_name == "dark_key_to_door":
        envs = SubprocVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in env_args])
    else:
        raise NotImplementedError(f'Environment not supported: {env_name}')
    
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(model, optimizer, train_dataloader, lr_sched)

    if is_main:
        start_time = datetime.now()
        print(f'Training started at {start_time}')

    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not is_main) as pbar:
        pbar.update(step)

        while True:
            batch = next(train_dataloader)
            
            step += 1
            
            with accelerator.autocast():
                output = model(batch)
            
            loss = output['loss_action']

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(loss=loss.item())

            if is_main and step % config['summary_interval'] == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'], step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
                writer.add_scalar('train/acc_action', output['acc_action'].item(), step)


            # Eval
            if is_main and step % config['eval_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f'Evaluating started at {eval_start_time}')

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_loss_reward = 0.0
                    test_acc_reward = 0.0
                    test_loss_next_state = 0.0
                    test_acc_next_state = 0.0
                    test_cnt = 0

                    for j, batch in enumerate(test_dataloader):
                        output = model(batch)
                        cnt = len(batch['states'])
                        test_loss_action += output['loss_action'].item() * cnt
                        test_acc_action += output['acc_action'].item() * cnt
                            
                        if config['dynamics']:
                            test_loss_reward += output['loss_reward'].item() * cnt
                            test_acc_reward += output['acc_reward'].item() * cnt
                            test_loss_next_state += output['loss_next_state'].item() * cnt
                            test_acc_next_state += output['acc_next_state'].item() * cnt
                            
                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)
                writer.add_scalar('test/acc_action', test_acc_action / test_cnt, step)              

                eval_end_time = datetime.now()
                print()
                print(f'Evaluating ended at {eval_end_time}')
                print(f'Elapsed time: {eval_end_time - eval_start_time}')
                model.train()
                torch.cuda.empty_cache()

            pbar.update(1)

            # LOGGING
            if is_main and step % config['ckpt_interval'] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for ckpt_path in ckpt_paths:
                    os.remove(ckpt_path)

                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')

                # Get unwrapped model state dict for saving
                unwrapped_model = accelerator.unwrap_model(model)

                torch.save({
                    'step': step,
                    'config': config,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')

            if step >= config['train_timesteps']:
                break

    if is_main:
        writer.flush()
    
    envs.close()

    if is_main:
        end_time = datetime.now()
        print()
        print(f'Training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')