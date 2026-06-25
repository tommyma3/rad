"""
Fine-tuning script for Recurrent Algorithm Distillation (RAD).

This script fine-tunes the full RAD system (compression + AD) after pre-training
the compression transformer. Supports multi-GPU training and curriculum learning.

Usage:
    accelerate launch train_rad.py
    
For multi-GPU:
    accelerate launch --multi_gpu --num_processes=N train_rad.py
    
With config:
    accelerate config  # First time setup
    accelerate launch train_rad.py
"""

from datetime import datetime
import os
import os.path as path
from glob import glob
import argparse
import gc

import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import RADDataset, ADDataset
from env import SAMPLE_ENVIRONMENT
from model import MODEL
from utils import get_config, next_dataloader, get_curriculum_aware_scheduler
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv

from env import make_env
import numpy as np
import torch.nn.functional as F
from functools import partial


def configure_torch_runtime(config):
    matmul_precision = config.get('float32_matmul_precision', 'high')
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
        config['float32_matmul_precision'] = matmul_precision

    if config.get('torch_compile', False) and config.get('torch_compile_cudagraph_skip_dynamic_graphs', True):
        try:
            import importlib
            inductor_config = importlib.import_module('torch._inductor.config')
            inductor_config.triton.cudagraph_skip_dynamic_graphs = True
            config['torch_compile_cudagraph_skip_dynamic_graphs'] = True
        except Exception:
            pass


def rad_collate_fn(batch, grid_size, num_actions=5):
    """
    Collate function for variable-length RAD dataset.
    Handles sequences of different lengths by padding.
    """
    # Find max context length in batch
    max_context_len = max(item['states'].shape[0] for item in batch)
    
    batch_size = len(batch)
    dim_state = batch[0]['states'].shape[1]
    
    # Initialize padded arrays
    states = np.zeros((batch_size, max_context_len, dim_state), dtype=np.float32)
    actions = np.zeros((batch_size, max_context_len), dtype=np.int64)
    rewards = np.zeros((batch_size, max_context_len), dtype=np.float32)
    
    context_lengths = []
    
    for i, item in enumerate(batch):
        ctx_len = item['states'].shape[0]
        states[i, :ctx_len] = item['states']
        actions[i, :ctx_len] = item['actions']
        rewards[i, :ctx_len] = item['rewards']
        
        context_lengths.append(ctx_len)
    
    res = {
        'states': torch.tensor(states, requires_grad=False, dtype=torch.float),
        'actions': F.one_hot(torch.tensor(actions, requires_grad=False, dtype=torch.long), num_classes=num_actions),
        'rewards': torch.tensor(rewards, dtype=torch.float, requires_grad=False),
        'context_lengths': torch.tensor(context_lengths, dtype=torch.long),  # For masking
    }
    
    return res


def get_rad_data_loader(dataset, batch_size, config, shuffle=True, use_length_grouping=True):
    """Data loader for RAD with variable-length collate function."""
    from torch.utils.data import DataLoader
    from dataset import CompressionBucketBatchSampler, LengthGroupedSampler
    
    collate_fn = partial(rad_collate_fn, grid_size=config['grid_size'], num_actions=config['num_actions'])
    num_workers = config.get('num_workers', 0)
    batching_strategy = config.get('rad_batching_strategy', 'length_grouped')

    if batching_strategy not in ('compression_buckets', 'length_grouped', 'variable'):
        raise ValueError(f'Unknown RAD batching strategy: {batching_strategy}')

    if batching_strategy == 'compression_buckets':
        sampler = CompressionBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    if use_length_grouping:
        sampler = LengthGroupedSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        persistent_workers=num_workers > 0
    )

# Curriculum schedule: (step, max_compressions)
DEFAULT_CURRICULUM = [
    (0, 1),       # Start with max 1 compression
    (10000, 2),   # Allow 2 compressions
    (25000, 3),   # Allow 3 compressions
    (40000, None), # Unlimited
]

# Default length distributions for each curriculum stage
DEFAULT_LENGTH_DISTRIBUTIONS = {
    1: {'short': 0.50, 'medium': 0.45, 'long': 0.05, 'very_long': 0.00},
    2: {'short': 0.35, 'medium': 0.40, 'long': 0.20, 'very_long': 0.05},
    3: {'short': 0.25, 'medium': 0.30, 'long': 0.30, 'very_long': 0.15},
    None: {'short': 0.20, 'medium': 0.25, 'long': 0.30, 'very_long': 0.25},
}


def get_curriculum_from_config(config):
    """
    Get curriculum schedule from config or use default.
    
    Returns:
        list of tuples: [(step, max_compressions, length_distribution), ...]
    """
    if 'curriculum_schedule' in config:
        curriculum = []
        for item in config['curriculum_schedule']:
            step = item['step']
            max_comp = item['max_compressions']
            # Get length distribution from config or use default
            length_dist = item.get('length_distribution', 
                                   DEFAULT_LENGTH_DISTRIBUTIONS.get(max_comp, DEFAULT_LENGTH_DISTRIBUTIONS[None]))
            curriculum.append((step, max_comp, length_dist))
        return curriculum
    # Default curriculum with default distributions
    return [(0, 1, DEFAULT_LENGTH_DISTRIBUTIONS[1]),
            (10000, 2, DEFAULT_LENGTH_DISTRIBUTIONS[2]),
            (25000, 3, DEFAULT_LENGTH_DISTRIBUTIONS[3]),
            (40000, None, DEFAULT_LENGTH_DISTRIBUTIONS[None])]


def get_curriculum_max_compressions(step, curriculum):
    """Get max compressions allowed at current step."""
    max_comp = curriculum[0][1]
    for threshold, comp, _ in curriculum:
        if step >= threshold:
            max_comp = comp
    return max_comp


def get_curriculum_length_distribution(step, curriculum):
    """Get length distribution for current curriculum stage."""
    length_dist = curriculum[0][2]
    for threshold, _, dist in curriculum:
        if step >= threshold:
            length_dist = dist
    return length_dist


def get_curriculum_stage(step, curriculum):
    """Get the current curriculum stage index."""
    stage = 0
    for i, (threshold, _, _) in enumerate(curriculum):
        if step >= threshold:
            stage = i
    return stage


def normalize_compiled_state_dict(state_dict):
    """Map torch.compile wrapper keys back to the uncompiled checkpoint format."""
    return {
        key.replace('._orig_mod.', '.').removeprefix('_orig_mod.'): value
        for key, value in state_dict.items()
    }


def checkpoint_state_dict(model):
    """Return a state dict that can be loaded with or without torch.compile."""
    return normalize_compiled_state_dict(model.state_dict())


def maybe_compile_model(model, config, is_main):
    """Compile tensor-heavy RAD submodules while leaving Python recurrence eager."""
    if not config.get('torch_compile', False):
        return model

    if not hasattr(torch, 'compile'):
        if is_main:
            print('WARNING: torch.compile requested but unavailable in this PyTorch build.')
        return model

    if config.get('torch_compile_suppress_errors', True):
        try:
            import importlib
            dynamo = importlib.import_module('torch._dynamo')
            dynamo.config.suppress_errors = True
        except Exception as exc:
            if is_main:
                print(f'WARNING: Unable to set torch._dynamo suppress_errors: {exc}')

    compile_kwargs = {
        'mode': config.get('torch_compile_mode', 'reduce-overhead'),
        'fullgraph': bool(config.get('torch_compile_fullgraph', False)),
        'dynamic': bool(config.get('torch_compile_dynamic', True)),
    }
    backend = config.get('torch_compile_backend')
    if backend:
        compile_kwargs['backend'] = backend

    module_names = config.get(
        'torch_compile_modules',
        ['ad_transformer', 'compression_transformer'],
    )
    compiled_modules = []
    for module_name in module_names:
        if not hasattr(model, module_name):
            if is_main:
                print(f'WARNING: Cannot torch.compile missing module: {module_name}')
            continue
        module = getattr(model, module_name)
        setattr(model, module_name, torch.compile(module, **compile_kwargs))
        compiled_modules.append(module_name)

    if is_main:
        print(f'torch.compile enabled for: {compiled_modules}')
        print(f'torch.compile kwargs: {compile_kwargs}')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_ckpt', type=str, default=None,
                       help='Path to pre-trained compression checkpoint')
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Disable curriculum learning')
    parser.add_argument('--config', type=str, default='rad_dr',
                       help='Model config name (without .yaml extension)')
    parser.add_argument('--env', type=str, default='darkroom',
                       help='Environment name: darkroom or dark_key_to_door')
    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load configs based on environment
    if args.env == 'darkroom':
        config = get_config('./config/env/darkroom.yaml')
        config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    elif args.env == 'dark_key_to_door':
        config = get_config('./config/env/dark_key_to_door.yaml')
        config.update(get_config('./config/algorithm/ppo_dark_key_to_door.yaml'))
    else:
        raise ValueError(f'Unknown environment: {args.env}')
    config.update(get_config(f'./config/model/{args.config}.yaml'))

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    log_dir = path.join('./runs', f"RAD-{config['env']}-seed{config['env_split_seed']}")
    
    # Check if already exists
    config_save_path = path.join(log_dir, 'config.yaml')
    try:
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists:
        print(f'WARNING: {log_dir} already exists. Will resume if checkpoint exists.')
    
    config['log_dir'] = log_dir
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'fp16'
    configure_torch_runtime(config)
    progress_interval = max(1, int(config.get('progress_interval', 50)))
    in_context_eval_episodes = max(1, int(config.get('in_context_eval_episodes', 100)))

    # Curriculum settings - load from config or use default
    use_curriculum = not args.no_curriculum
    curriculum = get_curriculum_from_config(config) if use_curriculum else [(0, None, DEFAULT_LENGTH_DISTRIBUTIONS[None])]

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
        print(f'Curriculum enabled: {use_curriculum}')
        print(f'Curriculum schedule: {curriculum}')
        print(f'Max context length: {config.get("max_context_length", 800)}')
        print(f'Train source timesteps: {config.get("train_source_timesteps", 1000)}')
        print(f'Train timesteps: {config.get("train_timesteps", 100000)}')
        print(f'RAD batching strategy: {config.get("rad_batching_strategy", "length_grouped")}')
        
        # Save config
        config_save_path = path.join(log_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)

    # Create model
    model = MODEL[config['model']](config)

    # Load pre-trained compression if available
    if args.pretrain_ckpt:
        if is_main:
            print(f'Loading pre-trained compression from {args.pretrain_ckpt}')
        model.load_pretrained_compression(args.pretrain_ckpt)
    else:
        # Try to find pre-trained checkpoint automatically
        pretrain_dir = path.join('./runs', f"RAD-pretrain-{config['env']}-seed{config['env_split_seed']}")
        pretrain_path = path.join(pretrain_dir, 'pretrain-final.pt')
        if path.exists(pretrain_path):
            if is_main:
                print(f'Found pre-trained compression at {pretrain_path}')
            model.load_pretrained_compression(pretrain_path)
        elif is_main:
            print('WARNING: No pre-trained compression found. Training from scratch.')

    step = 0
    resume_ckpt = None
    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        resume_ckpt = torch.load(ckpt_path, map_location=config['device'])
        load_result = model.load_state_dict(
            normalize_compiled_state_dict(resume_ckpt['model']),
            strict=False,
        )
        step = resume_ckpt['step']
        if is_main:
            print(f'Model checkpoint loaded from {ckpt_path}')
            if load_result.missing_keys:
                print(f'Missing model keys initialized from current config: {load_result.missing_keys}')
            if load_result.unexpected_keys:
                print(f'Unexpected model keys ignored: {load_result.unexpected_keys}')

    model = maybe_compile_model(model, config, is_main)

    if is_main:
        load_start_time = datetime.now()
        print(f'Data loading started at {load_start_time}')

    # Create datasets
    train_dataset = RADDataset(
        config, 
        config['traj_dir'], 
        'train', 
        config['train_n_stream'], 
        config['train_source_timesteps']
    )
    
    if is_main:
        print(f'Dataset sequence length: {train_dataset.seq_length}')
        print(f'Number of training histories: {train_dataset.n_histories}')

    if use_curriculum:
        initial_max_comp = get_curriculum_max_compressions(step, curriculum)
        model.set_curriculum(initial_max_comp)
        train_dataset.update_max_compressions(initial_max_comp)
        train_dataset.update_length_distribution(get_curriculum_length_distribution(step, curriculum))
    else:
        train_dataset.update_max_compressions(None)
    
    # Use standard AD dataset for testing (fixed length)
    test_dataset = ADDataset(
        config, 
        config['traj_dir'], 
        'test', 
        1, 
        config['train_source_timesteps']
    )

    train_dataloader = get_rad_data_loader(
        train_dataset, 
        batch_size=config['train_batch_size'], 
        config=config, 
        shuffle=True
    )
    train_dataloader = next_dataloader(train_dataloader)

    # Standard data loader for test - use fewer workers and no persistence to avoid leaks
    from torch.utils.data import DataLoader
    from functools import partial
    from utils import ad_collate_fn
    
    test_collate_fn = partial(ad_collate_fn, grid_size=config['grid_size'])
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['test_batch_size'], 
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=0,  # Use main process to avoid worker issues during evaluation
        persistent_workers=False
    )
    
    if is_main:
        load_end_time = datetime.now()
        print(f'Data loading ended at {load_end_time}')
        print(f'Elapsed time: {load_end_time - load_start_time}')

    # Optimizer for all parameters
    optimizer = AdamW(
        model.parameters(), 
        lr=config['lr'], 
        betas=(config['beta1'], config['beta2']), 
        weight_decay=config['weight_decay']
    )
    
    # Use curriculum-aware scheduler if curriculum is enabled, otherwise use standard cosine
    if use_curriculum:
        lr_sched = get_curriculum_aware_scheduler(
            optimizer=optimizer,
            curriculum=curriculum,
            total_steps=config['train_timesteps'],
            initial_warmup_steps=config.get('num_warmup_steps', 1000),
            stage_warmup_steps=config.get('stage_warmup_steps', 500),
            min_lr_ratio=config.get('min_lr_ratio', 0.1),
        )
        if is_main:
            print(f'Using curriculum-aware LR scheduler with stage warmups')
    else:
        lr_sched = get_cosine_schedule_with_warmup(
            optimizer, 
            config['num_warmup_steps'], 
            config['train_timesteps']
        )
        if is_main:
            print(f'Using standard cosine LR scheduler')

    if resume_ckpt is not None:
        try:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
            lr_sched.load_state_dict(resume_ckpt['lr_sched'])
            optimizer_state_loaded = True
        except ValueError as exc:
            optimizer_state_loaded = False
        if is_main:
            if optimizer_state_loaded:
                print(f'Optimizer and scheduler checkpoint loaded at step {step}')
            else:
                print(f'WARNING: Optimizer/scheduler checkpoint incompatible, reinitializing them: {exc}')

    # Setup evaluation environments
    env_name = config['env']
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    eval_num_train_envs = int(config.get('eval_num_train_envs', 10))
    eval_num_test_envs = int(config.get('eval_num_test_envs', 10))
    train_env_args = train_env_args[:eval_num_train_envs]
    test_env_args = test_env_args[:eval_num_test_envs]
    env_args = train_env_args + test_env_args    
    if len(env_args) == 0:
        raise ValueError('At least one in-context evaluation environment is required')

    if env_name == "darkroom":
        envs = DummyVecEnv([make_env(config, goal=arg) for arg in env_args])
    elif env_name == "dark_key_to_door":
        envs = DummyVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in env_args])
    else:
        raise NotImplementedError(f'Environment not supported: {env_name}')

    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )

    if is_main:
        start_time = datetime.now()
        print(f'Training started at {start_time}')

    # Track compression statistics
    compression_counts = []
    
    # Track current curriculum stage for length distribution updates
    current_curriculum_stage = -1
    
    # Best model tracking for early stopping / model selection
    best_eval_reward = -float('inf')
    best_step = 0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 5)  # Number of eval intervals without improvement
    save_best_model = config.get('save_best_model', True)

    # Training loop
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not is_main) as pbar:
        pbar.update(step)

        while step < config['train_timesteps']:
            next_step = step + 1
            
            # Update curriculum (model max_compressions AND dataset length distribution)
            if use_curriculum:
                # Check if curriculum stage changed - update dataset length distribution
                new_stage = get_curriculum_stage(next_step, curriculum)
                if new_stage != current_curriculum_stage:
                    current_curriculum_stage = new_stage
                    max_comp = get_curriculum_max_compressions(next_step, curriculum)
                    accelerator.unwrap_model(model).set_curriculum(max_comp)
                    train_dataset.update_max_compressions(max_comp)
                    new_length_dist = get_curriculum_length_distribution(next_step, curriculum)
                    train_dataset.update_length_distribution(new_length_dist)
                    if is_main:
                        print(f'\n[Step {next_step}] Curriculum stage {new_stage}: max_comp={max_comp}, '
                              f'length_dist={new_length_dist}')

            batch = next(train_dataloader)
            step = next_step
            
            with accelerator.autocast():
                output = model(batch)
            
            # Use total loss (action + reconstruction regularization)
            loss = output['loss_total']
            
            # Track compressions
            compression_counts.append(output['num_compressions'])
            if len(compression_counts) > 1000:
                compression_counts.pop(0)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            avg_compressions = np.mean(compression_counts) if compression_counts else 0
            if is_main and (step == 1 or step % progress_interval == 0 or step == config['train_timesteps']):
                pbar.set_postfix(
                    loss=loss.detach().float().item(),
                    acc=output['acc_action'].detach().float().item(),
                    n_comp=output['num_compressions'],
                    avg_comp=f'{avg_compressions:.2f}',
                )

            # Logging
            if is_main and step % config['summary_interval'] == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'].item(), step)
                writer.add_scalar('train/loss_recon', output['loss_recon'].item(), step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
                writer.add_scalar('train/acc_action', output['acc_action'].item(), step)
                writer.add_scalar('train/num_compressions', output['num_compressions'], step)
                writer.add_scalar('train/avg_compressions', avg_compressions, step)
                
                if use_curriculum:
                    curr_max = get_curriculum_max_compressions(step, curriculum)
                    writer.add_scalar('train/curriculum_max_compressions', 
                                    curr_max if curr_max is not None else -1, step)
                    writer.add_scalar('train/curriculum_stage', current_curriculum_stage, step)
                    
                    # Log length distribution
                    curr_dist = get_curriculum_length_distribution(step, curriculum)
                    for category, prob in curr_dist.items():
                        writer.add_scalar(f'train/length_dist_{category}', prob, step)

            # Evaluation
            if is_main and step % config['eval_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f'\nEvaluating started at {eval_start_time}')

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_cnt = 0

                    for j, test_batch in enumerate(test_dataloader):
                        test_output = model(test_batch)
                        cnt = len(test_batch['states'])
                        test_loss_action += test_output['loss_action'].item() * cnt
                        test_acc_action += test_output['acc_action'].item() * cnt
                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)
                writer.add_scalar('test/acc_action', test_acc_action / test_cnt, step)

                eval_end_time = datetime.now()
                print(f'Evaluating ended at {eval_end_time}')
                print(f'Elapsed time: {eval_end_time - eval_start_time}')
                
                # Clean up evaluation tensors
                del test_output, test_batch
                
                model.train()
                torch.cuda.empty_cache()
                gc.collect()

            # In-context evaluation (less frequent)
            if is_main and step % config['gen_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                
                with torch.no_grad():
                    unwrapped = accelerator.unwrap_model(model)
                    eval_output = unwrapped.evaluate_in_context(
                        vec_env=envs, 
                        eval_timesteps=config['horizon'] * in_context_eval_episodes
                    )
                    
                    mean_reward = eval_output['reward_episode'].mean()
                    total_compressions = eval_output['total_compressions']
                    
                    # Per-environment rewards for detailed tracking
                    env_rewards = eval_output['reward_episode'].mean(axis=1)
                    
                    writer.add_scalar('eval/mean_reward', mean_reward, step)
                    writer.add_scalar('eval/total_compressions', total_compressions, step)
                    for env_idx, env_reward in enumerate(env_rewards):
                        writer.add_scalar(f'eval/env_{env_idx}_reward', env_reward, step)
                    
                    print(f'\nIn-context eval: mean_reward={mean_reward:.3f}, compressions={total_compressions}')
                    print(f'Per-env rewards: {env_rewards}')
                    
                    # Best model tracking
                    if save_best_model and mean_reward > best_eval_reward:
                        best_eval_reward = mean_reward
                        best_step = step
                        patience_counter = 0
                        
                        # Save best model
                        best_ckpt_path = path.join(config['log_dir'], 'best-model.pt')
                        torch.save({
                            'step': step,
                            'config': config,
                            'model': checkpoint_state_dict(unwrapped),
                            'optimizer': optimizer.state_dict(),
                            'lr_sched': lr_sched.state_dict(),
                            'eval_reward': mean_reward,
                        }, best_ckpt_path)
                        print(f'New best model saved! reward={mean_reward:.3f} at step {step}')
                    else:
                        patience_counter += 1
                        print(f'No improvement. Best: {best_eval_reward:.3f} at step {best_step} (patience: {patience_counter}/{patience})')
                    
                    del eval_output
                
                model.train()
                torch.cuda.empty_cache()
                gc.collect()

            pbar.update(1)

            # Save checkpoint
            if is_main and step % config['ckpt_interval'] == 0:
                # Remove old checkpoints with error handling
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for old_ckpt_path in ckpt_paths:
                    try:
                        os.remove(old_ckpt_path)
                    except OSError:
                        pass  # File may be in use, skip

                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')
                
                # Get unwrapped model state dict
                unwrapped_model = accelerator.unwrap_model(model)
                
                torch.save({
                    'step': step,
                    'config': config,
                    'model': checkpoint_state_dict(unwrapped_model),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')

    # Cleanup
    if is_main:
        writer.flush()
    
    envs.close()

    if is_main:
        end_time = datetime.now()
        print(f'\nTraining ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
