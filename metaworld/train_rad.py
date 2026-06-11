"""
Fine-tuning script for Recurrent Algorithm Distillation (RAD) on Meta-world.

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
from modulefinder import ModuleFinder
import shutil
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
from env import make_env
from model import MODEL
from utils import get_config, next_dataloader, get_curriculum_aware_scheduler, log_in_context
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import metaworld

import numpy as np
import torch.nn.functional as F
from functools import partial


def rad_collate_fn(batch, dim_actions):
    """
    Collate function for variable-length RAD dataset.
    Handles sequences of different lengths by padding.
    For Meta-world with continuous actions.
    """
    # Find max context length in batch
    max_context_len = max(item['states'].shape[0] for item in batch)
    
    batch_size = len(batch)
    dim_state = batch[0]['states'].shape[1]
    
    # Initialize padded arrays
    states = np.zeros((batch_size, max_context_len, dim_state), dtype=np.float32)
    actions = np.zeros((batch_size, max_context_len, dim_actions), dtype=np.float32)
    rewards = np.zeros((batch_size, max_context_len), dtype=np.float32)
    next_states = np.zeros((batch_size, max_context_len, dim_state), dtype=np.float32)
    
    query_states = []
    target_actions = []
    context_lengths = []
    
    for i, item in enumerate(batch):
        ctx_len = item['states'].shape[0]
        states[i, :ctx_len] = item['states']
        actions[i, :ctx_len] = item['actions']
        rewards[i, :ctx_len] = item['rewards']
        next_states[i, :ctx_len] = item['next_states']
        
        query_states.append(item['query_states'])
        target_actions.append(item['target_actions'])
        context_lengths.append(ctx_len)
    
    res = {
        'query_states': torch.tensor(np.array(query_states), requires_grad=False, dtype=torch.float),
        'target_actions': torch.tensor(np.array(target_actions), requires_grad=False, dtype=torch.float),  # Continuous actions
        'states': torch.tensor(states, requires_grad=False, dtype=torch.float),
        'actions': torch.tensor(actions, requires_grad=False, dtype=torch.float),  # Continuous actions
        'rewards': torch.tensor(rewards, dtype=torch.float, requires_grad=False),
        'next_states': torch.tensor(next_states, requires_grad=False, dtype=torch.float),
        'context_lengths': torch.tensor(context_lengths, dtype=torch.long),  # For masking
    }
    
    return res


def get_rad_data_loader(dataset, batch_size, config, shuffle=True, distributed=False, use_length_grouping=True):
    """Data loader for RAD with variable-length collate function.
    
    Args:
        dataset: RADDataset instance
        batch_size: Number of samples per batch
        config: Configuration dict
        shuffle: Whether to shuffle data
        distributed: Whether using distributed training
        use_length_grouping: If True, use LengthGroupedSampler to minimize padding
    """
    from torch.utils.data import DataLoader
    from dataset import LengthGroupedSampler
    
    collate_fn = partial(rad_collate_fn, dim_actions=config['dim_actions'])
    
    # Use num_workers=0 to avoid distributed deadlocks (safer with Accelerate)
    num_workers = 0 if distributed else config.get('num_workers', 0)
    
    if use_length_grouping:
        # Use length-grouped sampler to reduce padding
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
            persistent_workers=False,
            pin_memory=False,
        )
    else:
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn, 
            num_workers=num_workers, 
            persistent_workers=False,
            pin_memory=False,
        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=False, default='./config/algorithm/ppo_ml1.yaml', help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=False, default='./config/env/ml1.yaml', help="Environment config")
    parser.add_argument('--model-config', '-mc', required=False, default='./config/model/rad_ml1.yaml', help="Model config")
    parser.add_argument('--log-dir', '-l', required=False, default='./runs', help="Log directory")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Trajectory directory")
    parser.add_argument('--no-backup', '-nb', required=False, default=False, help="Save code", action='store_true')
    parser.add_argument('--override', '-o', default='')
    parser.add_argument('--pretrain_ckpt', type=str, default=None,
                       help='Path to pre-trained compression checkpoint')
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Disable curriculum learning')
    parser.add_argument('--resume', required=False, default=False, help="Resume train", action='store_true')
    parser.add_argument('--mixed-precision', '-m', required=False, default='bf16')
    parser.add_argument('--disable-tqdm', '-d', required=False, default=False, action='store_true')
    parser.add_argument('--gradient-accumulation-steps', '-ga', type=int, default=1,
                       help='Number of gradient accumulation steps (effective batch = batch_size * ga_steps * n_gpus)')
    parser.add_argument('--gradient-checkpointing', '-gc', action='store_true',
                       help='Enable gradient checkpointing to reduce memory (slower but uses less VRAM)')
    args = parser.parse_args()
    return args

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

    log_dir = path.join(args.log_dir, f"RAD-ml1-{config['task']}-var{config.get('learn_var', False)}")
    
    config['log_dir'] = log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    
    traj_dir = path.join(args.traj_dir, config['task'])
    config['traj_dir'] = traj_dir
    config['mixed_precision'] = args.mixed_precision

    # Curriculum settings - load from config or use default
    use_curriculum = not args.no_curriculum
    curriculum = get_curriculum_from_config(config) if use_curriculum else [(0, None, DEFAULT_LENGTH_DISTRIBUTIONS[None])]

    # Initialize Accelerator for multi-GPU with gradient accumulation
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    if args.mixed_precision == 'bf16' or args.mixed_precision == 'fp16':
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    elif args.mixed_precision == 'fp32':
        accelerator = Accelerator(
            mixed_precision='no',
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    else:
        raise ValueError(f'Unsupported mixed precision: {args.mixed_precision}')

    # Debug print immediately after accelerator creation
    print(f'[Process {accelerator.process_index}] Accelerator initialized. Device: {accelerator.device}', flush=True)

    config['device'] = accelerator.device
    
    # Only main process handles logging and checkpointing
    is_main = accelerator.is_main_process

    # Synchronize all processes before file operations
    accelerator.wait_for_everyone()

    # Prevent overwrite: check only on main process
    config_exists = False
    if is_main:
        try:
            with open(config_save_path, 'r') as f:
                f.read(1)
                config_exists = True
        except FileNotFoundError:
            config_exists = False
        
        if config_exists and not args.resume:
            print(f'WARNING: {log_dir} already exists. Skipping...')
    
    # Broadcast the decision to all processes
    accelerator.wait_for_everyone()
    
    # Use a simple file check that all processes can agree on
    if path.exists(config_save_path) and not args.resume:
        # All processes must exit together to avoid deadlock
        accelerator.wait_for_everyone()
        exit(0)
    
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, flush_secs=15)
        print(f'Using Device: {config["device"]}')
        print(f'Number of processes: {accelerator.num_processes}')
        print(f'Gradient accumulation steps: {gradient_accumulation_steps}')
        effective_batch = config['train_batch_size'] * accelerator.num_processes * gradient_accumulation_steps
        print(f'Effective batch size: {config["train_batch_size"]} * {accelerator.num_processes} * {gradient_accumulation_steps} = {effective_batch}')
        print(f'Gradient checkpointing: {args.gradient_checkpointing}')
        print(f'Curriculum enabled: {use_curriculum}')
        print(f'Curriculum schedule: {curriculum}')
        print(f'Max context length: {config.get("max_context_length", 800)}')
        print(f'Train source timesteps: {config.get("train_source_timesteps", 1000)}')
        print(f'Train timesteps: {config.get("train_timesteps", 100000)}')
        
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

    # Synchronize all processes after main-only initialization
    # This ensures all processes start model creation together
    print(f'[Process {accelerator.process_index}] Waiting for initialization to complete...', flush=True)
    accelerator.wait_for_everyone()
    print(f'[Process {accelerator.process_index}] Initialization complete.', flush=True)

    # Create model
    print(f'[Process {accelerator.process_index}] Creating model...', flush=True)
    model = MODEL[config['model']](config)
    print(f'[Process {accelerator.process_index}] Model created.', flush=True)

    # Enable gradient checkpointing (saves memory, slower)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_main:
            print('Gradient checkpointing enabled - will use less memory but be ~20% slower')

    # Load pre-trained compression if available
    print(f'[Process {accelerator.process_index}] Loading pre-trained compression...', flush=True)
    if args.pretrain_ckpt:
        if is_main:
            print(f'Loading pre-trained compression from {args.pretrain_ckpt}')
        model.load_pretrained_compression(args.pretrain_ckpt)
    else:
        # Try to find pre-trained checkpoint automatically
        pretrain_dir = path.join('./runs', f"RAD-pretrain-ml1-{config['task']}")
        pretrain_path = path.join(pretrain_dir, 'pretrain-final.pt')
        if path.exists(pretrain_path):
            if is_main:
                print(f'Found pre-trained compression at {pretrain_path}')
            model.load_pretrained_compression(pretrain_path)
        elif is_main:
            print('WARNING: No pre-trained compression found. Training from scratch.')
    
    print(f'[Process {accelerator.process_index}] Pre-trained compression loaded.', flush=True)
    
    # Synchronize before data loading
    accelerator.wait_for_everyone()

    if is_main:
        load_start_time = datetime.now()
        print(f'Data loading started at {load_start_time}')

    # Sequential data loading: load one process at a time to avoid HDF5 conflicts
    # This is slower but guarantees no file locking issues
    for proc_idx in range(accelerator.num_processes):
        if accelerator.process_index == proc_idx:
            print(f'[Process {accelerator.process_index}] Creating train dataset...', flush=True)
            # Create datasets
            train_dataset = RADDataset(
                config, 
                traj_dir, 
                'train', 
                config['train_n_seed'],
                config['train_n_stream'], 
                config['train_source_timesteps']
            )
            
            print(f'[Process {accelerator.process_index}] Train dataset created.', flush=True)
            
            if is_main:
                print(f'Dataset sequence length: {train_dataset.seq_length}')
                print(f'Number of training histories: {train_dataset.n_histories}')
            
            print(f'[Process {accelerator.process_index}] Creating test dataset...', flush=True)
            # Use standard AD dataset for testing (fixed length)
            test_dataset = ADDataset(
                config, 
                traj_dir, 
                'test', 
                1,
                1, 
                config['train_source_timesteps']
            )
            print(f'[Process {accelerator.process_index}] Test dataset created.', flush=True)
        
        # Wait for current process to finish before next one starts
        accelerator.wait_for_everyone()
    
    print(f'[Process {accelerator.process_index}] Creating dataloaders...', flush=True)

    train_dataloader = get_rad_data_loader(
        train_dataset, 
        batch_size=config['train_batch_size'], 
        config=config, 
        shuffle=True,
        distributed=(accelerator.num_processes > 1)
    )
    # NOTE: Wrap dataloader after accelerator.prepare()

    # Test dataloader: fewer workers, no persistence to avoid leaks
    from torch.utils.data import DataLoader
    from utils import ad_collate_fn
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['test_batch_size'], 
        shuffle=False,
        collate_fn=ad_collate_fn,  # ad_collate_fn doesn't need dim_actions
        num_workers=0,  # Use main process to avoid worker issues during evaluation
        persistent_workers=False
    )
    
    if is_main:
        load_end_time = datetime.now()
        print(f'Data loading ended at {load_end_time}')
        print(f'Elapsed time: {load_end_time - load_start_time}')

    print(f'[Process {accelerator.process_index}] Creating optimizer...', flush=True)
    
    # Optimizer for all parameters
    optimizer = AdamW(
        model.parameters(), 
        lr=config['lr'], 
        betas=(config['beta1'], config['beta2']), 
        weight_decay=config['weight_decay']
    )
    
    print(f'[Process {accelerator.process_index}] Optimizer created. Creating LR scheduler...', flush=True)
    
    # Use curriculum-aware scheduler if enabled; else use cosine
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
    
    step = 0
    
    print(f'[Process {accelerator.process_index}] LR scheduler created.', flush=True)

    # Resume checkpoint
    if args.resume:
        ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_sched.load_state_dict(ckpt['lr_sched'])
            step = ckpt['step']
            if is_main:
                print(f'Checkpoint loaded from {ckpt_path}')

    # Define environments for evaluation - ONLY on main process
    # Non-main processes don't need envs; avoids spawning conflicting workers
    print(f'[Process {accelerator.process_index}] Before environment setup barrier', flush=True)
    accelerator.wait_for_everyone()
    print(f'[Process {accelerator.process_index}] After environment setup barrier', flush=True)
    
    eval_envs = None
    n_test_envs = config.get('n_test_envs_per_task', 50)  # Default for non-main processes
    
    if is_main:
        print(f'Initializing Meta-world ML1 benchmark...', flush=True)
        ml1 = metaworld.ML1(env_name=config['task'], seed=config['mw_seed'])
        
        test_envs = []
        
        for task_name, env_cls in ml1.test_classes.items():
            task_instances = [task for task in ml1.test_tasks if task.env_name == task_name]
            for task_instance in task_instances:
                test_envs.append(make_env(config, env_cls, task_instance))

        envs = test_envs
        n_test_envs = len(test_envs)
        
        # Use DummyVecEnv to avoid multiprocessing conflicts
        # with Accelerate's distributed training on Windows
        print(f'Creating DummyVecEnv with {len(envs)} environments...', flush=True)
        eval_envs = DummyVecEnv(envs)
        print(f'DummyVecEnv ready.', flush=True)
        
        # Get observation/action space from the vectorized env
        model.set_obs_space(eval_envs.observation_space)
        model.set_action_space(eval_envs.action_space)
        print(f'[Process {accelerator.process_index}] Main process env setup complete', flush=True)
    else:
        # Non-main processes set obs/action space from config
        # Create dummy tensors with the right shapes based on config
        import gymnasium as gym
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config['dim_obs'],), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(config['dim_actions'],), dtype=np.float32
        )
        model.set_obs_space(obs_space)
        model.set_action_space(action_space)
        print(f'[Process {accelerator.process_index}] Non-main process env setup complete', flush=True)
    
    # Synchronize all processes after environment setup
    print(f'[Process {accelerator.process_index}] Before prepare barrier', flush=True)
    accelerator.wait_for_everyone()
    print(f'[Process {accelerator.process_index}] After prepare barrier', flush=True)

    # Prepare for distributed training
    if is_main:
        print(f'Preparing model and optimizer for distributed training...', flush=True)
    
    print(f'[Process {accelerator.process_index}] Calling accelerator.prepare()', flush=True)
    # Prepare optimizer and dataloaders with Accelerator; avoid auto-wrapping so we can manual DDP wrap
    optimizer, train_dataloader, lr_sched = accelerator.prepare(
        optimizer, train_dataloader, lr_sched
    )
    print(f'[Process {accelerator.process_index}] accelerator.prepare() done', flush=True)

    # Move model to device and manually wrap in DDP with find_unused_parameters
    model.to(accelerator.device)
    if accelerator.num_processes > 1:
        try:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            if torch.cuda.is_available():
                # Obtain local rank from env if set, else fall back to process index
                local_rank = int(os.environ.get('LOCAL_RANK', str(accelerator.process_index)))
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)
        except Exception:
            # If wrapping fails, continue and let Accelerate handle DDP (best effort)
            pass
    
    if is_main:
        print(f'Model prepared. Starting dataloader...', flush=True)
    
    # Now wrap dataloader in infinite iterator (after prepare)
    train_dataloader = next_dataloader(train_dataloader)
    
    # Warm up the dataloader by fetching the first batch
    # This ensures workers are initialized before entering the training loop
    print(f'[Process {accelerator.process_index}] Warming up dataloader...', flush=True)
    first_batch = next(train_dataloader)
    print(f'[Process {accelerator.process_index}] First batch fetched', flush=True)
    
    # Synchronize all processes before entering training loop
    print(f'[Process {accelerator.process_index}] Before training loop barrier', flush=True)
    accelerator.wait_for_everyone()
    print(f'[Process {accelerator.process_index}] After training loop barrier', flush=True)
    
    if is_main:
        print(f'Dataloader ready.', flush=True)

    if is_main:
        start_time = datetime.now()
        print(f'Training started at {start_time}', flush=True)
        print(f'Entering training loop...', flush=True)

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
    # Use the pre-fetched first_batch for step 1 if starting from scratch
    use_prefetched_batch = (step == 0)
    
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not is_main or args.disable_tqdm) as pbar:
        pbar.update(step)

        while step < config['train_timesteps']:
            # Get batch (use prefetched for first iteration if available)
            if use_prefetched_batch:
                batch = first_batch
                use_prefetched_batch = False
            else:
                batch = next(train_dataloader)
            
            step += 1
            
            # Update curriculum (model max_compressions AND dataset length distribution)
            if use_curriculum:
                max_comp = get_curriculum_max_compressions(step, curriculum)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.set_curriculum(max_comp)
                
                # Check if curriculum stage changed - update dataset length distribution
                new_stage = get_curriculum_stage(step, curriculum)
                if new_stage != current_curriculum_stage:
                    current_curriculum_stage = new_stage
                    new_length_dist = get_curriculum_length_distribution(step, curriculum)
                    train_dataset.update_length_distribution(new_length_dist)
                    if is_main:
                        print(f'\n[Step {step}] Curriculum stage {new_stage}: max_comp={max_comp}, '
                              f'length_dist={new_length_dist}')
            
            # Use accelerator.accumulate() for proper gradient accumulation
            # This handles gradient syncing and scaling automatically
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(batch)
                
                # Use total loss (action + reconstruction regularization)
                loss = output['loss_total']
                
                accelerator.backward(loss)
                
                # Only step optimizer when accumulation is complete
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    lr_sched.step()
            
            # Track compressions and context lengths
            compression_counts.append(output['num_compressions'])
            if len(compression_counts) > 1000:
                compression_counts.pop(0)

            avg_compressions = np.mean(compression_counts) if compression_counts else 0
            
            # Get batch context length info
            ctx_lens = batch['context_lengths']
            max_ctx = ctx_lens.max().item()
            mean_ctx = ctx_lens.float().mean().item()
            
            pbar.set_postfix(
                loss=loss.item(), 
                n_comp=output['num_compressions'],
                avg_comp=f'{avg_compressions:.2f}',
                ctx_len=f'{mean_ctx:.0f}/{max_ctx}'
            )

            # Logging
            if is_main and step % config['summary_interval'] == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'].item(), step)
                writer.add_scalar('train/loss_recon', output['loss_recon'].item(), step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
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
                    test_cnt = 0

                    for j, test_batch in enumerate(test_dataloader):
                        with accelerator.autocast():
                            test_output = model(test_batch)
                        cnt = len(test_batch['states'])
                        test_loss_action += test_output['loss_action'].item() * cnt
                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)

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
                        vec_env=eval_envs, 
                        eval_timesteps=config['test_source_timesteps']
                    )
                    
                    test_rewards = eval_output['reward_episode']
                    total_compressions = eval_output['total_compressions']
                    
                    mean_test_reward = test_rewards.mean()
                    
                    if 'success' in eval_output.keys():
                        test_success = eval_output['success']
                        
                        writer.add_scalar('test/success_rate', test_success.max(axis=1).mean(), step)
                    else:
                        test_success = None
                    
                    writer.add_scalar('test_gen/mean_reward', mean_test_reward, step)
                    writer.add_scalar('eval/total_compressions', total_compressions, step)
                    
                    log_in_context(values=test_rewards,
                                   max_reward=config['max_reward'],
                                   success=test_success,
                                   episode_length=config['horizon'],
                                   tag='test_gen/reward_episode',
                                   title='',
                                   xlabel='In-context steps',
                                   ylabel='Reward',
                                   step=step,
                                   writer=writer)
                    
                    print(f'\nIn-context eval: test_envs={n_test_envs}, test_reward={mean_test_reward:.3f}, compressions={total_compressions}')
                    
                    # Best model tracking
                    if save_best_model and mean_test_reward > best_eval_reward:
                        best_eval_reward = mean_test_reward
                        best_step = step
                        patience_counter = 0
                        
                        # Save best model
                        best_ckpt_path = path.join(config['log_dir'], 'best-model.pt')
                        torch.save({
                            'step': step,
                            'config': config,
                            'model': unwrapped.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_sched': lr_sched.state_dict(),
                            'eval_reward': mean_test_reward,
                        }, best_ckpt_path)
                        print(f'New best model saved! reward={mean_test_reward:.3f} at step {step}')
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
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')

    # Cleanup
    if is_main:
        writer.flush()
        eval_envs.close()

    if is_main:
        end_time = datetime.now()
        print(f'\nTraining ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
