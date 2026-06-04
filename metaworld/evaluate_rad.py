"""
Evaluation script for Recurrent Algorithm Distillation (RAD) on Meta-world.

Evaluates a trained RAD model on test environments with in-context learning.

Usage:
    python evaluate_rad.py --ckpt_dir ./runs/RAD-ml1-pick-place-v2
"""

from datetime import datetime
from glob import glob
import argparse
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path
import numpy as np
import metaworld

from env import make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def _load_ckpt_with_compat(ckpt_path, device):
    """Load a checkpoint with compatibility fallbacks."""
    import pickle

    try:
        return torch.load(ckpt_path, map_location=device)
    except Exception as e:
        try:
            with torch.serialization.safe_globals([np._core.multiarray.scalar]):
                return torch.load(ckpt_path, map_location=device)
        except Exception:
            try:
                return torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                raise


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--eval_timesteps', type=int, default=None, help='Evaluation timesteps (default: test_source_timesteps from config)')
    parser.add_argument('--use_best', action='store_true', help='Use best-model.pt instead of latest checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ckpt_dir = args.ckpt_dir
    
    best_model_path = path.join(ckpt_dir, 'best-model.pt')
    if args.use_best and path.exists(best_model_path):
        ckpt_path = best_model_path
        ckpt = _load_ckpt_with_compat(ckpt_path, device)
        print(f'Best model loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = _load_ckpt_with_compat(ckpt_path, device)
            print(f'Checkpoint loaded from {ckpt_path}')
            config = ckpt['config']
        else:
            raise ValueError('No checkpoint found.')
    
    config['device'] = device
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    print(f"Model: {model_name}")
    print(f"Task: {config['task']}")
    print(f"Max sequence length: {config['n_transit']}")
    print(f"Compression tokens: {config.get('n_compress_tokens', 'N/A')}")

    ml1 = metaworld.ML1(env_name=config['task'], seed=config['mw_seed'])
    train_envs = []
    test_envs = []
    for task_name, env_cls in ml1.train_classes.items():
        task_instances = [task for task in ml1.train_tasks if task.env_name == task_name]
        for i in range(config['n_train_envs_per_task']):
            train_envs.append(make_env(config, env_cls, task_instances[i]))
    for task_name, env_cls in ml1.test_classes.items():
        task_instances = [task for task in ml1.test_tasks if task.env_name == task_name]
        for i in range(config['n_test_envs_per_task']):
            test_envs.append(make_env(config, env_cls, task_instances[i]))

    envs = train_envs + test_envs
    envs = SubprocVecEnv(envs)
    model.set_obs_space(envs.observation_space)
    model.set_action_space(envs.action_space)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    eval_timesteps = args.eval_timesteps if args.eval_timesteps else config['test_source_timesteps']

    start_time = datetime.now()
    print(f'Starting at {start_time}')
    print(f'Evaluating for {eval_timesteps} timesteps')

    with torch.no_grad():
        eval_output = model.evaluate_in_context(
            vec_env=envs, 
            eval_timesteps=eval_timesteps
        )
        train_rewards = eval_output['reward_episode'][:len(train_envs)]
        test_rewards = eval_output['reward_episode'][len(train_envs):]
        if 'success' in eval_output.keys():
            train_success = eval_output['success'][:len(train_envs)]
            test_success = eval_output['success'][len(train_envs):]
        else:
            train_success = None
            test_success = None
        # RAD-specific metrics
        total_compressions = eval_output.get('total_compressions', 0)
        compression_events = eval_output.get('compression_events', [])
        result_path = path.join(ckpt_dir, 'eval_result.npy')
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    # Save full results
    with open(result_path, 'wb') as f:
        np.save(f, eval_output['reward_episode'])

    print(f'\nResults saved to {result_path}')
