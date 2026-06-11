"""
Evaluation script for Algorithm Distillation (AD) on Meta-world.

This script evaluates a trained AD model on Meta-world environments.

Usage:
    python evaluate.py --ckpt_dir ./runs/AD-ml1-pick-place-v2
"""

from datetime import datetime
from glob import glob
import argparse

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path

from env import make_env
from model import MODEL
from stable_baselines3.common.vec_env import DummyVecEnv
import metaworld
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--eval_timesteps', type=int, default=None, help='Evaluation timesteps (default: test_source_timesteps from config)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ckpt_dir = args.ckpt_dir
    ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))

    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        raise ValueError('No checkpoint found.')
    
    config['device'] = device
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    # Some checkpoints include obs/action bounds as buffers saved at save time;
    # allow extra keys and set the real spaces after env creation.
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    # Define environments for evaluation
    ml1 = metaworld.ML1(env_name=config['task'], seed=config['mw_seed'])
    
    test_envs = []
    
    for task_name, env_cls in ml1.test_classes.items():
        task_instances = [task for task in ml1.test_tasks if task.env_name == task_name]
        for task_instance in task_instances:
            test_envs.append(make_env(config, env_cls, task_instance))

    envs = DummyVecEnv(test_envs)
    model.set_obs_space(envs.observation_space)
    model.set_action_space(envs.action_space)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    eval_timesteps = args.eval_timesteps if args.eval_timesteps else config['test_source_timesteps']

    start_time = datetime.now()
    print(f'Starting at {start_time}')
    print(f'Evaluating for {eval_timesteps} timesteps')
    print(f'Evaluating {len(test_envs)} test environments and 0 train environments')

    with torch.no_grad():
        output = model.evaluate_in_context(vec_env=envs, eval_timesteps=eval_timesteps)
        
        test_rewards = output['reward_episode']
        
        if 'success' in output.keys():
            test_success = output['success']
        else:
            test_success = None
        
        save_path = path.join(ckpt_dir, 'eval_result.npy')
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    # Save full results
    with open(save_path, 'wb') as f:
        np.save(f, output['reward_episode'])

    print(f'\nResults saved to {save_path}')

    # Print summary statistics
    print('\n=== Test Environments ===')
    print(f'Mean reward per env: {test_rewards.mean(axis=1)}')
    print(f'Overall mean reward: {test_rewards.mean():.4f}')
    print(f'Std deviation: {test_rewards.std():.4f}')
    if test_success is not None:
        print(f'Success rate (max per env): {test_success.max(axis=1).mean():.4f}')