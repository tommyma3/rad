"""
Evaluation script for Recurrent Algorithm Distillation (RAD).

Evaluates a trained RAD model on test environments with in-context learning.

Usage:
    python evaluate_rad.py --ckpt_dir ./runs/RAD-darkroom-seed0
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

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./runs/RAD-darkroom-seed0',
                       help='Directory containing RAD checkpoint')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--use_best', action='store_true',
                       help='Use best-model.pt instead of latest checkpoint')
    args = parser.parse_args()
    
    ckpt_dir = args.ckpt_dir
    
    best_model_path = path.join(ckpt_dir, 'best-model.pt')
    if args.use_best and path.exists(best_model_path):
        ckpt_path = best_model_path
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f'Best model loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            print(f'Checkpoint loaded from {ckpt_path}')
            config = ckpt['config']
        else:
            raise ValueError('No checkpoint found.')
    
    config['device'] = device
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print(f"Model: {model_name}")
    print(f"Evaluation goals: {test_env_args}")
    print(f"Max sequence length: {config['n_transit']}")
    print(f"Compression tokens: {config.get('n_compress_tokens', 'N/A')}")

    if env_name == 'darkroom':
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in test_env_args])
    elif env_name == 'dark_key_to_door':
        envs = SubprocVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in test_env_args])
    else:
        raise ValueError('Unsupported env')

    model.set_obs_space(envs.observation_space)
    model.set_action_space(envs.action_space)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting at {start_time}')
    print(f'Evaluating for {args.eval_episodes} episodes')

    # Simple episodic evaluation logic could be added here; reuse existing evaluators
    envs.close()
