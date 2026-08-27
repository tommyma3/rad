from datetime import datetime
from glob import glob
import argparse

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from utils import normalize_compiled_state_dict
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./runs/AD-darkroom-seed0',
                        help='Directory containing AD checkpoint')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to a specific checkpoint file')
    parser.add_argument('--eval_timesteps', type=int, default=None,
                        help='Evaluation timesteps (overrides --eval_episodes)')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of episodes to evaluate when --eval_timesteps is not set')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_best', action='store_true',
                        help='Use best-model.pt instead of latest checkpoint')
    args = parser.parse_args()
    return args


def load_checkpoint(args):
    ckpt_dir = args.ckpt_dir

    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt_dir = path.dirname(ckpt_path) or ckpt_dir
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f'Checkpoint loaded from {ckpt_path}')
        return ckpt, ckpt_dir

    best_model_path = path.join(ckpt_dir, 'best-model.pt')
    if args.use_best and path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        print(f'Best model loaded from {best_model_path}')
        return ckpt, ckpt_dir

    ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f'Checkpoint loaded from {ckpt_path}')
        return ckpt, ckpt_dir

    raise ValueError('No checkpoint found.')


if __name__ == '__main__':
    args = parse_arguments()
    ckpt, ckpt_dir = load_checkpoint(args)
    config = ckpt['config']
    config['device'] = device

    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(normalize_compiled_state_dict(ckpt['model']))
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print("Evaluation goals: ", test_env_args)

    if env_name == 'darkroom':
        envs = DummyVecEnv([make_env(config, goal=arg) for arg in test_env_args])
    elif env_name == 'dktd':
        envs = DummyVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in test_env_args])
    else:
        raise NotImplementedError(f'Environment not supported: {env_name}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.eval_timesteps is not None:
        eval_timesteps = args.eval_timesteps
    else:
        eval_timesteps = config['horizon'] * args.eval_episodes

    start_time = datetime.now()
    print(f'Starting at {start_time}')
    print(f'Evaluating for {eval_timesteps} timesteps')

    with torch.no_grad():
        test_rewards = model.evaluate_in_context(vec_env=envs, eval_timesteps=eval_timesteps)['reward_episode']
        result_path = path.join(ckpt_dir, 'eval_result.npy')

    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    with open(result_path, 'wb') as f:
        np.save(f, test_rewards)

    #for i in range(len(test_env_args)):
        #print(f'Env {i} (goal={test_env_args[i]}): {test_rewards[i]}')

    print("Mean reward per environment:", test_rewards.mean(axis=1))
    print("Overall mean reward: ", test_rewards.mean())
    print("Std deviation: ", test_rewards.std())
    print(f'Results saved to {result_path}')
