from datetime import datetime

from glob import glob

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    ckpt_dir = './runs/AD-darkroom-seed0'
    ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))

    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        raise ValueError('No checkpoint found.')
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print("Evaluation goals: ", test_env_args)

    if env_name == 'darkroom':
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in test_env_args])
    elif env_name == 'dark_key_to_door':
        envs = SubprocVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in test_env_args])
    else:
        raise NotImplementedError(f'Environment not supported: {env_name}')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting at {start_time}')

    with torch.no_grad():
        test_rewards = model.evaluate_in_context(vec_env=envs, eval_timesteps=config['horizon'] * 100)['reward_episode']
        path = path.join(ckpt_dir, 'eval_result.npy')
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elpased time: {end_time - start_time}')

    envs.close()

    with open(path, 'wb') as f:
        np.save(f, test_rewards)

    for i in range(len(test_env_args)):
        print(f'Env {i} (goal={test_env_args[i]}): {test_rewards[i]}')

    print("Mean reward per environment:", test_rewards.mean(axis=1))
    print("Overall mean reward: ", test_rewards.mean())
    print("Std deviation: ", test_rewards.std())