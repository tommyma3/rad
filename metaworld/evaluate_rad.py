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

from env import get_ml1_test_env_fns
from model import MODEL
from utils import normalize_compiled_state_dict
from stable_baselines3.common.vec_env import DummyVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
CHECKPOINT_FORMAT = 'metaworld-sar-v1'


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
    if ckpt.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'{ckpt_path} uses the legacy packed-transition checkpoint format')
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(normalize_compiled_state_dict(ckpt['model']), strict=False)
    model.eval()

    print(f"Model: {model_name}")
    print(f"Task: {config['task']}")
    print(f"Max sequence length: {config['n_transit']}")
    print(f"Compression tokens: {config.get('n_compress_tokens', 'N/A')}")
    print(f"Latent update mode: {config.get('latent_update_mode', 'replace')}")

    test_envs = get_ml1_test_env_fns(config, max_envs_per_task=None)

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
        eval_output = model.evaluate_in_context(
            vec_env=envs, 
            eval_timesteps=eval_timesteps
        )
        test_rewards = eval_output['reward_episode']
        if 'success' in eval_output.keys():
            test_success = eval_output['success']
        else:
            test_success = None
        # RAD-specific metrics
        total_compressions = eval_output.get('total_compressions', 0)
        compression_events = eval_output.get('compression_events', [])
        result_path = path.join(ckpt_dir, 'eval_result.npy')
        success_result_path = path.join(ckpt_dir, 'eval_success.npy')
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    # Save full results
    with open(result_path, 'wb') as f:
        np.save(f, eval_output['reward_episode'])
    if test_success is not None:
        with open(success_result_path, 'wb') as f:
            np.save(f, test_success)

    print(f'\nResults saved to {result_path}')

    # Print summary statistics
    print('\n=== Test Environments ===')
    print(f'Mean reward per env: {test_rewards.mean(axis=1)}')
    print(f'Overall mean reward: {test_rewards.mean():.4f}')
    print(f'Std deviation: {test_rewards.std():.4f}')
    if test_success is not None:
        print(f'Success rate (max per env): {test_success.max(axis=1).mean():.4f}')

    print('\n=== RAD Metrics ===')
    print(f'Total compressions: {total_compressions}')
    if compression_events:
        print(f'Compression events: {compression_events}')
