import os
import signal
from datetime import datetime
import yaml
import multiprocessing
import argparse

from env import SAMPLE_ENVIRONMENT, make_env, Darkroom, DarkroomPermuted, DarkKeyToDoor
from algorithm import ALGORITHM, HistoryLoggerCallback
import h5py
import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from utils import get_config, get_traj_file_name


# Global flag for graceful shutdown
_shutdown_requested = False


def _worker_init():
    """Initialize worker process to ignore SIGINT (let parent handle it)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)



def worker(arg, config, traj_dir, env_idx, history, file_name):
    # limit CPU threads in worker to avoid oversubscription
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    
    n_stack = config.get('n_stack', 1)
    
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    elif config['env'] == 'dark_key_to_door':
        # arg is (key_x, key_y, goal_x, goal_y)
        key = arg[:2]
        goal = arg[2:]
        env = DummyVecEnv([make_env(config, key=key, goal=goal)] * config['n_stream'])
        # Apply VecFrameStack for dark_key_to_door environment
        if n_stack > 1:
            env = VecFrameStack(env, n_stack=n_stack)
    else:
        raise ValueError(f'Invalid environment: {config["env"]}')
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx

    config['device'] = 'cpu'
    # Disable tensorboard logging
    config['tensorboard_log'] = None

    alg = ALGORITHM[alg_name](config, env, seed)
    callback = HistoryLoggerCallback(config['env'], env_idx, history, n_stack=n_stack)

    print(f'Worker {env_idx} starting training on env {arg} with seed {seed}...')
    
    alg.learn(total_timesteps=config['total_source_timesteps'],
              callback=callback,
              log_interval=None,
              reset_num_timesteps=True,
              progress_bar=False)
    env.close()

    print(f'Worker {env_idx} finished training.')


if __name__ == '__main__':
    # Use 'spawn' start method for consistency (fork can break CUDA/threads)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # start method already set (possible on some platforms/runs)
        pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='darkroom',
                       help='Environment name: darkroom or dark_key_to_door')
    parser.add_argument('--n-stack', type=int, default=8,
                       help='Number of frames to stack (only for dark_key_to_door)')
    args = parser.parse_args()
    
    # Determine config files based on environment
    env_config_map = {
        'darkroom': ('darkroom', 'ppo_darkroom'),
        'dark_key_to_door': ('dark_key_to_door', 'ppo_dark_key_to_door'),
    }
    if args.env not in env_config_map:
        raise ValueError(f'Unknown environment: {args.env}')
    env_cfg, alg_cfg = env_config_map[args.env]
    
    config = get_config(f"config/env/{env_cfg}.yaml")
    config.update(get_config(f"config/algorithm/{alg_cfg}.yaml"))
    
    # Add n_stack to config for dark_key_to_door
    if args.env == 'dark_key_to_door':
        config['n_stack'] = args.n_stack

    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
        
    traj_dir = 'datasets'

    # Shuffle tasks for a diverse train/test split
    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=True)
    total_args = train_args + test_args
    n_envs = len(total_args)

    file_name = get_traj_file_name(config)
    path = f'datasets/{file_name}.hdf5'
    
    start_time = datetime.now()
    print(f'Training started at {start_time}')

    # Use Manager for shared history; allow clean shutdown on Ctrl+C
    with multiprocessing.Manager() as manager:
        history = manager.dict()

        pool = None
        try:
            # Create pool with initializer to make workers ignore SIGINT
            pool = multiprocessing.Pool(
                processes=config['n_process'],
                initializer=_worker_init
            )

            # Prepare arguments once to avoid lambda capture issues
            tasks = [(total_args[i], config, traj_dir, i, history, file_name) for i in range(n_envs)]

            # Run workers async to handle SIGINT
            result = pool.starmap_async(worker, tasks)
            
            # Poll for completion to allow interrupts
            while not result.ready():
                try:
                    result.get(timeout=1.0)
                except multiprocessing.TimeoutError:
                    continue

            # close normally
            pool.close()
            pool.join()

        except KeyboardInterrupt:
            print('\nKeyboardInterrupt received — terminating workers...')
            if pool is not None:
                pool.terminate()
                pool.join()
            print('Workers terminated.')
        finally:
            # Ensure pool is cleaned up if something else went wrong
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    pass

        # Save collected history (skip missing entries)
        print(f'Saving history to {path}...')
        try:
            with h5py.File(path, 'w') as f:
                for i in range(n_envs):
                    if str(i) in history:
                        env_data = history[str(i)] if isinstance(history[str(i)], dict) else history[i]
                    elif i in history:
                        env_data = history[i]
                    else:
                        # no data collected for this env index
                        continue

                    env_group = f.create_group(f'{i}')
                    for key, value in env_data.items():
                        env_group.create_dataset(key, data=value)
            print(f'History saved successfully.')
        except Exception as e:
            print(f'Warning: failed to write history to {path}: {e}')

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    