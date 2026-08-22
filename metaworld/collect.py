import os
import sys
import signal
from datetime import datetime
import yaml
import multiprocessing
import argparse
import atexit
import tempfile
import pickle
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from env import SAMPLE_ENVIRONMENT, make_env
from algorithm import ALGORITHM, HistoryLoggerCallback
import h5py
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from utils import get_config, get_traj_file_name


# Global flag for graceful shutdown
_shutdown_requested = False
_active_executor = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested, _active_executor
    
    # Prevent re-entry if signal received multiple times
    if _shutdown_requested:
        print("\n[!] Force exit requested. Killing all processes...")
        os._exit(1)
    
    print(f"\n\n[!] Received signal {signum}. Initiating graceful shutdown...")
    print("[!] Press Ctrl+C again to force exit immediately.")
    _shutdown_requested = True
    
    # Kill child processes first
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        # Wait briefly for graceful termination
        gone, alive = psutil.wait_procs(children, timeout=2)
        # Force kill any remaining
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except ImportError:
        # Fallback without psutil
        for p in multiprocessing.active_children():
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for p in multiprocessing.active_children():
            try:
                p.kill()
            except Exception:
                pass
    except Exception as e:
        print(f"[!] Error during cleanup: {e}")
    
    # Shutdown executor
    if _active_executor is not None:
        try:
            _active_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    print("[!] Shutdown complete. Exiting.")
    os._exit(1)  # Force exit to avoid hanging


def init_worker():
    """Initialize worker process to ignore SIGINT (let parent handle it)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=False, default='./config/algorithm/ppo_ml1.yaml', help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=False, default='./config/env/ml1.yaml', help="Environment config")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Directory for history saving")
    parser.add_argument('--override', '-o', default='')
    args = parser.parse_args()
    return args


def worker(config, env_cls, task_instance, traj_dir, env_idx, file_name, temp_dir):
    """Worker function that saves results to a temp file instead of shared memory."""
    env = None
    alg = None
    temp_file = os.path.join(temp_dir, f'history_{env_idx}.pkl')
    
    try:
        # Performance optimizations for PyTorch
        import torch
        
        # Choose device from config
        use_gpu = config.get('use_gpu', False)
        if use_gpu and torch.cuda.is_available():
            # For GPU: limit to specific GPU if multiple workers
            gpu_id = env_idx % torch.cuda.device_count() if torch.cuda.device_count() > 1 else 0
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
            # CPU optimizations
            torch.set_num_threads(max(1, os.cpu_count() // config['n_process']))
        
        # Grad settings for rollouts (PPO still requires gradients)
        torch.set_grad_enabled(True)
        
        n_stream = config['n_stream']
        
        # Use DummyVecEnv to avoid subprocess overhead
        env = DummyVecEnv([make_env(config, env_cls, task_instance)] * n_stream)
        
        alg_name = config['alg']
        seed = config['alg_seed'] + env_idx
        
        # Disable TensorBoard logging to reduce I/O
        use_tensorboard = config.get('use_tensorboard', False)
        log_dir = traj_dir if use_tensorboard else None
        
        # Initialize algorithm with explicit device
        alg = ALGORITHM[alg_name](config, env, seed, log_dir, device=device)

        # Use optimized callback (no Manager dict)
        callback = OptimizedHistoryCallback(
            config['env'], env_idx, 
            dim_obs=config['dim_obs'],
            preallocate_size=config['total_source_timesteps'] // n_stream + 1000
        )

        log_name = f'{file_name}_{env_idx}' if use_tensorboard else None

        print(f'[Worker {env_idx}] Starting collection with seed {seed}')

        # Execute learning algorithm
        alg.learn(
            total_timesteps=config['total_source_timesteps'],
            callback=callback,
            log_interval=config.get('log_interval', 100),  # Reduce logging frequency
            tb_log_name=log_name,
            reset_num_timesteps=True,
            progress_bar=False
        )
        
        # Save history to temp file (avoid Manager IPC overhead)
        history_data = callback.get_history()
        with open(temp_file, 'wb') as f:
            pickle.dump(history_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f'[Worker {env_idx}] Collection completed.')
        return env_idx, temp_file, True
        
    except Exception as e:
        print(f'[!] Worker {env_idx} error: {e}')
        import traceback
        traceback.print_exc()
        return env_idx, None, False
    finally:
        # Always close environment
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        
        # Delete algorithm to release model memory
        if alg is not None:
            try:
                del alg
            except Exception:
                pass
        
        # Clean up PyTorch/CUDA resources if using GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass



def save_collection_config(output_dir, file_name, config, description):
    """Save the merged PPO/environment config next to the collected HDF5 file."""
    config_path = os.path.join(output_dir, f'{file_name}_config.yaml')
    payload = dict(config)
    payload['collection_description'] = description
    payload['saved_at'] = datetime.now().isoformat(timespec='seconds')

    with open(config_path, 'w') as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f'  - Config saved to: {config_path}')


def _json_scalar(value):
    """Convert numpy scalar values to JSON-friendly Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_float(value):
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return _safe_float(values.mean())


def _task_metadata(task_instance, env_cls=None):
    metadata = {
        'env_name': getattr(task_instance, 'env_name', None),
    }
    if env_cls is not None:
        metadata['env_class'] = getattr(env_cls, '__name__', str(env_cls))
    return {key: value for key, value in metadata.items() if value is not None}


def _episode_metrics(rewards, dones, success):
    episode_success = []
    episode_returns = []
    episode_lengths = []
    final_quarter_success = []
    final_quarter_returns = []

    n_steps, n_envs = rewards.shape
    final_quarter_start = int(0.75 * n_steps)
    running_success = np.zeros(n_envs, dtype=np.bool_)
    running_returns = np.zeros(n_envs, dtype=np.float64)
    running_lengths = np.zeros(n_envs, dtype=np.int64)

    for step in range(n_steps):
        running_success |= success[step].astype(np.bool_)
        running_returns += rewards[step]
        running_lengths += 1

        done_envs = np.flatnonzero(dones[step].astype(np.bool_))
        for env_id in done_envs:
            succeeded = bool(running_success[env_id])
            ep_return = float(running_returns[env_id])
            episode_success.append(succeeded)
            episode_returns.append(ep_return)
            episode_lengths.append(int(running_lengths[env_id]))
            if step >= final_quarter_start:
                final_quarter_success.append(succeeded)
                final_quarter_returns.append(ep_return)

            running_success[env_id] = False
            running_returns[env_id] = 0.0
            running_lengths[env_id] = 0

    return {
        'episode_count': len(episode_success),
        'episode_success_rate': _safe_mean(episode_success),
        'episode_return_mean': _safe_mean(episode_returns),
        'episode_length_mean': _safe_mean(episode_lengths),
        'final_quarter_episode_count': len(final_quarter_success),
        'final_quarter_episode_success_rate': _safe_mean(final_quarter_success),
        'final_quarter_episode_return_mean': _safe_mean(final_quarter_returns),
    }


def _chunk_metrics(rewards, success, n_chunks=4):
    n_steps = rewards.shape[0]
    chunks = []
    for chunk_idx, step_indices in enumerate(np.array_split(np.arange(n_steps), n_chunks)):
        if step_indices.size == 0:
            continue
        chunk_rewards = rewards[step_indices]
        chunk_success = success[step_indices]
        chunks.append({
            'chunk': chunk_idx,
            'start_step': int(step_indices[0]),
            'end_step_exclusive': int(step_indices[-1] + 1),
            'step_success_rate': _safe_float(chunk_success.mean()),
            'reward_mean': _safe_float(chunk_rewards.mean()),
        })
    return chunks


def summarize_history_metrics(history_data, idx, task_instance=None, env_cls=None):
    rewards = np.asarray(history_data['rewards'], dtype=np.float32)
    success = np.asarray(history_data['success'], dtype=np.float32)
    dones = np.asarray(history_data['dones'], dtype=np.bool_)

    metrics = {
        'task_index': int(idx),
        'steps_per_stream': int(rewards.shape[0]),
        'n_streams': int(rewards.shape[1]) if rewards.ndim == 2 else 1,
        'total_env_steps': int(rewards.size),
        'step_success_rate': _safe_float(success.mean()),
        'reward_mean': _safe_float(rewards.mean()),
        'reward_std': _safe_float(rewards.std()),
        'reward_min': _safe_float(rewards.min()),
        'reward_max': _safe_float(rewards.max()),
        'task': _task_metadata(task_instance, env_cls),
        'chunks': _chunk_metrics(rewards, success),
    }
    metrics.update(_episode_metrics(rewards, dones, success))
    return metrics


def aggregate_collection_metrics(task_metrics, description, config):
    if not task_metrics:
        return {
            'description': description,
            'task_count': 0,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
        }

    def weighted_average(key, weight_key):
        numerator = 0.0
        denominator = 0.0
        for item in task_metrics:
            value = item.get(key)
            weight = item.get(weight_key, 0)
            if value is None or weight <= 0:
                continue
            numerator += float(value) * float(weight)
            denominator += float(weight)
        return _safe_float(numerator / denominator) if denominator > 0 else None

    task_success_rates = [
        item['episode_success_rate']
        for item in task_metrics
        if item.get('episode_success_rate') is not None
    ]
    final_success_rates = [
        item['final_quarter_episode_success_rate']
        for item in task_metrics
        if item.get('final_quarter_episode_success_rate') is not None
    ]
    mean_task_final_success_rate = _safe_mean(final_success_rates)

    return {
        'description': description,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'task': config.get('task'),
        'alg': config.get('alg'),
        'alg_seed': config.get('alg_seed'),
        'n_stream': config.get('n_stream'),
        'total_source_timesteps': config.get('total_source_timesteps'),
        'task_count': len(task_metrics),
        'total_env_steps': int(sum(item['total_env_steps'] for item in task_metrics)),
        'mean_task_episode_success_rate': _safe_mean(task_success_rates),
        'min_task_episode_success_rate': _safe_float(np.min(task_success_rates)) if task_success_rates else None,
        'max_task_episode_success_rate': _safe_float(np.max(task_success_rates)) if task_success_rates else None,
        'mean_task_final_quarter_episode_success_rate': mean_task_final_success_rate,
        'final_task_success_rate': mean_task_final_success_rate,
        'step_success_rate': weighted_average('step_success_rate', 'total_env_steps'),
        'episode_success_rate': weighted_average('episode_success_rate', 'episode_count'),
        'final_quarter_episode_success_rate': weighted_average(
            'final_quarter_episode_success_rate',
            'final_quarter_episode_count',
        ),
        'reward_mean': weighted_average('reward_mean', 'total_env_steps'),
        'tasks': sorted(task_metrics, key=lambda item: item['task_index']),
    }


def save_collection_metrics(output_dir, file_name, metrics):
    metrics_path = os.path.join(output_dir, f'{file_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  - Metrics saved to: {metrics_path}')
    return metrics_path


def attach_metrics_attrs(env_group, metrics):
    for key, value in metrics.items():
        if key in ('task', 'chunks'):
            continue
        value = _json_scalar(value)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            env_group.attrs[key] = value
    for key, value in metrics.get('task', {}).items():
        env_group.attrs[f'task_{key}'] = value


class OptimizedHistoryCallback(BaseCallback):
    """Optimized callback that preallocates arrays and minimizes overhead.
    
    This is a SB3-compatible callback that stores data in preallocated numpy arrays
    instead of lists, and saves to a local file instead of using Manager dict.
    """
    
    def __init__(self, env_name, env_idx, dim_obs=11, preallocate_size=100000):
        super().__init__(verbose=0)
        
        self.env_name = env_name
        self.env_idx = env_idx
        self.dim_obs = dim_obs
        self.preallocate_size = preallocate_size
        
        # Preallocate arrays (will be trimmed at the end)
        self._idx = 0
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._dones = None
        self._success = None
        self._initialized = False
        self._n_envs = None
        self._action_dim = None
    
    def _initialize_arrays(self, n_envs, action_dim):
        """Lazy initialization once we know the dimensions."""
        self._n_envs = n_envs
        self._action_dim = action_dim
        self._states = np.zeros((self.preallocate_size, n_envs, self.dim_obs), dtype=np.float32)
        self._actions = np.zeros((self.preallocate_size, n_envs, action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.preallocate_size, n_envs), dtype=np.float32)
        self._next_states = np.zeros((self.preallocate_size, n_envs, self.dim_obs), dtype=np.float32)
        self._dones = np.zeros((self.preallocate_size, n_envs), dtype=np.bool_)
        self._success = np.zeros((self.preallocate_size, n_envs), dtype=np.float32)
        self._initialized = True
    
    def _expand_arrays(self):
        """Double array size when needed."""
        new_size = self._states.shape[0] * 2
        
        new_states = np.zeros((new_size, self._n_envs, self.dim_obs), dtype=np.float32)
        new_states[:self._idx] = self._states[:self._idx]
        self._states = new_states
        
        new_actions = np.zeros((new_size, self._n_envs, self._action_dim), dtype=np.float32)
        new_actions[:self._idx] = self._actions[:self._idx]
        self._actions = new_actions
        
        new_rewards = np.zeros((new_size, self._n_envs), dtype=np.float32)
        new_rewards[:self._idx] = self._rewards[:self._idx]
        self._rewards = new_rewards
        
        new_next_states = np.zeros((new_size, self._n_envs, self.dim_obs), dtype=np.float32)
        new_next_states[:self._idx] = self._next_states[:self._idx]
        self._next_states = new_next_states
        
        new_dones = np.zeros((new_size, self._n_envs), dtype=np.bool_)
        new_dones[:self._idx] = self._dones[:self._idx]
        self._dones = new_dones
        
        new_success = np.zeros((new_size, self._n_envs), dtype=np.float32)
        new_success[:self._idx] = self._success[:self._idx]
        self._success = new_success
    
    def _on_step(self) -> bool:
        """Called at each step (SB3 BaseCallback interface)."""
        new_obs = self.locals["new_obs"]
        if new_obs.ndim == 1:
            new_obs = new_obs.reshape(1, -1)
        
        actions = self.locals["actions"]
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        # Initialize arrays on first step
        if not self._initialized:
            self._initialize_arrays(new_obs.shape[0], actions.shape[1])
        
        # Expand arrays if needed
        if self._idx >= self._states.shape[0]:
            self._expand_arrays()
        
        # Store data directly without creating intermediate lists
        self._states[self._idx] = new_obs[:, self.dim_obs:2*self.dim_obs]
        self._next_states[self._idx] = new_obs[:, :self.dim_obs]
        self._actions[self._idx] = actions
        
        rewards = self.locals["rewards"]
        if np.isscalar(rewards):
            rewards = np.array([rewards])
        elif isinstance(rewards, np.ndarray) and rewards.ndim == 0:
            rewards = np.array([rewards.item()])
        self._rewards[self._idx] = rewards
        
        dones = self.locals["dones"]
        if np.isscalar(dones):
            dones = np.array([dones])
        elif isinstance(dones, np.ndarray) and dones.ndim == 0:
            dones = np.array([dones.item()])
        self._dones[self._idx] = dones
        
        infos = self.locals['infos']
        if isinstance(infos, dict):
            infos = [infos]
        self._success[self._idx] = [info.get('success', False) for info in infos]
        
        self._idx += 1
        return True
    
    def get_history(self):
        """Return the collected history, trimmed to actual size."""
        return {
            'states': self._states[:self._idx].copy(),
            'actions': self._actions[:self._idx].copy(),
            'rewards': self._rewards[:self._idx].copy(),
            'next_states': self._next_states[:self._idx].copy(),
            'dones': self._dones[:self._idx].copy(),
            'success': self._success[:self._idx].copy()
        }


def collect_histories(task_instances, path, file_name, config, description=""):
    """Collect histories with proper resource management and parallel execution."""
    global _active_executor, _shutdown_requested
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    start_time = datetime.now()
    print(f'Collecting {description} histories started at {start_time}')
    print(f'  - Total tasks: {len(task_instances)}')
    print(f'  - Parallel processes: {config["n_process"]}')
    print(f'  - Timesteps per task: {config["total_source_timesteps"]:,}')
    
    hdf5_path = os.path.join(path, f'{file_name}.hdf5')
    save_collection_config(path, file_name, config, description)
    
    # Determine starting index
    start_idx = 0
    with h5py.File(hdf5_path, 'a') as f:
        while f'{start_idx}' in f.keys():
            start_idx += 1
    
    if start_idx > 0:
        print(f'  - Resuming from task {start_idx}')

    collected_metrics = []
    with h5py.File(hdf5_path, 'a') as hf:
        for key in sorted(hf.keys(), key=lambda item: int(item)):
            idx = int(key)
            env_cls, task_instance = task_instances[idx]
            history_data = {
                'rewards': hf[key]['rewards'][()],
                'dones': hf[key]['dones'][()],
                'success': hf[key]['success'][()],
            }
            metrics = summarize_history_metrics(history_data, idx, task_instance, env_cls)
            attach_metrics_attrs(hf[key], metrics)
            collected_metrics.append(metrics)
    
    # Create temp directory for worker results
    temp_dir = tempfile.mkdtemp(prefix='collect_histories_')
    
    try:
        # Use ProcessPoolExecutor for better management and as_completed for non-blocking
        n_workers = min(config['n_process'], len(task_instances) - start_idx)

        if n_workers <= 0:
            print('  - All tasks already collected; regenerating metrics only.')
        else:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
                _active_executor = executor

                # Submit all remaining tasks
                futures = {}
                for i in range(start_idx, len(task_instances)):
                    if _shutdown_requested:
                        break

                    env_cls, task_instance = task_instances[i]
                    future = executor.submit(
                        worker, config, env_cls, task_instance,
                        path, i, file_name, temp_dir
                    )
                    futures[future] = i

                # Process results as they complete (no waiting for slower workers)
                completed = 0
                total = len(futures)

                for future in as_completed(futures):
                    if _shutdown_requested:
                        break

                    env_idx = futures[future]
                    try:
                        idx, temp_file, success = future.result()

                        if success and temp_file and os.path.exists(temp_file):
                            # Load from temp file and save to HDF5
                            with open(temp_file, 'rb') as f:
                                history_data = pickle.load(f)

                            with h5py.File(hdf5_path, 'a') as hf:
                                if f'{idx}' not in hf.keys():
                                    env_group = hf.create_group(f'{idx}')
                                    for key, value in history_data.items():
                                        env_group.create_dataset(key, data=value, compression='gzip', compression_opts=1)
                                    env_cls, task_instance = task_instances[idx]
                                    metrics = summarize_history_metrics(
                                        history_data, idx, task_instance, env_cls
                                    )
                                    attach_metrics_attrs(env_group, metrics)
                                    collected_metrics.append(metrics)

                            # Clean up temp file
                            os.remove(temp_file)

                            completed += 1
                            elapsed = datetime.now() - start_time
                            rate = elapsed / completed if completed > 0 else elapsed
                            eta = rate * (total - completed)
                            print(f'  Progress: {completed}/{total} tasks completed. ETA: {eta}')
                        else:
                            print(f'  [!] Task {idx} failed')

                    except Exception as e:
                        print(f'  [!] Error processing task {env_idx}: {e}')

                _active_executor = None
                
    except KeyboardInterrupt:
        print("\n[!] Interrupt received, cleaning up...")
        raise
    finally:
        # Clean up temp directory
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass
    
    end_time = datetime.now()
    print()
    print(f'Collecting {description} histories ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    metrics = aggregate_collection_metrics(collected_metrics, description, config)
    save_collection_metrics(path, file_name, metrics)
    print(
        f'  - {description} mean task episode success: '
        f'{metrics.get("mean_task_episode_success_rate")}'
    )
    print(
        f'  - {description} final-quarter episode success: '
        f'{metrics.get("final_quarter_episode_success_rate")}'
    )
    if description == "train task":
        print(
            '  - train task final success rate: '
            f'{metrics.get("final_task_success_rate")}'
        )


if __name__ == '__main__':
    # Set up signal handlers BEFORE any multiprocessing
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))    

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
    
    task = config['task']
    
    train_envs, test_envs = SAMPLE_ENVIRONMENT[config['env']](config)
        
    file_name = get_traj_file_name(config)
    
    try:
        # Collect train task histories
        train_path = f'{args.traj_dir}/{task}/'
        collect_histories(train_envs, train_path, file_name, config, description="train task")
        
        # Collect test task histories
        test_path = f'{args.traj_dir}/{task}/test/'
        collect_histories(test_envs[:10], test_path, file_name, config, description="test task")
        
    except KeyboardInterrupt:
        print("\n[!] Program interrupted by user. All resources cleaned up.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Program error: {e}")
        sys.exit(1)
