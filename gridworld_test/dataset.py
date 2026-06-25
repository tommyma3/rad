"""
Datasets for Algorithm Distillation and Recurrent Algorithm Distillation (RAD).

Contains:
1. ADDataset - Original AD dataset with fixed context length
2. RADDataset - Variable-length dataset for RAD fine-tuning
3. CompressionPretrainDataset - Dataset for compression pre-training
"""

from torch.utils.data import Dataset, Sampler
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat
from typing import Iterator, List
import math


class ADDataset(Dataset):
    """Fixed-length s/a/r token dataset for reward-aware AD."""
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        elif self.env == 'dark_key_to_door':
            n_total_envs = min(200, config['grid_size'] ** 4)  # Limited to 200 tasks
        else:
            raise ValueError(f'Invalid env: {self.env}')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                grp = f.get(f'{i}')
                if grp is None:
                    continue  # Skip missing trajectory groups
                states.append(grp['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(grp['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(grp['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(grp['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
    
    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
            
        end_idx = transition_idx + self.n_transit
        traj = {
            'states': self.states[history_idx, transition_idx:end_idx],
            'actions': self.actions[history_idx, transition_idx:end_idx],
            'rewards': self.rewards[history_idx, transition_idx:end_idx],
            'next_states': self.next_states[history_idx, transition_idx:end_idx],
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, end_idx - 1],
                'target_rewards': self.rewards[history_idx, end_idx - 1],
            })
        
        return traj


class LengthGroupedSampler(Sampler[List[tuple]]):
    """
    Sampler that groups RAD samples by length category to minimize padding.

    Batches are formed from one category at a time, then shuffled across
    categories. Each sampled index carries its category so DataLoader workers
    can sample the matching context-length range without relying on shared
    mutable dataset state.
    """

    def __init__(self, dataset: 'RADDataset', batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._refresh_length_assignments()

    def _refresh_length_assignments(self):
        """Assign each sample index to a length category based on current distribution."""
        self.length_categories = {}

        n_samples = len(self.dataset)
        dist = self.dataset.length_distribution

        category_counts = {}
        remaining = n_samples
        categories = list(dist.keys())

        for category in categories[:-1]:
            count = int(n_samples * dist[category])
            category_counts[category] = count
            remaining -= count
        category_counts[categories[-1]] = remaining

        indices = list(range(n_samples))
        if self.shuffle:
            random.shuffle(indices)

        idx = 0
        for category in categories:
            count = category_counts[category]
            self.length_categories[category] = indices[idx:idx + count]
            idx += count

    def __iter__(self) -> Iterator[List[tuple]]:
        self._refresh_length_assignments()

        all_batches = []
        for category, indices in self.length_categories.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append((category, batch))

        if self.shuffle:
            random.shuffle(all_batches)

        for category, batch in all_batches:
            yield [(category, idx) for idx in batch]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


class RADDataset(Dataset):
    """
    Dataset for Recurrent AD (RAD) fine-tuning with variable-length sequences.

    Samples sequences of varying lengths to ensure the model learns to handle:
    - Short sequences (no compression)
    - Medium sequences (1 compression)
    - Long sequences (2-3 compressions)
    - Very long sequences (4+ compressions)

    Supports curriculum-aware sampling where length distribution can be
    dynamically updated during training to prevent catastrophic forgetting.
    """

    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']  # Environment timesteps, represented by 3 tokens each
        self.n_compress_tokens = config.get('n_compress_tokens', 40)
        self.dynamics = config['dynamics']
        
        # Context length distribution
        self.min_context = config.get('min_context_length', 50)
        self.max_context = config.get('max_context_length', 800)
        
        # Length distribution (for curriculum)
        default_distribution = {
            'short': 0.2,      # No compression (50-200)
            'medium': 0.3,     # 1 compression (250-400)
            'long': 0.3,       # 2-3 compressions (450-700)
            'very_long': 0.2,  # 4+ compressions (750-1000)
        }
        self.length_distribution = config.get('length_distribution', default_distribution).copy()
        
        # Validate distribution sums to 1.0
        self._validate_distribution(self.length_distribution)
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        elif self.env == 'dark_key_to_door':
            n_total_envs = min(200, config['grid_size'] ** 4)  # Limited to 200 tasks
        else:
            raise ValueError(f'Invalid environment: {self.env}')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                grp = f.get(f'{i}')
                if grp is None:
                    continue  # Skip missing trajectory groups
                states.append(grp['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(grp['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(grp['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(grp['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        
        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]
        self._current_batch_category = None
    
    def _validate_distribution(self, dist):
        """Validate that distribution sums to 1.0."""
        total_prob = sum(dist.values())
        assert abs(total_prob - 1.0) < 1e-6, f"Length distribution must sum to 1.0, got {total_prob}"
    
    def update_length_distribution(self, new_distribution):
        """
        Update the length distribution for curriculum-aware sampling.
        
        This allows the training script to dynamically change what sequence
        lengths are sampled based on the current curriculum stage.
        
        Args:
            new_distribution: dict with keys like 'short', 'medium', 'long', 'very_long'
                             and values summing to 1.0
        """
        self._validate_distribution(new_distribution)
        self.length_distribution = new_distribution.copy()
    
    def __len__(self):
        # Approximate: we can sample many different windows
        return self.n_histories * (self.seq_length - self.min_context)
    
    def _sample_context_length(self):
        """Sample context length from distribution supporting multi-compression training."""
        r = random.random()
        cumulative = 0
        
        max_available = self.seq_length
        
        for category, prob in self.length_distribution.items():
            cumulative += prob
            if r < cumulative:
                if category == 'short':
                    # No compression needed (fits in n_transit)
                    low, high = 20, min(self.n_transit - 1, 50)
                elif category == 'medium':
                    # 1-2 compressions
                    low, high = self.n_transit, min(150, self.max_context)
                elif category == 'long':
                    # 3-8 compressions
                    low, high = 200, min(400, self.max_context)
                elif category == 'very_long':
                    # 10+ compressions
                    low, high = 500, self.max_context
                elif category == 'extended':
                    # Maximum compressions (requires longer dataset)
                    low, high = 800, self.max_context
                else:
                    # Fallback for unknown categories
                    low, high = self.min_context, self.max_context
                
                # Clamp to available data
                high = min(high, max_available)
                low = min(low, high)  # Ensure low <= high
                
                return random.randint(low, high)
        
        # Fallback
        return random.randint(self.min_context, min(self.max_context, max_available))

    def _sample_context_length_for_category(self, category):
        """Sample context length for a specific length category."""
        max_available = self.seq_length

        if category == 'short':
            low, high = 20, min(self.n_transit - 1, 50)
        elif category == 'medium':
            low, high = self.n_transit, min(150, self.max_context)
        elif category == 'long':
            low, high = 200, min(400, self.max_context)
        elif category == 'very_long':
            low, high = 500, self.max_context
        elif category == 'extended':
            low, high = 800, self.max_context
        else:
            low, high = self.min_context, self.max_context

        high = min(high, max_available)
        low = min(low, high)

        return random.randint(low, high)
    
    def __getitem__(self, i):
        category = None
        if isinstance(i, tuple):
            category, i = i

        # Use index for reproducibility but also allow randomness
        history_idx = i % self.n_histories
        
        # Sample context length from the grouped category if provided.
        if category is not None:
            context_length = self._sample_context_length_for_category(category)
        elif self._current_batch_category is not None:
            context_length = self._sample_context_length_for_category(self._current_batch_category)
        else:
            context_length = self._sample_context_length()
        
        # Sample a valid end position for a full supervision window.
        max_end = self.seq_length
        min_end = context_length
        
        if min_end >= max_end:
            end_idx = max_end
        else:
            end_idx = random.randint(min_end, max_end)
        
        start_idx = end_idx - context_length
        
        traj = {
            'states': self.states[history_idx, start_idx:end_idx],
            'actions': self.actions[history_idx, start_idx:end_idx],
            'rewards': self.rewards[history_idx, start_idx:end_idx],
            'next_states': self.next_states[history_idx, start_idx:end_idx],
            'context_length': context_length,  # For logging/debugging
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, end_idx - 1],
                'target_rewards': self.rewards[history_idx, end_idx - 1],
            })
        
        return traj


class CompressionPretrainDataset(Dataset):
    """
    Dataset for pre-training the compression transformer.
    
    Samples fixed-length windows of transitions for reconstruction loss training.
    No query states or target actions needed - just sequences to compress and reconstruct.
    """
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.window_size = config['n_transit']  # Environment timesteps to compress
        self.dynamics = config['dynamics']
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        elif self.env == 'dark_key_to_door':
            n_total_envs = min(200, config['grid_size'] ** 4)  # Limited to 200 tasks
        else:
            raise ValueError(f'Invalid env: {self.env}')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                grp = f.get(f'{i}')
                if grp is None:
                    continue  # Skip missing trajectory groups
                states.append(grp['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(grp['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(grp['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(grp['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        
        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]
    
    def __len__(self):
        return self.n_histories * (self.seq_length - self.window_size)
    
    def __getitem__(self, i):
        history_idx = i % self.n_histories
        
        # Random window start
        max_start = self.seq_length - self.window_size
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + self.window_size
        
        traj = {
            'states': self.states[history_idx, start_idx:end_idx],
            'actions': self.actions[history_idx, start_idx:end_idx],
            'rewards': self.rewards[history_idx, start_idx:end_idx],
            'next_states': self.next_states[history_idx, start_idx:end_idx],
        }
        
        return traj
