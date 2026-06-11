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
from typing import Iterator, List, Optional
import math


class ADDataset(Dataset):
    """Original AD dataset with fixed context length for Meta-world."""
    
    def __init__(self, config, traj_dir, mode='train', n_seed=None, n_stream=None, source_timesteps=None):
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.config = config
        
        states = []
        actions = []
        rewards = []
        next_states = []
        
        if mode == 'train':
            file_path = f'{traj_dir}/{get_traj_file_name(config)}.hdf5'
        elif mode == 'test':
            file_path = f'{traj_dir}/test/{get_traj_file_name(config)}.hdf5'
        else:
            raise ValueError('Invalid mode')

        with h5py.File(file_path, 'r') as f:
            for i in range(n_seed):
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])                    
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)

    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)

    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
        
        traj = {
            'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
            'target_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
        }

        return traj


class LengthGroupedSampler(Sampler[List[int]]):
    """
    Sampler that groups samples by length category to minimize padding.
    
    Samples are pre-assigned to length categories, and batches are formed
    by drawing samples from the same category. This ensures minimal padding
    within each batch while maintaining the desired length distribution.
    """
    
    def __init__(
        self, 
        dataset: 'RADDataset',
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Pre-generate length categories for all indices
        self._refresh_length_assignments()
    
    def _refresh_length_assignments(self):
        """Assign each sample index to a length category based on current distribution."""
        self.length_categories = {}  # category -> list of indices
        
        n_samples = len(self.dataset)
        dist = self.dataset.length_distribution
        
        # Calculate how many samples per category
        category_counts = {}
        remaining = n_samples
        categories = list(dist.keys())
        
        for i, cat in enumerate(categories[:-1]):
            count = int(n_samples * dist[cat])
            category_counts[cat] = count
            remaining -= count
        # Last category gets the remainder
        category_counts[categories[-1]] = remaining
        
        # Assign indices to categories
        indices = list(range(n_samples))
        if self.shuffle:
            random.shuffle(indices)
        
        idx = 0
        for cat in categories:
            count = category_counts[cat]
            self.length_categories[cat] = indices[idx:idx + count]
            idx += count
    
    def __iter__(self) -> Iterator[List[int]]:
        # Refresh assignments each epoch to get different samples
        self._refresh_length_assignments()
        
        # Collect all batches from each category
        all_batches = []
        
        for category, indices in self.length_categories.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            
            # Form batches from this category
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    # Tag batch with category for the dataset to use
                    all_batches.append((category, batch))
        
        # Shuffle batches across categories
        if self.shuffle:
            random.shuffle(all_batches)
        
        for category, batch in all_batches:
            # Tell dataset which category this batch is from
            self.dataset._current_batch_category = category
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
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

    When used with LengthGroupedSampler, samples in the same batch will have
    similar lengths, minimizing padding waste.
    """

    def __init__(self, config, traj_dir, mode='train', n_seed=None, n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']  # Max AD sequence length
        self.n_compress_tokens = config.get('n_compress_tokens', 32)
        
        # Context length distribution
        self.min_context = config.get('min_context_length', 50)
        self.max_context = config.get('max_context_length', 2000)
        
        # Length distribution (used for curriculum)
        default_distribution = {
            'short': 0.2,      # No compression
            'medium': 0.3,     # 1 compression
            'long': 0.3,       # 2-3 compressions
            'very_long': 0.2,  # 4+ compressions
        }
        self.length_distribution = config.get('length_distribution', default_distribution).copy()
        
        # Validate distribution sums to 1.0
        self._validate_distribution(self.length_distribution)

        states = []
        actions = []
        rewards = []
        next_states = []
        
        if mode == 'train':
            file_path = f'{traj_dir}/{get_traj_file_name(config)}.hdf5'
        elif mode == 'test':
            file_path = f'{traj_dir}/test/{get_traj_file_name(config)}.hdf5'
        else:
            raise ValueError('Invalid mode')

        with h5py.File(file_path, 'r') as f:
            for i in range(n_seed):
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        
        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]
        
        # For grouped batching
        self._current_batch_category = None
    
    def _validate_distribution(self, dist):
        """Validate that distribution sums to 1.0."""
        total_prob = sum(dist.values())
        assert abs(total_prob - 1.0) < 1e-6, f"Length distribution must sum to 1.0, got {total_prob}"
    
    def update_length_distribution(self, new_distribution):
        """
        Update the length distribution for curriculum-aware sampling.
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
        
        # Maximum available context (leave room for query state)
        max_available = self.seq_length - 1
        
        for category, prob in self.length_distribution.items():
            cumulative += prob
            if r < cumulative:
                if category == 'short':
                    # No compression needed (fits in n_transit)
                    low, high = 20, min(self.n_transit - 1, 100)
                elif category == 'medium':
                    # 1-2 compressions
                    low, high = self.n_transit, min(400, self.max_context)
                elif category == 'long':
                    # 3-8 compressions
                    low, high = 500, min(1000, self.max_context)
                elif category == 'very_long':
                    # 10+ compressions
                    low, high = 1200, self.max_context
                elif category == 'extended':
                    # Maximum compressions
                    low, high = 1500, self.max_context
                else:
                    # Fallback for unknown categories
                    low, high = self.min_context, self.max_context
                
                # Clamp to available data
                high = min(high, max_available)
                low = min(low, high)
                
                return random.randint(low, high)
        
        # Fallback
        return random.randint(self.min_context, min(self.max_context, max_available))
    
    def _sample_context_length_for_category(self, category):
        """Sample context length for a specific category."""
        max_available = self.seq_length - 1
        
        if category == 'short':
            low, high = 20, min(self.n_transit - 1, 100)
        elif category == 'medium':
            low, high = self.n_transit, min(400, self.max_context)
        elif category == 'long':
            low, high = 500, min(1000, self.max_context)
        elif category == 'very_long':
            low, high = 1200, self.max_context
        elif category == 'extended':
            low, high = 1500, self.max_context
        else:
            low, high = self.min_context, self.max_context
        
        # Clamp to available data
        high = min(high, max_available)
        low = min(low, high)
        
        return random.randint(low, high)

    def __getitem__(self, i):
        # Use index for reproducibility but also allow randomness
        history_idx = i % self.n_histories
        
        # Sample context length - use category if set by LengthGroupedSampler
        if self._current_batch_category is not None:
            context_length = self._sample_context_length_for_category(self._current_batch_category)
        else:
            context_length = self._sample_context_length()
        
        # Sample a valid end position (where we predict action)
        max_end = self.seq_length - 1
        min_end = context_length
        
        if min_end >= max_end:
            end_idx = max_end
        else:
            end_idx = random.randint(min_end, max_end)
        
        start_idx = end_idx - context_length
        
        traj = {
            'query_states': self.states[history_idx, end_idx],
            'target_actions': self.actions[history_idx, end_idx],
            'states': self.states[history_idx, start_idx:end_idx],
            'actions': self.actions[history_idx, start_idx:end_idx],
            'rewards': self.rewards[history_idx, start_idx:end_idx],
            'next_states': self.next_states[history_idx, start_idx:end_idx],
            'context_length': context_length,  # For logging/debugging
        }
        
        return traj


class CompressionPretrainDataset(Dataset):
    """
    Dataset for pre-training the compression transformer.
    
    Samples fixed-length windows of transitions for reconstruction loss training.
    """
    
    def __init__(self, config, traj_dir, mode='train', n_seed=None, n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.window_size = config['n_transit'] - 1  # Size of sequences to compress

        states = []
        actions = []
        rewards = []
        next_states = []
        
        if mode == 'train':
            file_path = f'{traj_dir}/{get_traj_file_name(config)}.hdf5'
        elif mode == 'test':
            file_path = f'{traj_dir}/test/{get_traj_file_name(config)}.hdf5'
        else:
            raise ValueError('Invalid mode')

        with h5py.File(file_path, 'r') as f:
            for i in range(n_seed):
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                    
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