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
from pathlib import Path


def _trajectory_path(config, traj_dir, mode):
    if mode not in ('train', 'test'):
        raise ValueError(f'Invalid Meta-World dataset mode: {mode}')
    root = Path(traj_dir)
    if root.name != config['task']:
        root = root / config['task']
    if mode == 'test':
        root = root / 'test'
    return root / f'{get_traj_file_name(config)}.hdf5'


def _load_trajectory_arrays(config, traj_dir, mode, n_seed, n_stream, source_timesteps):
    states, actions, rewards, next_states = [], [], [], []
    file_path = _trajectory_path(config, traj_dir, mode)
    with h5py.File(file_path, 'r') as f:
        group_ids = sorted((int(key) for key in f.keys()))
        if n_seed is not None:
            group_ids = group_ids[:n_seed]
        for group_id in group_ids:
            group = f[str(group_id)]
            states.append(
                group['states'][()].transpose(1, 0, 2)[
                    :n_stream, :source_timesteps, :config['dim_obs']
                ]
            )
            actions.append(
                group['actions'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps]
            )
            rewards.append(
                group['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps]
            )
            next_states.append(
                group['next_states'][()].transpose(1, 0, 2)[
                    :n_stream, :source_timesteps, :config['dim_obs']
                ]
            )

    if not states:
        raise ValueError(f'No trajectory groups found in {file_path}')
    return tuple(np.concatenate(values, axis=0) for values in (states, actions, rewards, next_states))


class ADDataset(Dataset):
    """Fixed-length s/a/r token dataset for reward-aware AD."""
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None, n_seed=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config.get('dynamics', False)
        if n_seed is None:
            n_seed = config.get('train_n_seed') if mode == 'train' else config.get('test_n_seed')
        self.states, self.actions, self.rewards, self.next_states = _load_trajectory_arrays(
            config, traj_dir, mode, n_seed, n_stream, source_timesteps
        )
    
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


class CompressionBucketBatchSampler(Sampler[List[tuple]]):
    """
    Batch sampler for RAD fixed compression-count training.

    Each batch uses one compression-count bucket, so every item has the same
    context length and the collate path does not pad mixed sequence lengths.
    The bucket distribution is read lazily from the dataset on each yielded
    batch, so curriculum updates take effect without waiting for a full epoch.
    """

    def __init__(self, dataset: 'RADDataset', batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[tuple]]:
        n_samples = len(self.dataset)
        n_batches = len(self)

        for batch_idx in range(n_batches):
            if self.drop_last or batch_idx < n_batches - 1:
                curr_batch_size = self.batch_size
            else:
                remainder = n_samples % self.batch_size
                curr_batch_size = remainder if remainder > 0 else self.batch_size

            bucket = self.dataset.sample_compression_bucket()
            if self.shuffle:
                batch = [random.randrange(n_samples) for _ in range(curr_batch_size)]
            else:
                start = batch_idx * self.batch_size
                batch = [idx % n_samples for idx in range(start, start + curr_batch_size)]

            yield [(bucket, idx) for idx in batch]

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

    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None, n_seed=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']  # Environment timesteps, represented by 3 tokens each
        self.n_compress_tokens = config.get('n_compress_tokens', 40)
        if self.n_compress_tokens % 3 != 0:
            raise ValueError('n_compress_tokens must be divisible by 3 for fixed RAD buckets')
        self.compress_timesteps = self.n_compress_tokens // 3
        self.always_use_latent_prefix = config.get('always_use_latent_prefix', False)
        self.short_memory_keep = config.get('short_memory_keep', max(1, self.compress_timesteps))
        self.max_compressions = config.get('max_compressions', None)
        self.dynamics = config.get('dynamics', False)
        
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
        
        if n_seed is None:
            n_seed = config.get('train_n_seed') if mode == 'train' else config.get('test_n_seed')
        self.states, self.actions, self.rewards, loaded_next_states = _load_trajectory_arrays(
            config, traj_dir, mode, n_seed, n_stream, source_timesteps
        )
        self.next_states = loaded_next_states if self.dynamics else None
        
        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]
        self._current_batch_category = None

        self.max_bucket_context = min(self.max_context, self.seq_length)
    
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

    def update_max_compressions(self, max_compressions):
        """Update the maximum compression bucket allowed by the curriculum."""
        self.max_compressions = max_compressions

    def _first_memory_capacity(self):
        """Environment timesteps available before the first compression."""
        reserved = self.compress_timesteps if self.always_use_latent_prefix else 0
        return max(1, self.n_transit - reserved)

    def _compressed_memory_capacity(self):
        """Environment timesteps available next to a real latent prefix."""
        return max(1, self.n_transit - self.compress_timesteps)

    def _raw_bucket_length_for_compressions(self, n_compressions):
        """
        Maximum context length that ends with n_compressions and a full recent
        memory window under RAD's recurrent compression rule.
        """
        if n_compressions < 0:
            raise ValueError(f'n_compressions must be non-negative, got {n_compressions}')

        first_capacity = self._first_memory_capacity()
        if n_compressions == 0:
            return first_capacity

        refill = max(1, self._compressed_memory_capacity() - self.short_memory_keep)
        return first_capacity + n_compressions * refill

    def _bucket_length_for_compressions(self, n_compressions):
        return min(self._raw_bucket_length_for_compressions(n_compressions), self.max_bucket_context)

    def _available_compression_buckets(self):
        """Return valid compression-count buckets for current data/config limits."""
        buckets = []
        bucket = 0
        while True:
            length = self._raw_bucket_length_for_compressions(bucket)
            if length > self.max_bucket_context:
                break
            buckets.append(bucket)

            bucket += 1

        if self.max_compressions is not None:
            buckets = [bucket for bucket in buckets if bucket <= self.max_compressions]

        return buckets or [0]

    def _compression_bucket_distribution(self):
        """
        Convert the existing broad curriculum length distribution into exact
        compression-count buckets. Probabilities for unavailable harder ranges
        are folded into the hardest currently allowed bucket.
        """
        buckets = self._available_compression_buckets()
        max_bucket = max(buckets)
        bucket_probs = {bucket: 0.0 for bucket in buckets}

        category_ranges = {
            'short': (0, 0),
            'medium': (1, 2),
            'long': (3, 8),
            'very_long': (9, None),
            'extended': (9, None),
        }

        for category, prob in self.length_distribution.items():
            if prob <= 0:
                continue

            low, high = category_ranges.get(category, (0, max_bucket))
            high = max_bucket if high is None else min(high, max_bucket)
            low = min(low, max_bucket)

            selected = [bucket for bucket in buckets if low <= bucket <= high]
            if not selected:
                selected = [max_bucket]

            share = prob / len(selected)
            for bucket in selected:
                bucket_probs[bucket] += share

        total = sum(bucket_probs.values())
        if total <= 0:
            bucket_probs[max_bucket] = 1.0
            return bucket_probs

        return {bucket: prob / total for bucket, prob in bucket_probs.items()}

    def sample_compression_bucket(self):
        """Sample a compression-count bucket from the current curriculum state."""
        dist = self._compression_bucket_distribution()
        r = random.random()
        cumulative = 0.0

        for bucket, prob in sorted(dist.items()):
            cumulative += prob
            if r < cumulative:
                return bucket

        return max(dist)
    
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

    def _sample_context_length_for_compressions(self, n_compressions):
        """Return the fixed context length for a compression-count bucket."""
        return self._bucket_length_for_compressions(n_compressions)
    
    def __getitem__(self, i):
        category = None
        if isinstance(i, tuple):
            category, i = i

        # Use index for reproducibility but also allow randomness
        history_idx = i % self.n_histories
        
        # Sample context length from the grouped category if provided.
        if isinstance(category, int):
            context_length = self._sample_context_length_for_compressions(category)
        elif category is not None:
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
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None, n_seed=None):
        self.config = config
        self.env = config['env']
        self.window_size = config['n_transit']  # Environment timesteps to compress
        self.dynamics = config.get('dynamics', False)
        if n_seed is None:
            n_seed = config.get('train_n_seed') if mode == 'train' else config.get('test_n_seed')
        self.states, self.actions, self.rewards, self.next_states = _load_trajectory_arrays(
            config, traj_dir, mode, n_seed, n_stream, source_timesteps
        )
        
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
