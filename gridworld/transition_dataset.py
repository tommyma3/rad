"""Source-action endpoint sampling exclusive to the tokenization ablation.

Read the SAME groups/streams as ADDataset/RADDataset. No oracle relabeling or
task-split migration is performed. Index tuples carry their lengths explicitly,
so worker prefetching cannot retain a stale curriculum distribution.
"""

from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from baseline_dataset import transition_next_states
from dataset import ADDataset
from model.transition_ad import MemorySchedule, validate_config


def curriculum_stage(config, step):
    stage = {'max_compressions': config.get('max_compressions'),
             'length_distribution': config.get('length_distribution', {'short': 1.})}
    for entry in sorted(config.get('curriculum_schedule', []), key=lambda item: item['step']):
        if step >= entry['step']:
            stage = {**stage, **entry}
    return stage


class TransitionDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        validate_config(config)
        self.config = dict(config)
        source = ADDataset(config, traj_dir, mode, n_stream, source_timesteps)
        self.states, self.actions, self.rewards = source.states, source.actions, source.rewards
        # The collector's terminal next_states contain reset observations.
        self.next_states = transition_next_states(self.states, self.actions.astype(np.int64), config['grid_size'])
        self.n_histories, self.seq_length = self.actions.shape
        if not self.n_histories or self.seq_length < 2:
            raise ValueError('Need source histories with at least two decisions')

    def __len__(self):
        return self.n_histories * self.seq_length

    def __getitem__(self, index):
        stream, start, length = index
        end = start + length
        if not (0 <= stream < self.n_histories and 0 <= start <= end < self.seq_length):
            raise IndexError('Endpoint must be a recorded decision strictly after its history')
        result = {key: getattr(self, key)[stream, start:end] for key in
                  ('states', 'actions', 'rewards', 'next_states')}
        result.update(query_states=self.states[stream, end], target_actions=self.actions[stream, end])
        return result


def exact_length_collate(examples):
    """Return an unpadded list of microbatches; weight their losses by size."""
    groups = defaultdict(list)
    for example in examples:
        groups[len(example['states'])].append(example)
    return [{key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}
            for _, rows in sorted(groups.items())]


class EndpointBatchSampler(Sampler):
    """One deterministic logical batch per optimizer update, with varied lengths.

    Length choices match across DDP ranks; stream/window choices differ. This
    keeps the number and order of microbatch forwards identical across ranks.
    A checkpoint's next update number is sufficient to restore data sampling.
    """

    def __init__(self, dataset, config, batch_size, start_step, stop_step, rank=0, pretrain=False):
        self.dataset, self.config = dataset, dict(config)
        self.batch_size, self.start_step, self.stop_step = batch_size, start_step, stop_step
        self.rank, self.pretrain = rank, pretrain
        if batch_size < 1 or not 0 <= start_step <= stop_step:
            raise ValueError('Invalid sampler batch size or update interval')
        self.micro_count = min(batch_size, int(config.get('lengths_per_batch', 4)))
        if self.micro_count < 1:
            raise ValueError('lengths_per_batch must be positive')
        self.schedule = None
        if config['model'] == 'RAD_DPT':
            self.schedule = MemorySchedule(int(config['policy_token_budget']), int(config['n_compress_tokens']),
                                           int(config['short_memory_keep']), config.get('always_use_latent_prefix', False))
        self._length_cache = {}

    def __len__(self):
        return self.stop_step - self.start_step

    def length_groups(self, step):
        if self.pretrain:
            length = int(self.config['pretrain'].get('n_transit', self.config['n_transit']))
            if not 1 <= length < self.dataset.seq_length:
                raise ValueError('Pretraining window must fit the source history')
            return {'pretrain': [length]}, {'pretrain': 1.}
        if self.schedule is None:
            maximum = min(int(self.config['policy_token_budget']) - 1, self.dataset.seq_length - 1)
            return {'ad': list(range(maximum + 1))}, {'ad': 1.}
        stage = curriculum_stage(self.config, step)
        maximum = min(int(self.config['max_context_length']), self.dataset.seq_length - 1)
        minimum = int(self.config.get('min_context_length', 0))
        cap = stage['max_compressions']
        cache_key = (minimum, maximum, cap)
        if cache_key not in self._length_cache:
            groups = defaultdict(list)
            for length in range(minimum, maximum + 1):
                count, _ = self.schedule.state_after(length)
                if cap is None or count <= cap:
                    category = 'short' if length <= 50 else 'medium' if length <= 150 else 'long' if length <= 400 else 'very_long'
                    groups[category].append(length)
            self._length_cache[cache_key] = dict(groups)
        groups = self._length_cache[cache_key]
        weights = stage['length_distribution']
        if any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.) > 1e-6:
            raise ValueError('Length distribution must be nonnegative and sum to one')
        groups = {key: values for key, values in groups.items() if weights.get(key, 0) > 0}
        if not groups:
            raise ValueError('No lengths permitted by the source, distribution, and curriculum')
        return groups, weights

    def batch_at(self, step):
        seed = int(self.config.get('seed', 42)) + 1_000_003 * step
        length_rng = random.Random(seed)
        data_rng = random.Random(seed + 97_409 * (self.rank + 1))
        groups, weights = self.length_groups(step)
        categories = list(groups)
        result = []
        for micro in range(self.micro_count):
            category = length_rng.choices(categories, [weights[key] for key in categories])[0]
            # Sample both compression depths and positions in their refill cycle.
            lengths = groups[category]
            if self.schedule is not None and not self.pretrain:
                by_count = defaultdict(list)
                for length in lengths:
                    by_count[self.schedule.state_after(length)[0]].append(length)
                lengths = length_rng.choice(list(by_count.values()))
            length = length_rng.choice(lengths)
            size = self.batch_size // self.micro_count + int(micro < self.batch_size % self.micro_count)
            for _ in range(size):
                stream = data_rng.randrange(self.dataset.n_histories)
                start = data_rng.randrange(self.dataset.seq_length - length)
                result.append((stream, start, length))
        return result

    def __iter__(self):
        for step in range(self.start_step, self.stop_step):
            yield self.batch_at(step)
