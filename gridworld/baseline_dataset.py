"""Independent DPT/IDT sampling and collation; never mutates AD/RAD data.

Reference protocol: https://github.com/jaehyeon-son/dicp/tree/main/gridworld
History arrays are read in the existing collector's (time, stream, ...) layout.
"""

from functools import lru_cache
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import get_traj_file_name


@lru_cache(maxsize=32)
def collection_task_ids(grid_size, power, seed):
    ids = list(range(grid_size ** power))
    random.Random(seed).shuffle(ids)
    return tuple(ids)


def selected_group_ids(config, mode):
    """Map the environment task split to collect.py's shuffled group IDs."""
    power = {'darkroom': 2, 'dktd': 4}[config['env']]
    ids = list(range(config['grid_size'] ** power))
    random.Random(config['env_split_seed']).shuffle(ids)
    split = round(len(ids) * config['train_env_ratio'])
    if mode == 'train':
        ids = ids[:split]
    elif mode == 'test':
        ids = ids[split:]
    elif mode != 'all':
        raise ValueError(f'Invalid dataset mode: {mode}')
    order = collection_task_ids(config['grid_size'], power,
                                config.get('collection_env_split_seed', config['env_split_seed']))
    group_for_task = {task: group for group, task in enumerate(order)}
    return [group_for_task[task] for task in ids]


def task_for_group(config, group_id):
    """collect.py numbers its shuffled task list, not canonical coordinates."""
    size = config['grid_size']
    power = {'darkroom': 2, 'dktd': 4}[config['env']]
    ids = collection_task_ids(size, power, config.get('collection_env_split_seed', config['env_split_seed']))
    return np.asarray(np.unravel_index(ids[group_id], (size,) * power))


def transition_next_states(states, actions, grid_size):
    # The existing collector stores reset observations on terminal steps.
    # Deterministic Gridworld transitions let these new baselines recover s'.
    moves = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])
    return np.clip(states + moves[actions], 0, grid_size - 1)


def optimal_labels(config, group_id, states, actions, rewards):
    """Vectorized oracle labels, validating task identity against true rewards.

    DKTD possession is BEFORE the current action, reconstructed per episode.
    Starting on the key does not imply possession until an action collects it.
    """
    task = task_for_group(config, group_id)
    next_states = transition_next_states(states, actions, config['grid_size'])
    if config['env'] == 'darkroom':
        targets = np.broadcast_to(task, states.shape)
        expected = np.all(next_states == task, -1).astype(np.int64)
    else:
        horizon = config['horizon']
        before = np.zeros(rewards.shape, dtype=np.int64)
        for start in range(0, rewards.shape[1], horizon):
            end = min(start + horizon, rewards.shape[1])
            before[:, start:end] = rewards[:, start:end].cumsum(-1) - rewards[:, start:end]
        have_key = before >= 1
        targets = np.where(have_key[..., None], task[2:], task[:2])
        expected = ((~have_key & np.all(next_states == task[:2], -1))
                    | (have_key & (before < 2) & np.all(next_states == task[2:], -1))).astype(np.int64)
    if not np.array_equal(expected, rewards):
        raise ValueError(
            f'Group {group_id}: oracle rewards disagree with history. Check collection_env_split_seed '
            '(the seed used by collect.py), environment settings, and true reward data.'
        )
    delta = targets - states
    return np.select((delta[..., 0] > 0, delta[..., 0] < 0, delta[..., 1] > 0, delta[..., 1] < 0),
                     (0, 1, 2, 3), default=4).astype(np.int64)


class BaselineDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.n_transit = int(config['n_transit'])
        arrays = {key: [] for key in ('states', 'actions', 'rewards', 'next_states')}
        labels = []
        history_path = f'{traj_dir}/{get_traj_file_name(config)}.hdf5'
        with h5py.File(history_path, 'r') as history:
            for group_id in selected_group_ids(config, mode):
                if str(group_id) not in history:
                    continue
                group = history[str(group_id)]
                states = group['states'][()].swapaxes(0, 1)[:n_stream, :source_timesteps]
                actions = group['actions'][()].T[:n_stream, :source_timesteps].astype(np.int64)
                rewards = group['rewards'][()].T[:n_stream, :source_timesteps]
                if not np.all(np.isin(rewards, [0, 1])):
                    raise ValueError('DPT/IDT require binary true Gridworld rewards')
                next_states = transition_next_states(states, actions, config['grid_size'])
                for key, value in zip(arrays, (states, actions, rewards, next_states)):
                    arrays[key].append(value)
                if isinstance(self, DPTDataset):
                    if 'optimal_actions' in group:
                        # dicp labels use (stream, time); ordinary histories use (time, stream).
                        value = group['optimal_actions'][()][:n_stream, :source_timesteps]
                        if value.shape != actions.shape:
                            raise ValueError('optimal_actions must have shape (stream, time)')
                        if not np.all((value >= 0) & (value < config['num_actions']) & (value == value.astype(np.int64))):
                            raise ValueError('optimal_actions contains invalid action IDs')
                        labels.append(value.astype(np.int64))
                    else:
                        labels.append(optimal_labels(config, group_id, states, actions, rewards))
        if not arrays['states']:
            raise ValueError(f'No {mode} task histories found in {history_path}')
        for key, chunks in arrays.items():
            setattr(self, key, np.concatenate(chunks, 0))
        if self.states.shape[1] < self.n_transit:
            raise ValueError('Source history is shorter than n_transit')
        if labels:
            self.optimal_actions = np.concatenate(labels, 0)

    def __len__(self):
        return len(self.states) * (self.states.shape[1] - self.n_transit + 1)

    def indices(self, index):
        return divmod(index, self.states.shape[1] - self.n_transit + 1)


class DPTDataset(BaselineDataset):
    def __getitem__(self, index):
        stream, start = self.indices(index)
        query = start + self.n_transit - 1
        result = {key: getattr(self, key)[stream, start:query] for key in ('states', 'actions', 'rewards', 'next_states')}
        result.update(query_states=self.states[stream, query], target_actions=self.optimal_actions[stream, query])
        if self.config.get('dynamics', False):
            result.update(query_actions=self.actions[stream, query], target_rewards=self.rewards[stream, query],
                          target_next_states=self.next_states[stream, query])
        return result


class IDTDataset(BaselineDataset):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        horizon = config['horizon']
        low = config['low_per_high']
        if low < 1 or self.n_transit % low or horizon % low:
            raise ValueError('IDT low_per_high must divide n_transit and horizon')
        if self.rewards.shape[1] % horizon:
            raise ValueError('IDT source_timesteps must contain complete episodes')
        episode_rewards = self.rewards.reshape(len(self.rewards), -1, horizon)
        rtg = np.flip(np.flip(episode_rewards, -1).cumsum(-1), -1)
        order = np.argsort(rtg[:, :, 0], axis=1, kind='stable')
        rtg = np.take_along_axis(rtg, order[..., None], axis=1)
        # Reorder whole episodes independently inside each task/source stream.
        for key in ('states', 'actions', 'rewards', 'next_states'):
            values = getattr(self, key)
            episodes = values.reshape(len(values), -1, horizon, *values.shape[2:])
            indices = order.reshape(*order.shape, *([1] * (episodes.ndim - 2)))
            setattr(self, key, np.take_along_axis(episodes, indices, 1).reshape(values.shape))
        offset = rtg[:, :, 0].max(1, keepdims=True) - rtg[:, :, 0]
        self.return_to_go = (rtg + offset[..., None]).reshape(self.rewards.shape).astype(np.float32)

    def __getitem__(self, index):
        stream, start = self.indices(index)
        return {key: getattr(self, key)[stream, start:start + self.n_transit]
                for key in ('states', 'actions', 'rewards', 'next_states', 'return_to_go')}


def baseline_collate(batch):
    # DPT packs one-hot actions inside its model; IDT needs integer IDs.
    return {key: torch.as_tensor(np.stack([item[key] for item in batch])) for key in batch[0]}


def get_baseline_data_loader(dataset, batch_size, config, shuffle=True):
    workers = config.get('num_workers', 0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=baseline_collate,
                      num_workers=workers, persistent_workers=workers > 0)


BASELINE_DATASET = {'DPT': DPTDataset, 'IDT': IDTDataset}
