from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
import torch
from typing import Any, Tuple
import random
import itertools


def map_dark_states(states, grid_size):
    return torch.sum(states * torch.tensor((grid_size, 1), device=states.device, requires_grad=False), dim=-1)


def map_dark_states_inverse(index, grid_size):
    return torch.stack((index // grid_size, index % grid_size), dim=-1)


def sample_darkroom(config, shuffle=True):
    goals = [np.array([i, j]) for i in range(config['grid_size']) for j in range(config['grid_size'])]

    if shuffle:
        random.seed(config['env_split_seed'])
        random.shuffle(goals)

    n_train_envs = round(config['grid_size'] ** 2 * config['train_env_ratio'])

    train_goals = goals[:n_train_envs]
    test_goals = goals[n_train_envs:]

    return train_goals, test_goals


def sample_darkroom_permuted(config, shuffle=True):
    perms = list(range(120))

    if shuffle:
        random.seed(config['env_split_seed'])
        random.shuffle(perms)

    n_train_envs = round(120 * config['train_env_ratio'])

    train_perms = perms[:n_train_envs]
    test_perms = perms[n_train_envs:]

    return train_perms, test_perms


class Darkroom(gym.Env):
    def __init__(self, config, **kwargs):
        super(Darkroom, self).__init__()
        self.grid_size = config['grid_size']
        if 'goal' in kwargs:
            self.goal = kwargs['goal']
        self.horizon = config['horizon']
        self.dim_obs = 2
        self.dim_action = 1
        self.num_action = 5
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(self.dim_obs,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_action)
        
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        self.current_step = 0

        center = self.grid_size // 2
        self.state = np.array([center, center])

        return self.state, {}

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        s = np.array(self.state)
        a = action

        # Action handling
        if a == 0:
            s[0] += 1
        elif a == 1:
            s[0] -= 1
        elif a == 2:
            s[1] += 1
        elif a == 3:
            s[1] -= 1

        s = np.clip(s, 0, self.grid_size - 1)
        self.state = s

        reward = 1 if np.array_equal(s, self.goal) else 0
        self.current_step += 1
        done = self.current_step >= self.horizon
        info = {}
        return s.copy(), reward, done, done, info
    
    def get_optimal_action(self, state):
        if state[0] < self.goal[0]:
            a = 0
        elif state[0] > self.goal[0]:
            a = 1
        elif state[1] < self.goal[1]:
            a = 2
        elif state[1] > self.goal[1]:
            a = 3
        else:
            a = 4
            
        return a
    
    def transit(self, s, a):
        if a == 0:
            s[0] += 1
        elif a == 1:
            s[0] -= 1
        elif a == 2:
            s[1] += 1
        elif a == 3:
            s[1] -= 1
        elif a == 4:
            pass
        else:
            raise ValueError('Invalid action')
        
        s = np.clip(s, 0, self.grid_size - 1)

        if np.all(s == self.goal):
            r = 1
        else:
            r = 0
            
        return s, r
    
    def get_max_return(self):
        center = self.grid_size // 2
        return (self.horizon + 1 - np.sum(np.absolute(self.goal - np.array([center, center])))).clip(0, self.horizon)
    
    
class DarkroomPermuted(Darkroom):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        self.perm_idx = kwargs['perm_idx']
        self.goal = np.array([self.grid_size-1, self.grid_size-1])
        
        assert self.perm_idx < 120     # 5! permutations in darkroom
        
        actions = np.arange(self.action_space.n)
        permutations = list(itertools.permutations(actions))
        self.perm = permutations[self.perm_idx]

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        self.current_step = 0

        self.state = np.array([0, 0])

        return self.state, {}
    
    def step(self, action):
        return super().step(self.perm[action])
    
    def transit(self, s, a):
        return super().transit(s, self.perm[a])

    def get_optimal_action(self, state):
        action = super().get_optimal_action(state)
        return self.perm.index(action)
    
    def get_max_return(self):
        return (self.horizon + 1 - np.sum(np.absolute(self.goal - np.array([0, 0]))))