from __future__ import annotations

import random
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType


def sample_dark_key_to_door(config, shuffle=True):
    keys_goals_all = [np.array([i, j, k, l])
             for i in range(config['grid_size']) for j in range(config['grid_size'])
             for k in range(config['grid_size']) for l in range(config['grid_size'])]
    
    if shuffle:
        random.seed(config['env_split_seed'])
        random.shuffle(keys_goals_all)

    total_tasks = len(keys_goals_all)  # Use all possible tasks (grid_size^4)

    n_train_envs = round(total_tasks * config['train_env_ratio'])

    train_keys_goals = keys_goals_all[:n_train_envs]
    test_keys_goals = keys_goals_all[n_train_envs:]

    return train_keys_goals, test_keys_goals


class DarkKeyToDoor(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, **kwargs):
        super(DarkKeyToDoor, self).__init__()
        self.grid_size = config['grid_size']
        if 'key' in kwargs:
            self.key = kwargs['key']
        else:
            self.key = np.random.randint(0, self.grid_size, 2)
        if 'goal' in kwargs:
            self.goal = kwargs['goal']
        else:
            self.goal = np.random.randint(0, self.grid_size, 2)
        self.horizon = config['horizon']
        self.dim_obs = 2
        self.dim_action = 5
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(self.dim_obs,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.dim_action)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        self.current_step = 0

        center = self.grid_size // 2
        self.state = np.array([center, center])
        self.have_key = False
        self.reach_goal = False

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

        info = {}
        info['already_success'] = self.reach_goal

        if not self.have_key and np.array_equal(s, self.key):
            self.have_key = True
            reward = 1
        elif self.have_key and not self.reach_goal and np.array_equal(s, self.goal):
            self.reach_goal = True
            reward = 1
        else:
            reward = 0

        self.current_step += 1

        done = self.current_step >= self.horizon
        
        info['success'] = self.reach_goal

        return s.copy(), reward, done, done, info
    
    def get_optimal_action(self, state, have_key=False):
        if have_key:
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
        else:
            if state[0] < self.key[0]:
                a = 0
            elif state[0] > self.key[0]:
                a = 1
            elif state[1] < self.key[1]:
                a = 2
            elif state[1] > self.key[1]:
                a = 3
            else:
                a = 4
            
        return a
    
    def get_max_return(self):
        return 2