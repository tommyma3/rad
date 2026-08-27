import numpy as np
import gymnasium as gym


class OptimisticExplorationWrapper(gym.Wrapper):
    """Reward wrapper for PPO with tabular optimistic exploration."""

    def __init__(self, env, config, visit_counts=None):
        super().__init__(env)
        self.bonus_coef = config.get('optimistic_bonus_coef', 1.0)
        self.count_key = config.get('optimistic_count_key', 'full_state')
        self.visit_counts = visit_counts if visit_counts is not None else {}

    def _state_key(self, obs):
        state = tuple(np.asarray(obs, dtype=np.int64).reshape(-1).tolist())
        if self.count_key == 'full_state' and hasattr(self.unwrapped, 'have_key'):
            state = state + (int(bool(self.unwrapped.have_key)),)
        return state

    def _increment_visit(self, obs):
        key = self._state_key(obs)
        count = self.visit_counts.get(key, 0) + 1
        self.visit_counts[key] = count
        return key, count

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._increment_visit(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        key, count = self._increment_visit(obs)
        true_reward = reward
        bonus = self.bonus_coef / count
        pseudo_reward = true_reward + bonus

        info = dict(info)
        info['true_reward'] = true_reward
        info['exploration_bonus'] = bonus
        info['pseudo_reward'] = pseudo_reward
        info['visit_count'] = count
        info['visit_state'] = key

        return obs, pseudo_reward, terminated, truncated, info
