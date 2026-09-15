"""Stationary Gaussian bandit with rewards indexed by genuine pull, not gap time."""
import numpy as np

from .task_sampler import BanditTask


class AdversarialBandit:
    def __init__(self, task: BanditTask, reward_seed=0, horizon=100):
        self.task = task
        self.means = np.asarray(task.means, dtype=np.float64)
        if (self.means.ndim != 1 or len(self.means) < 2 or
                not np.all(np.isfinite(self.means)) or
                np.any((self.means < 0) | (self.means > 1))):
            raise ValueError("Task must contain finite reward means in [0, 1]")
        self.num_arms = len(self.means)
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        self.reset(reward_seed)

    def reset(self, reward_seed=None):
        if reward_seed is not None:
            self.reward_seed = int(reward_seed)
        self.pull_count = 0
        # Common potential outcomes let policies and delays share reward randomness.
        self.reward_noise = np.random.default_rng(self.reward_seed).normal(
            0.0, self.task.reward_std, size=(self.horizon, self.num_arms))

    def pull(self, action):
        if not 0 <= int(action) < self.num_arms:
            raise ValueError("Action outside arm range")
        if self.pull_count >= self.horizon:
            raise RuntimeError("Bandit history is finished")
        reward = float(self.means[int(action)] + self.reward_noise[self.pull_count, int(action)])
        self.pull_count += 1
        return reward
