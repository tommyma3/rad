"""DPT-style UCB: ordered warmup, then empirical mean + c / sqrt(arm pulls)."""
import numpy as np


class UCB:
    def __init__(self, num_arms=10, exploration_coefficient=1.0, seed=0):
        if num_arms < 2 or not np.isfinite(exploration_coefficient) or exploration_coefficient < 0:
            raise ValueError("Invalid UCB settings")
        self.counts = np.zeros(num_arms, dtype=np.int64)
        self.reward_sums = np.zeros(num_arms, dtype=np.float64)
        self.exploration_coefficient = float(exploration_coefficient)
        # Retain the seed/state interface for existing rollout audits. DPT's
        # online UCB rule chooses actions deterministically given its history.
        self.rng = np.random.default_rng(seed)

    @property
    def total_pulls(self):
        return int(self.counts.sum())

    def select_action(self):
        unvisited = np.flatnonzero(self.counts == 0)
        if len(unvisited):
            return int(unvisited[0])
        scores = self.reward_sums / self.counts + self.exploration_coefficient / np.sqrt(self.counts)
        # Match DPT's vectorized online controller: first index among ties.
        return int(np.argmax(scores))

    def update(self, action, reward):
        if not 0 <= int(action) < len(self.counts) or not np.isfinite(reward):
            raise ValueError("Invalid UCB observation")
        self.counts[int(action)] += 1
        self.reward_sums[int(action)] += float(reward)

    def state_dict(self):
        import copy
        return {"counts": self.counts.copy(), "reward_sums": self.reward_sums.copy(),
                "total_pulls": self.total_pulls, "rng_state": copy.deepcopy(self.rng.bit_generator.state)}
