"""Insert irrelevant transitions without changing the underlying task or pull clock."""
from .adversarial_bandit import AdversarialBandit

BANDIT = 0
DISTRACTOR = 1


class DelayedBandit:
    def __init__(self, task, reward_seed=0, pre_steps=50, post_steps=50, delay=100):
        self.pre_steps, self.post_steps, self.delay = map(int, (pre_steps, post_steps, delay))
        if min(self.pre_steps, self.post_steps) < 1 or self.delay < 0:
            raise ValueError("Both bandit phases must be positive and delay nonnegative")
        self.bandit = AdversarialBandit(task, reward_seed, self.pre_steps + self.post_steps)
        self.num_arms = self.bandit.num_arms
        self.sequence_length = self.pre_steps + self.delay + self.post_steps
        self.reset()

    @property
    def observation(self):
        return DISTRACTOR if self.pre_steps <= self.sequence_step < self.pre_steps + self.delay else BANDIT

    def reset(self, reward_seed=None):
        self.bandit.reset(reward_seed)
        self.sequence_step = 0
        return self.observation, {}

    def step(self, action):
        if self.sequence_step >= self.sequence_length:
            raise RuntimeError("History finished; explicitly reset before the next task")
        if not 0 <= int(action) < self.num_arms:
            raise ValueError("Action outside arm range")
        observation = self.observation
        pull_index = self.bandit.pull_count if observation == BANDIT else -1
        reward = self.bandit.pull(action) if observation == BANDIT else 0.0
        info = {"transition_type": observation, "bandit_step": pull_index,
                "loss_mask": observation == BANDIT, "sequence_step": self.sequence_step}
        self.sequence_step += 1
        done = self.sequence_step == self.sequence_length
        return self.observation, reward, done, False, info
