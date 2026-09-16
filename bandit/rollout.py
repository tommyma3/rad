"""One rollout implementation shared by collection and policy evaluation."""
import numpy as np

from .algorithm import UCB
from .env import BANDIT, DelayedBandit


def generate_history(task, *, delay=100, pre_steps=50, post_steps=50,
                     reward_seed=0, learner_seed=1, distractor_seed=2,
                     exploration_coefficient=1.0, policy=None, prefix=None):
    env = DelayedBandit(task, reward_seed, pre_steps, post_steps, delay)
    learner = UCB(env.num_arms, exploration_coefficient, learner_seed)
    distractor = np.random.default_rng(distractor_seed)
    history = {key: [] for key in ("states", "actions", "rewards", "next_states",
                                   "loss_mask", "bandit_steps", "compression_events",
                                   "recent_steps")}
    if policy is not None:
        policy.reset()
    before_gap = after_gap = None
    for step in range(env.sequence_length):
        observation = env.observation
        if step == pre_steps:
            before_gap = learner.state_dict()
        if step == pre_steps + delay:
            after_gap = learner.state_dict()
        if observation != BANDIT:
            action = int(distractor.integers(env.num_arms))
        elif prefix is not None and step < pre_steps:
            action = int(prefix["actions"][step])
        elif policy is None:
            action = learner.select_action()
        else:
            action = policy.action(observation)
        next_observation, reward, _, _, info = env.step(action)
        if prefix is not None and step < pre_steps:
            if reward != float(prefix["rewards"][step]):
                raise ValueError("Shared prefix requires the same task and reward stream")
        # Capture the context used for this decision, before adding its outcome.
        history["compression_events"].append(getattr(policy, "compression_count", 0))
        history["recent_steps"].append(getattr(policy, "recent_steps", 0))
        if observation == BANDIT:
            learner.update(action, reward)
        if policy is not None:
            policy.observe(observation, action, reward)
        for key, value in (("states", observation), ("actions", action),
                           ("rewards", reward), ("next_states", next_observation),
                           ("loss_mask", info["loss_mask"]),
                           ("bandit_steps", info["bandit_step"])):
            history[key].append(value)
    # Preserve exact environment draws for replay/shared-prefix equality. Models
    # cast rewards to their embedding dtype when constructing tokens.
    arrays = {key: np.asarray(values, dtype=np.float64 if key == "rewards" else
                              np.bool_ if key == "loss_mask" else np.int64)
              for key, values in history.items()}
    return arrays, {"before_gap": before_gap, "after_gap": after_gap,
                    "final_ucb_state": learner.state_dict()}
