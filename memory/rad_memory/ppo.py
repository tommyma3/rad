"""Feed-forward PPO source learner for fixed MiniGrid Memory tasks."""
from dataclasses import dataclass
from pathlib import Path

from .recurrent_ppo import RecurrentPPOConfig, evaluate_recurrent_ppo


@dataclass(frozen=True)
class PPOConfig(RecurrentPPOConfig):
    """Standard PPO with an MLP, without recurrence or observation stacking."""
    policy: str = "MlpPolicy"


def build_ppo(env, *, seed: int, config: PPOConfig,
              tensorboard_log: str | Path | None, verbose: int = 1, device: str = "cpu"):
    from stable_baselines3 import PPO

    if config.policy != "MlpPolicy":
        raise ValueError("Feed-forward Memory PPO requires MlpPolicy")
    kwargs = config.to_dict()
    policy = kwargs.pop("policy")
    return PPO(policy, env, seed=seed, **kwargs, verbose=verbose, device=device,
               tensorboard_log=None if tensorboard_log is None else str(tensorboard_log))


def evaluate_ppo(model, spec, *, episodes: int, deterministic: bool = True):
    # SB3's feed-forward predict accepts the same state/episode_start arguments
    # and returns no recurrent state. Keep reward/success accounting identical.
    return evaluate_recurrent_ppo(model, spec, episodes=episodes, deterministic=deterministic)
