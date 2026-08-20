"""Naive feed-forward convolutional PPO source learner."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class CNNPPO(PPO):
    def __init__(self, config: dict, env, seed: int, log_dir: str | None = None) -> None:
        policy_kwargs = dict(config.get("policy_kwargs", {}))
        super().__init__(
            policy="CnnPolicy",
            env=env,
            learning_rate=float(config.get("source_lr", 2.5e-4)),
            n_steps=int(config.get("n_steps", 1024)),
            batch_size=int(config.get("batch_size", 256)),
            n_epochs=int(config.get("n_epochs", 4)),
            gamma=float(config.get("gamma", 0.995)),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            clip_range=float(config.get("clip_range", 0.2)),
            ent_coef=float(config.get("ent_coef", 0.001)),
            vf_coef=float(config.get("vf_coef", 0.5)),
            max_grad_norm=float(config.get("max_grad_norm", 0.5)),
            target_kl=config.get("target_kl"),
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            verbose=int(config.get("verbose", 0)),
            seed=int(seed),
            device=config.get("device", "auto"),
            tensorboard_log=log_dir,
        )


class HistoryRecorderCallback(BaseCallback):
    """Stream SB3 behavior transitions into a TaskHistoryWriter."""

    def __init__(self, writer, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        observations = self.locals["obs_tensor"].detach().cpu().numpy()
        actions = np.asarray(self.locals["actions"])
        rewards = np.asarray(self.locals["rewards"])
        dones = np.asarray(self.locals["dones"], dtype=np.bool_)
        infos = self.locals["infos"]
        new_observations = np.asarray(self.locals["new_obs"])
        if len(observations) != 1:
            raise ValueError("PPO source-history recording requires exactly one environment")
        next_observation = (
            np.asarray(infos[0]["terminal_observation"])
            if dones[0] and "terminal_observation" in infos[0]
            else new_observations[0]
        )
        truncated = bool(infos[0].get("TimeLimit.truncated", dones[0]))
        terminated = bool(dones[0] and not truncated)
        self.writer.append(
            observation=observations[0],
            action=int(np.asarray(actions[0]).item()),
            reward=float(rewards[0]),
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            learner_step=int(self.num_timesteps),
        )
        return True

    def _on_training_end(self) -> None:
        self.writer.finish_episode()
