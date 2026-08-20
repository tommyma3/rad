"""Train one fresh source learner per fixed task and record learning histories."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from algorithm import CNNPPO, DreamerTBTT, HistoryRecorderCallback
from algorithm.dreamer_tbtt import ReplayEpisode
from artifacts import TaskHistoryWriter
from env import FixedMemoryMazeEnv, load_task_spec
from utils import get_config, set_all_seeds


class ConvergenceTracker:
    """Stop after the moving return has remained on a plateau."""

    def __init__(self, config: dict) -> None:
        self.window = int(config.get("convergence_window", 20))
        self.patience = int(config.get("convergence_patience", 5))
        self.min_steps = int(config.get("convergence_min_steps", 100_000))
        self.tolerance = float(config.get("convergence_tolerance", 0.01))
        self.target_return = config.get("convergence_target_return")
        self.returns: deque[float] = deque(maxlen=2 * self.window)
        self.stable_windows = 0

    def update(self, episode_return: float, learner_step: int) -> bool:
        self.returns.append(float(episode_return))
        if learner_step < self.min_steps or len(self.returns) < 2 * self.window:
            return False
        recent = np.mean(list(self.returns)[-self.window :])
        previous = np.mean(list(self.returns)[: self.window])
        scale = max(1.0, abs(previous))
        if self.target_return is not None and recent >= float(self.target_return):
            return True
        if abs(recent - previous) <= self.tolerance * scale:
            self.stable_windows += 1
        else:
            self.stable_windows = 0
        return self.stable_windows >= self.patience


class PPOConvergenceCallback(BaseCallback):
    def __init__(self, tracker: ConvergenceTracker) -> None:
        super().__init__()
        self.tracker = tracker
        self.current_return = 0.0

    def _on_step(self) -> bool:
        self.current_return += float(np.asarray(self.locals["rewards"])[0])
        if bool(np.asarray(self.locals["dones"])[0]):
            converged = self.tracker.update(self.current_return, self.num_timesteps)
            self.current_return = 0.0
            return not converged
        return True


def _artifact_path(root: Path, spec, source_algorithm: str) -> Path:
    return root / spec.split / source_algorithm / f"{spec.task_id}.hdf5"


def collect_ppo(config: dict, spec, output_root: Path) -> None:
    raw_env = FixedMemoryMazeEnv(spec)
    raw_env.reset()
    resolved = raw_env.resolved_task_spec
    raw_env.close()
    env = VecTransposeImage(DummyVecEnv([lambda: FixedMemoryMazeEnv(resolved)]))
    path = _artifact_path(output_root, resolved, "ppo")
    with TaskHistoryWriter(path, resolved, "ppo", config) as writer:
        learner = CNNPPO(config, env, int(config.get("source_seed", 0)))
        callbacks = CallbackList(
            [
                HistoryRecorderCallback(writer),
                PPOConvergenceCallback(ConvergenceTracker(config)),
            ]
        )
        learner.learn(
            total_timesteps=int(config["max_source_timesteps"]),
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=bool(config.get("progress_bar", True)),
        )
        checkpoint = path.with_suffix(".ppo.zip")
        learner.save(checkpoint)
    env.close()


def collect_dreamer(config: dict, spec, output_root: Path) -> None:
    n_envs = int(config.get("parallel_envs", 8))
    environments = [
        FixedMemoryMazeEnv(spec, target_sequence_stream=index)
        for index in range(n_envs)
    ]
    observations = []
    resolved = None
    for environment in environments:
        observation, _ = environment.reset()
        observations.append(observation)
        resolved = environment.resolved_task_spec
    if resolved is None:
        raise RuntimeError("No Dreamer environments were constructed")
    learner = DreamerTBTT(config, config.get("device", "cuda"))
    tracker = ConvergenceTracker(config)
    path = _artifact_path(output_root, resolved, "dreamer_tbtt")
    states = [None] * n_envs
    previous_actions = [0] * n_envs
    episode_images = [[] for _ in range(n_envs)]
    episode_actions = [[] for _ in range(n_envs)]
    episode_rewards = [[] for _ in range(n_envs)]
    episode_dones = [[] for _ in range(n_envs)]
    episode_returns = np.zeros(n_envs, dtype=np.float64)
    global_step = 0
    converged = False
    prefill = int(config.get("prefill_steps", 5_000))
    update_every = int(config.get("environment_steps_per_update", 25))
    with TaskHistoryWriter(path, resolved, "dreamer_tbtt", config) as writer:
        while global_step < int(config["max_source_timesteps"]) and not converged:
            for stream_id, environment in enumerate(environments):
                observation = observations[stream_id]
                if global_step < prefill:
                    action = environment.action_space.sample()
                else:
                    action, states[stream_id] = learner.policy(
                        observation,
                        previous_actions[stream_id],
                        states[stream_id],
                    )
                next_observation, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                writer.append(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                    truncated,
                    learner.updates,
                    stream_id=stream_id,
                )
                # Dreamer replay pairs action_t with the resulting image_{t+1}.
                episode_images[stream_id].append(next_observation)
                episode_actions[stream_id].append(action)
                episode_rewards[stream_id].append(reward)
                episode_dones[stream_id].append(done)
                episode_returns[stream_id] += reward
                previous_actions[stream_id] = action
                global_step += 1
                if done:
                    learner.replay.add(
                        ReplayEpisode(
                            images=np.asarray(episode_images[stream_id], dtype=np.uint8),
                            actions=np.asarray(episode_actions[stream_id], dtype=np.int64),
                            rewards=np.asarray(episode_rewards[stream_id], dtype=np.float32),
                            dones=np.asarray(episode_dones[stream_id], dtype=np.float32),
                        )
                    )
                    converged |= tracker.update(episode_returns[stream_id], global_step)
                    episode_images[stream_id].clear()
                    episode_actions[stream_id].clear()
                    episode_rewards[stream_id].clear()
                    episode_dones[stream_id].clear()
                    episode_returns[stream_id] = 0.0
                    states[stream_id] = None
                    previous_actions[stream_id] = 0
                    next_observation, _ = environment.reset()
                observations[stream_id] = next_observation
                if learner.replay.ready() and global_step >= prefill and global_step % update_every == 0:
                    learner.train_step()
                if global_step >= int(config["max_source_timesteps"]):
                    break
        torch.save(learner.state_dict(), path.with_suffix(".dreamer.pt"))
    for environment in environments:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    config = get_config(args.config)
    set_all_seeds(int(config.get("source_seed", 0)))
    spec = load_task_spec(args.task)
    source_algorithm = config["source_algorithm"]
    if source_algorithm == "ppo":
        collect_ppo(config, spec, args.output)
    elif source_algorithm == "dreamer_tbtt":
        collect_dreamer(config, spec, args.output)
    else:
        raise ValueError(f"Unknown source_algorithm: {source_algorithm}")


if __name__ == "__main__":
    main()
