"""Sample one fixed Memory Maze task and probe a source learner on it."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from algorithm import CNNPPO, DreamerTBTT
from algorithm.dreamer_tbtt import ReplayEpisode
from env import FixedMemoryMazeEnv, MemoryMazeTaskSpec, save_task_spec
from utils import get_config, set_all_seeds


class EpisodeRewardPrinter(BaseCallback):
    """Print mean episode returns every fixed number of completed episodes."""

    def __init__(self, print_every_episodes: int, window: int = 10) -> None:
        super().__init__()
        if print_every_episodes <= 0:
            raise ValueError("print_every_episodes must be positive")
        self.print_every_episodes = int(print_every_episodes)
        self.window = int(window)
        self.completed_episodes = 0
        self.current_return = 0.0
        self.recent_returns: deque[float] = deque(maxlen=max(1, self.window))

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=np.bool_)
        if len(rewards) != 1 or len(dones) != 1:
            raise ValueError("sample_and_train_ppo.py expects exactly one environment")

        self.current_return += float(rewards[0])
        if dones[0]:
            self.completed_episodes += 1
            episode_return = self.current_return
            self.recent_returns.append(episode_return)
            if self.completed_episodes % self.print_every_episodes == 0:
                mean_return = float(np.mean(self.recent_returns))
                print(
                    "episode "
                    f"{self.completed_episodes}: "
                    f"reward={episode_return:.3f}, "
                    f"mean_last_{len(self.recent_returns)}={mean_return:.3f}, "
                    f"timesteps={self.num_timesteps}",
                    flush=True,
                )
            self.current_return = 0.0
        return True


class EpisodeRewardReporter:
    """Track and print episode returns from one or more environment streams."""

    def __init__(self, print_every_episodes: int, window: int = 10) -> None:
        if print_every_episodes <= 0:
            raise ValueError("print_every_episodes must be positive")
        self.print_every_episodes = int(print_every_episodes)
        self.recent_returns: deque[float] = deque(maxlen=max(1, int(window)))
        self.completed_episodes = 0

    def update(self, episode_return: float, global_step: int, updates: int) -> None:
        self.completed_episodes += 1
        self.recent_returns.append(float(episode_return))
        if self.completed_episodes % self.print_every_episodes != 0:
            return
        mean_return = float(np.mean(self.recent_returns))
        print(
            "episode "
            f"{self.completed_episodes}: "
            f"reward={episode_return:.3f}, "
            f"mean_last_{len(self.recent_returns)}={mean_return:.3f}, "
            f"timesteps={global_step}, "
            f"updates={updates}",
            flush=True,
        )


def sample_task(maze_size: int, seed: int) -> MemoryMazeTaskSpec:
    rng = np.random.default_rng(seed)
    generation_seed = int(rng.integers(0, np.iinfo(np.int32).max, dtype=np.int32))
    return MemoryMazeTaskSpec(
        maze_size=int(maze_size),
        generation_seed=generation_seed,
        split="train",
    )


def resolve_task(spec: MemoryMazeTaskSpec) -> MemoryMazeTaskSpec:
    env = FixedMemoryMazeEnv(spec)
    try:
        env.reset()
        return env.resolved_task_spec
    finally:
        env.close()


def run_ppo(config: dict, resolved: MemoryMazeTaskSpec, args: argparse.Namespace) -> None:
    seed = int(config.get("source_seed", 0))
    env = VecTransposeImage(DummyVecEnv([lambda: FixedMemoryMazeEnv(resolved)]))
    try:
        learner = CNNPPO(config, env, seed)
        learner.learn(
            total_timesteps=int(config["max_source_timesteps"]),
            callback=EpisodeRewardPrinter(
                print_every_episodes=args.print_every_episodes,
                window=args.reward_window,
            ),
            reset_num_timesteps=True,
            progress_bar=bool(config.get("progress_bar", False)),
        )
        if args.save_model is not None:
            learner.save(args.save_model)
            print(f"saved PPO checkpoint: {args.save_model}", flush=True)
    finally:
        env.close()


def run_dreamer_tbtt(
    config: dict,
    resolved: MemoryMazeTaskSpec,
    args: argparse.Namespace,
) -> None:
    n_envs = int(config.get("parallel_envs", 8))
    environments = [
        FixedMemoryMazeEnv(resolved, target_sequence_stream=index)
        for index in range(n_envs)
    ]
    learner = DreamerTBTT(config, config.get("device", "cuda"))
    reporter = EpisodeRewardReporter(
        print_every_episodes=args.print_every_episodes,
        window=args.reward_window,
    )
    observations = []
    for environment in environments:
        observation, _ = environment.reset()
        observations.append(observation)

    states = [None] * n_envs
    previous_actions = [0] * n_envs
    episode_images = [[] for _ in range(n_envs)]
    episode_actions = [[] for _ in range(n_envs)]
    episode_rewards = [[] for _ in range(n_envs)]
    episode_dones = [[] for _ in range(n_envs)]
    episode_returns = np.zeros(n_envs, dtype=np.float64)
    global_step = 0
    prefill = int(config.get("prefill_steps", 5_000))
    update_every = int(config.get("environment_steps_per_update", 25))

    try:
        while global_step < int(config["max_source_timesteps"]):
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
                    reporter.update(
                        episode_returns[stream_id],
                        global_step,
                        learner.updates,
                    )
                    episode_images[stream_id].clear()
                    episode_actions[stream_id].clear()
                    episode_rewards[stream_id].clear()
                    episode_dones[stream_id].clear()
                    episode_returns[stream_id] = 0.0
                    states[stream_id] = None
                    previous_actions[stream_id] = 0
                    next_observation, _ = environment.reset()

                observations[stream_id] = next_observation
                if (
                    learner.replay.ready()
                    and global_step >= prefill
                    and global_step % update_every == 0
                ):
                    metrics = learner.train_step()
                    if (
                        int(args.print_every_updates) > 0
                        and learner.updates % int(args.print_every_updates) == 0
                    ):
                        loss_model = metrics.get("loss_model", float("nan"))
                        imagined_return = metrics.get("imagined_return", float("nan"))
                        print(
                            "update "
                            f"{learner.updates}: "
                            f"loss_model={loss_model:.4f}, "
                            f"imagined_return={imagined_return:.4f}, "
                            f"timesteps={global_step}",
                            flush=True,
                        )
                if global_step >= int(config["max_source_timesteps"]):
                    break

        if args.save_model is not None:
            torch.save(learner.state_dict(), args.save_model)
            print(f"saved Dreamer TBTT checkpoint: {args.save_model}", flush=True)
    finally:
        for environment in environments:
            environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-algorithm",
        choices=("ppo", "dreamer_tbtt"),
        default=None,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--maze-size", type=int, default=9, choices=(9, 11, 13, 15))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--print-every-episodes", type=int, default=5)
    parser.add_argument("--print-every-updates", type=int, default=10)
    parser.add_argument("--reward-window", type=int, default=10)
    parser.add_argument("--save-task", type=Path)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    default_algorithm = args.source_algorithm or "ppo"
    default_config = Path(f"config/algorithm/{default_algorithm}.yaml")
    config_path = args.config or default_config
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT / config_path
    config = get_config(config_path)
    source_algorithm = str(config.get("source_algorithm", default_algorithm))
    if args.source_algorithm is not None and source_algorithm != args.source_algorithm:
        raise ValueError(
            f"--source-algorithm={args.source_algorithm} does not match "
            f"{config_path} source_algorithm={source_algorithm}"
        )
    config["max_source_timesteps"] = int(args.total_timesteps)
    config["progress_bar"] = bool(args.progress_bar)
    if args.device is not None:
        config["device"] = args.device

    seed = int(config.get("source_seed", 0))
    set_all_seeds(seed)

    sampled = sample_task(args.maze_size, args.sample_seed)
    resolved = resolve_task(sampled)
    print(
        "sampled task: "
        f"maze_size={resolved.maze_size}, "
        f"generation_seed={resolved.generation_seed}, "
        f"task_id={resolved.task_id}",
        flush=True,
    )

    if args.save_task is not None:
        save_task_spec(resolved, args.save_task)
        print(f"saved task manifest: {args.save_task}", flush=True)

    if source_algorithm == "ppo":
        run_ppo(config, resolved, args)
    elif source_algorithm == "dreamer_tbtt":
        run_dreamer_tbtt(config, resolved, args)
    else:
        raise ValueError(f"Unknown source_algorithm: {source_algorithm}")


if __name__ == "__main__":
    main()
