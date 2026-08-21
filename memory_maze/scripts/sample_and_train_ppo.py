"""Sample one fixed Memory Maze task and train the PPO source learner on it."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from algorithm import CNNPPO
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/algorithm/ppo.yaml"))
    parser.add_argument("--maze-size", type=int, default=9, choices=(9, 11, 13, 15))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--print-every-episodes", type=int, default=5)
    parser.add_argument("--reward-window", type=int, default=10)
    parser.add_argument("--save-task", type=Path)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT / config_path
    config = get_config(config_path)
    config["source_algorithm"] = "ppo"
    config["max_source_timesteps"] = int(args.total_timesteps)
    config["progress_bar"] = bool(args.progress_bar)

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


if __name__ == "__main__":
    main()
