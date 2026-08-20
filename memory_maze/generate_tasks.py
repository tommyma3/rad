"""Generate and capture fixed Memory Maze task manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from env import FixedMemoryMazeEnv, generate_task_specs, save_task_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-size", type=int, default=9, choices=(9, 11, 13, 15))
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("tasks"))
    args = parser.parse_args()
    train, test = generate_task_specs(
        args.maze_size, args.n_train, args.n_test, args.split_seed
    )
    for split, specs in (("train", train), ("test", test)):
        for index, spec in enumerate(specs):
            env = FixedMemoryMazeEnv(spec)
            env.reset()
            captured = env.resolved_task_spec
            env.close()
            save_task_spec(
                captured,
                args.output / f"{args.maze_size}x{args.maze_size}" / split / f"task-{index:06d}.json",
            )


if __name__ == "__main__":
    main()
