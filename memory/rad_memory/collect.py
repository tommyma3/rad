"""Collect episode artifacts from recurrent PPO learning checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from .artifacts import TaskHistoryWriter, transition_record
from .envs import MemoryTaskSpec, flatten_numeric_observation, make_memory_env


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"(?:checkpoint|teacher)[-_](\d+)", path.stem)
    return int(match.group(1)) if match else 10**18


def collect_checkpoint(
    model,
    spec: MemoryTaskSpec,
    writer: TaskHistoryWriter,
    episodes: int,
    learner_step: int,
    deterministic: bool,
) -> dict[str, float]:
    env = make_memory_env(spec)
    successes = 0
    returns = []
    for _ in range(episodes):
        observation, observation_info = env.reset()
        recurrent_state = None
        episode_start = np.ones((1,), dtype=bool)
        steps = []
        while True:
            source_observation = flatten_numeric_observation(observation)
            action, recurrent_state = model.predict(
                source_observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            action = int(np.asarray(action).item())
            next_observation, reward, terminated, truncated, info = env.step(action)
            steps.append(
                transition_record(
                    observation,
                    action,
                    reward,
                    terminated,
                    truncated,
                    next_observation,
                    info,
                    observation_info,
                )
            )
            observation = next_observation
            observation_info = info
            episode_start[:] = terminated or truncated
            if terminated or truncated:
                break
        writer.write_episode(steps, learner_step=learner_step)
        episode_return = sum(step["reward"] for step in steps)
        returns.append(episode_return)
        successes += int(bool(steps[-1]["success"] and episode_return > 0))
    env.close()
    return {
        "episodes": episodes,
        "success_rate": successes / max(episodes, 1),
        "mean_return": sum(returns) / max(len(returns), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MiniGrid Memory teacher histories")
    parser.add_argument("--task-spec", required=True, help="JSON task-spec file")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=100)
    parser.add_argument("--output-root", default="datasets")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as error:
        raise RuntimeError(
            "RecurrentPPO requires sb3-contrib; finish the environment setup first"
        ) from error

    with Path(args.task_spec).open("r", encoding="utf-8") as handle:
        spec = MemoryTaskSpec.from_dict(json.load(handle))
    if spec.configuration is not None:
        raise ValueError("Fixed tasks require online histories: use rad_memory.train_task_pool")
    if args.split is not None or args.seed is not None:
        spec = MemoryTaskSpec(
            env_id=spec.env_id,
            seed=spec.seed if args.seed is None else args.seed,
            split=spec.split if args.split is None else args.split,
            horizon=spec.horizon,
            controlled=spec.controlled,
            size=spec.size,
            random_length=spec.random_length,
        )
    output = (
        Path(args.output_root)
        / spec.split
        / "recurrent_ppo"
        / f"{spec.task_id}.hdf5"
    )
    checkpoints = sorted((Path(value) for value in args.checkpoint), key=_checkpoint_step)
    with TaskHistoryWriter(output, spec, "recurrent_ppo") as writer:
        for checkpoint in checkpoints:
            model = RecurrentPPO.load(checkpoint)
            parsed_step = _checkpoint_step(checkpoint)
            learner_step = int(model.num_timesteps) if parsed_step == 10**18 else parsed_step
            already_collected = writer.count_episodes_at_learner_step(learner_step)
            remaining = max(0, args.episodes_per_checkpoint - already_collected)
            if not remaining:
                print(f"Skipping {checkpoint}: {already_collected} episodes already collected")
                continue
            metrics = collect_checkpoint(
                model,
                spec,
                writer,
                remaining,
                learner_step,
                deterministic=not args.stochastic,
            )
            print(json.dumps({"checkpoint": str(checkpoint), "learner_step": learner_step, **metrics}))


if __name__ == "__main__":
    main()
