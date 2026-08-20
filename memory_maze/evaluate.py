"""Evaluate AD or RAD through repeated episodes on unseen fixed tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env import FixedMemoryMazeEnv, load_task_spec
from model import MODEL
from utils import CHECKPOINT_FORMAT, normalize_state_dict


def evaluate_task(model, task_spec, episodes: int, sample: bool) -> dict:
    env = FixedMemoryMazeEnv(task_spec)
    observation, _ = env.reset()
    context = model.start_context(observation)
    returns = []
    targets_by_bin = []
    for _ in range(episodes):
        episode_return = 0.0
        bins = np.zeros(10, dtype=np.float32)
        for step in range(env.episode_steps):
            action = model.act(context, sample=sample)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            bins[min(9, 10 * step // env.episode_steps)] += reward
            if done:
                reset_observation, _ = env.reset()
                model.observe(context, action, reward, True, reset_observation)
                observation = reset_observation
                break
            model.observe(context, action, reward, False, next_observation)
            observation = next_observation
        returns.append(episode_return)
        targets_by_bin.append(bins)
    env.close()
    result = {
        "task_id": task_spec.task_id,
        "episode_returns": returns,
        "targets_by_time_bin": np.stack(targets_by_bin).tolist(),
    }
    if "num_compressions" in context:
        result["num_compressions"] = int(context["num_compressions"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not memory-maze-sar-v1")
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL[config["model"]](config).to(device)
    model.load_state_dict(normalize_state_dict(checkpoint["model"]))
    model.eval()
    results = [
        evaluate_task(model, load_task_spec(path), args.episodes, not args.greedy)
        for path in args.task
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
