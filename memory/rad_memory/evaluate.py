"""Episode-reset evaluation with cue-gap and compression diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from .envs import MemoryTaskSpec, make_memory_env
from .model import MODEL


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    spec: MemoryTaskSpec,
    episodes: int,
    *,
    sample: bool = False,
) -> tuple[list[dict], dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MODEL[config["model"]](config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    env = make_memory_env(spec)
    records = []
    for episode_index in range(episodes):
        observation, info = env.reset(seed=spec.seed + episode_index)
        context = model.start_context(observation)
        last_cue_step = 0 if info["memory_cue_visible"] else None
        total_reward = 0.0
        while True:
            action = model.act(context, sample=sample)
            next_observation, reward, terminated, truncated, info = env.step(action)
            model.observe(
                context,
                action,
                reward,
                terminated,
                truncated,
                next_observation,
            )
            total_reward += float(reward)
            if info["memory_cue_visible"]:
                last_cue_step = int(info["memory_step"])
            if terminated or truncated:
                break
            observation = next_observation
        length = int(info["memory_step"])
        cue_gap = None if last_cue_step is None else length - last_cue_step
        active_context = int(config["n_transit"])
        records.append(
            {
                "episode": episode_index,
                "seed": spec.seed + episode_index,
                "return": total_reward,
                "success": bool(info["memory_success"] and total_reward > 0),
                "failure": bool(info["memory_failure"]),
                "truncated": bool(truncated),
                "length": length,
                "last_cue_step": last_cue_step,
                "cue_gap": cue_gap,
                "cue_outside_active_context": cue_gap is None or cue_gap >= active_context,
                "compressions": int(context.get("num_compressions", 0)),
            }
        )
    env.close()
    outside = [item for item in records if item["cue_outside_active_context"]]
    summary = {
        "checkpoint": str(checkpoint_path),
        "model": config["model"],
        "n_transit": int(config["n_transit"]),
        "episodes": episodes,
        "success_rate": mean(float(item["success"]) for item in records),
        "mean_return": mean(item["return"] for item in records),
        "mean_length": mean(item["length"] for item in records),
        "out_of_window_episodes": len(outside),
        "out_of_window_success_rate": (
            mean(float(item["success"]) for item in outside) if outside else None
        ),
        "mean_compressions": mean(item["compressions"] for item in records),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AD/RAD on MiniGrid Memory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--size", type=int)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--random-length", action="store_true")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = MemoryTaskSpec(
        args.env_id,
        args.seed,
        "test",
        horizon=args.horizon,
        controlled=args.controlled,
        size=args.size,
        random_length=args.random_length,
    )
    records, summary = evaluate_checkpoint(
        args.checkpoint,
        spec,
        args.episodes,
        sample=args.sample,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
