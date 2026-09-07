"""Fixed-task adaptation or legacy episode-reset evaluation."""

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
    reset_context_each_episode: bool = False,
    trial_seed: int = 0,
) -> tuple[list[dict], dict]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    torch.manual_seed(trial_seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    fixed = spec.configuration is not None
    if fixed != (config.get("history_scope", "episode") == "task"):
        raise ValueError("Checkpoint history scope does not match evaluation task mode")
    model = MODEL[config["model"]](config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    env = make_memory_env(spec)
    records = []
    observation, info = env.reset(seed=spec.seed)
    context = model.start_context(observation)
    for episode_index in range(episodes):
        if episode_index and (not fixed or reset_context_each_episode):
            context = model.start_context(observation)
        compressions_before = int(context.get("num_compressions", 0))
        last_cue_step = 0 if info["memory_cue_visible"] else None
        total_reward = 0.0
        while True:
            action = model.act(context, sample=sample)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # The next policy state after a boundary is the reset state. Keep the
            # terminal reward/done tokens, without inserting an unacted terminal state.
            if done:
                reset_observation, reset_info = env.reset(seed=spec.seed + episode_index + 1)
            model.observe(
                context,
                action,
                reward,
                terminated,
                truncated,
                reset_observation if done else next_observation,
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
                "task_id": spec.task_id,
                "seed": spec.seed if fixed else spec.seed + episode_index,
                "trial_seed": trial_seed,
                "return": total_reward,
                "success": bool(info["memory_success"] and total_reward > 0),
                "failure": bool(info["memory_failure"]),
                "truncated": bool(truncated),
                "length": length,
                "last_cue_step": last_cue_step,
                "cue_gap": cue_gap,
                "cue_outside_active_context": cue_gap is None or cue_gap >= active_context,
                "compressions": int(context.get("num_compressions", 0)) - compressions_before,
            }
        )
        observation, info = reset_observation, reset_info
    env.close()
    outside = [item for item in records if item["cue_outside_active_context"]]
    summary = {
        "checkpoint": str(checkpoint_path),
        "model": config["model"],
        "n_transit": int(config["n_transit"]),
        "episodes": episodes,
        "history_scope": "task" if fixed else "episode",
        "reset_context_each_episode": reset_context_each_episode,
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


def evaluate_pool(checkpoint, manifest, episodes_per_task, *, trials=3, seed=0,
                  sample=False, reset_context_each_episode=False):
    from .task_pool import load_pool
    if trials <= 0:
        raise ValueError("trials must be positive")
    pool = load_pool(manifest)
    checkpoint_config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    if checkpoint_config.get("manifest_fingerprint") != pool["fingerprint"]:
        raise ValueError("Checkpoint and evaluation manifest do not match")
    records = []
    for value in pool["tasks"]:
        spec = MemoryTaskSpec.from_dict(value)
        if spec.split != "test":
            continue
        for trial in range(trials):
            trial_records, _ = evaluate_checkpoint(
                checkpoint, spec, episodes_per_task, sample=sample, trial_seed=seed + trial,
                reset_context_each_episode=reset_context_each_episode)
            records.extend(dict(record, trial=trial) for record in trial_records)
    curves = []
    for index in range(episodes_per_task):
        selected = [r for r in records if r["episode"] == index]
        curves.append({"episode": index, "success_rate": mean(float(r["success"]) for r in selected),
                       "mean_return": mean(r["return"] for r in selected)})
    return records, {
        "model": checkpoint_config["model"],
        "n_transit": int(checkpoint_config["n_transit"]),
        "history_scope": "task",
        "out_of_window_success_rate": (
            mean(float(r["success"]) for r in records if r["cue_outside_active_context"])
            if any(r["cue_outside_active_context"] for r in records) else None),
        "mean_compressions": mean(r["compressions"] for r in records),
        "manifest_fingerprint": pool["fingerprint"], "episodes": len(records),
        "tasks": len({r["task_id"] for r in records}), "trials": trials,
        "sample": sample, "reset_context_each_episode": reset_context_each_episode,
        "success_rate": mean(float(r["success"]) for r in records),
        "mean_return": mean(r["return"] for r in records), "adaptation_curve": curves,
        "first_episode_success_rate": curves[0]["success_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AD/RAD on MiniGrid Memory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--reset-context-each-episode", action="store_true")
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
    if args.manifest:
        records, summary = evaluate_pool(
            args.checkpoint, args.manifest, args.episodes, trials=args.trials, seed=args.seed,
            sample=args.sample, reset_context_each_episode=args.reset_context_each_episode)
    else:
        records, summary = evaluate_checkpoint(args.checkpoint, spec, args.episodes, sample=args.sample,
                                               reset_context_each_episode=args.reset_context_each_episode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
