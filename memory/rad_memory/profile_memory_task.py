"""Measure actual episode and cue-to-decision lengths with a privileged planner."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path

from .envs import MemoryTaskSpec, make_memory_env


def _traversable(base, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= base.width or y >= base.height:
        return False
    cell = base.grid.get(x, y)
    return cell is None or cell.can_overlap()


def _plan(base, destination: tuple[int, int]) -> list[int]:
    from minigrid.core.actions import Actions

    start = (int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir))
    queue = deque([start])
    previous = {start: None}
    previous_action = {}
    goal = None
    vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    while queue:
        state = queue.popleft()
        x, y, direction = state
        if (x, y) == tuple(destination):
            goal = state
            break
        candidates = [
            ((x, y, (direction - 1) % 4), int(Actions.left)),
            ((x, y, (direction + 1) % 4), int(Actions.right)),
        ]
        dx, dy = vectors[direction]
        if _traversable(base, x + dx, y + dy):
            candidates.append(((x + dx, y + dy, direction), int(Actions.forward)))
        for candidate, action in candidates:
            if candidate not in previous:
                previous[candidate] = state
                previous_action[candidate] = action
                queue.append(candidate)
    if goal is None:
        raise RuntimeError(f"No route from {start} to {destination}")
    actions = []
    cursor = goal
    while previous[cursor] is not None:
        actions.append(previous_action[cursor])
        cursor = previous[cursor]
    return list(reversed(actions))


def profile_episode(spec: MemoryTaskSpec, seed: int) -> dict:
    env = make_memory_env(spec)
    _, info = env.reset(seed=seed)
    last_cue_step = 0 if info["memory_cue_visible"] else None
    actions = []
    base = env.unwrapped
    cue_waypoint = (1, int(base.height) // 2)
    if not info["memory_cue_visible"]:
        actions.extend(_plan(base, cue_waypoint))
        for action in actions:
            _, _, terminated, truncated, info = env.step(action)
            if info["memory_cue_visible"]:
                last_cue_step = info["memory_step"]
            if terminated or truncated:
                raise RuntimeError("Episode ended before the cue waypoint")
        if last_cue_step is None:
            from minigrid.core.actions import Actions

            for _ in range(4):
                _, _, terminated, truncated, info = env.step(int(Actions.left))
                if info["memory_cue_visible"]:
                    last_cue_step = info["memory_step"]
                    break
                if terminated or truncated:
                    raise RuntimeError("Episode ended while orienting toward the cue")
        if last_cue_step is None:
            raise RuntimeError("Planner reached the cue room but never observed the cue")
    route = _plan(base, tuple(base.success_pos))
    for action in route:
        _, reward, terminated, truncated, info = env.step(action)
        if info["memory_cue_visible"]:
            last_cue_step = info["memory_step"]
        if terminated or truncated:
            break
    result = {
        "seed": seed,
        "length": int(info["memory_step"]),
        "last_cue_step": last_cue_step,
        "cue_to_decision": None if last_cue_step is None else int(info["memory_step"] - last_cue_step),
        "success": bool(info["memory_success"] and reward > 0),
        "native_horizon": int(base.max_steps),
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MiniGrid Memory dependency lengths")
    parser.add_argument("--env-id", default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--size", type=int)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--random-length", action="store_true")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--slack", type=float, default=1.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = MemoryTaskSpec(
        args.env_id,
        args.seed,
        "profile",
        horizon=args.horizon,
        controlled=args.controlled,
        size=args.size,
        random_length=args.random_length,
    )
    episodes = [profile_episode(spec, args.seed + index) for index in range(args.episodes)]
    if not all(item["success"] for item in episodes):
        raise RuntimeError("Privileged profiler failed at least one task; inspect its route planner")
    lengths = sorted(item["length"] for item in episodes)
    gaps = sorted(item["cue_to_decision"] for item in episodes if item["cue_to_decision"] is not None)
    recommendation = int(math.ceil(max(lengths) * args.slack))
    median_gap = gaps[len(gaps) // 2]
    short_context = min(
        int(math.ceil(0.5 * recommendation)),
        max(1, int(math.floor(0.75 * median_gap))),
    )
    summary = {
        "task_spec": spec.to_dict(),
        "episodes": len(episodes),
        "max_episode_length": max(lengths),
        "p95_episode_length": lengths[int(0.95 * (len(lengths) - 1))],
        "max_cue_to_decision": max(gaps),
        "median_cue_to_decision": median_gap,
        "p95_cue_to_decision": gaps[int(0.95 * (len(gaps) - 1))],
        "recommended_horizon": recommendation,
        "short_context": short_context,
        "short_context_ratio": short_context / recommendation,
        "fraction_cue_gap_outside_short_context": sum(gap >= short_context for gap in gaps) / len(gaps),
        "records": episodes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
