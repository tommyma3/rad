"""Generate and validate a pool of distinct, reset-invariant Memory tasks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

from .envs import MemoryTaskSpec, make_memory_env

POOL_FORMAT = "rad-memory-task-pool-v1"


def freeze_task(spec: MemoryTaskSpec) -> MemoryTaskSpec:
    env = make_memory_env(spec)
    try:
        env.reset(seed=spec.seed)
        base = env.unwrapped
        configuration = {
            "grid": base.grid.encode().tolist(),
            "size": int(base.width),
            "max_steps": int(base.max_steps),
            "agent_view_size": int(base.agent_view_size),
            "agent_pos": [int(x) for x in base.agent_pos],
            "agent_dir": int(base.agent_dir),
            "success_pos": [int(x) for x in base.success_pos],
            "failure_pos": [int(x) for x in base.failure_pos],
            "mission": base.mission,
        }
        return replace(spec, configuration=configuration)
    finally:
        env.close()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_pool(template: MemoryTaskSpec, count: int, train_ratio: float,
                  split_seed: int, max_candidates: int = 10000) -> dict:
    if count < 2 or not 0 < train_ratio < 1 or max_candidates < count:
        raise ValueError("Require count >= 2, 0 < train_ratio < 1 and max_candidates >= count")
    unique = {}
    for offset in range(max_candidates):
        task = freeze_task(replace(template, seed=template.seed + offset, configuration=None))
        unique.setdefault(task.task_id, task)
        if len(unique) == count:
            break
    if len(unique) < count:
        raise ValueError(f"Found only {len(unique)} unique tasks in {max_candidates} candidates; "
                         "reduce pool size or increase the candidate budget")
    tasks = list(unique.values())
    random.Random(split_seed).shuffle(tasks)
    n_train = min(count - 1, max(1, round(count * train_ratio)))
    payload = {
        "format": POOL_FORMAT,
        "pool_seed": template.seed,
        "env_split_seed": split_seed,
        "train_env_ratio": train_ratio,
        "tasks": [replace(task, split="train" if i < n_train else "test").to_dict()
                  for i, task in enumerate(tasks)],
    }
    return payload | {"fingerprint": fingerprint(payload)}


def load_pool(path: str | Path) -> dict:
    pool = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {k: v for k, v in pool.items() if k != "fingerprint"}
    if pool.get("format") != POOL_FORMAT or pool.get("fingerprint") != fingerprint(payload):
        raise ValueError("Invalid task manifest format or fingerprint")
    tasks = [MemoryTaskSpec.from_dict(value) for value in pool["tasks"]]
    if any(t.configuration is None or t.split not in {"train", "test"} for t in tasks):
        raise ValueError("Pool must contain fixed train/test tasks")
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task configurations in pool")
    if {t.split for t in tasks} != {"train", "test"}:
        raise ValueError("Pool must have nonempty train and test splits")
    if any(t.task_id != v["task_id"] for t, v in zip(tasks, pool["tasks"])):
        raise ValueError("Task fingerprint mismatch")
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--size", type=int)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--random-length", action="store_true")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--env-split-seed", type=int, default=0)
    parser.add_argument("--train-env-ratio", type=float, default=0.8)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = MemoryTaskSpec(args.env_id, args.pool_seed, "train", horizon=args.horizon,
                          controlled=args.controlled, size=args.size, random_length=args.random_length)
    pool = generate_pool(spec, args.num_tasks, args.train_env_ratio,
                         args.env_split_seed, args.max_candidates)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(pool, handle, indent=2, sort_keys=True)
    print(json.dumps({"fingerprint": pool["fingerprint"], "tasks": len(pool["tasks"]),
                      "train": sum(t["split"] == "train" for t in pool["tasks"])}))


if __name__ == "__main__":
    main()
