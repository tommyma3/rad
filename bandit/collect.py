"""Collect chronological UCB histories; run directly or as python -m bandit.collect."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from bandit.env import make_manifest
from bandit.env.task_sampler import SAMPLER_VERSION
from bandit.rollout import generate_history
from bandit.utils import SCHEMA, get_config, project_path, write_json


def collect_dataset(output, config, tasks=10000, validation_tasks=1000, seed=0):
    if min(tasks, validation_tasks) < 1:
        raise ValueError("Training and validation counts must be positive")
    delays = config["train_delays"]
    pre_choices = config.get("pre_steps_choices", [config["pre_steps"]])
    horizon = config["pre_steps"] + config["post_steps"]
    if not delays or any(int(d) != d or d < 0 for d in delays):
        raise ValueError("train_delays must contain nonnegative integers")
    if not pre_choices or any(int(p) != p or not 0 < p < horizon for p in pre_choices):
        raise ValueError("Insertion points must leave both bandit phases nonempty")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Collection directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    settings = {"schema": SCHEMA, "sampler": SAMPLER_VERSION, "seed": seed, "config": config}
    write_json(output / "collection.json", settings)
    roots = np.random.SeedSequence(seed).spawn(2)
    for split, count, root_seed in zip(("train", "validation"), (tasks, validation_tasks), roots):
        task_root, rollout_root = root_seed.spawn(2)
        manifest = make_manifest(int(task_root.generate_state(1, dtype=np.uint64)[0]), count,
                                 config.get("distribution", "uniform"), config["num_arms"],
                                 split, reward_std=config["reward_std"])
        rollout_seeds = rollout_root.spawn(count)
        records = []
        partial = output / f"{split}.hdf5.partial"
        with h5py.File(partial, "w") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["split"] = split
            handle.attrs["config"] = json.dumps(settings)
            for i, (task, rollout_seed) in enumerate(zip(manifest, rollout_seeds)):
                reward_seed, learner_seed, distractor_seed, schedule_seed = [
                    int(s.generate_state(1, dtype=np.uint64)[0]) for s in rollout_seed.spawn(4)]
                schedule = np.random.default_rng(schedule_seed)
                delay = int(schedule.choice(config["train_delays"]))
                pre = int(schedule.choice(config.get("pre_steps_choices", [config["pre_steps"]])))
                post = config["pre_steps"] + config["post_steps"] - pre
                arrays, _ = generate_history(task, delay=delay, pre_steps=pre, post_steps=post,
                                              reward_seed=reward_seed, learner_seed=learner_seed,
                                              distractor_seed=distractor_seed,
                                              exploration_coefficient=config["exploration_coefficient"])
                record = {"task": task.to_dict(), "delay": delay, "pre_steps": pre,
                          "post_steps": post, "reward_seed": reward_seed,
                          "learner_seed": learner_seed, "distractor_seed": distractor_seed}
                records.append(record)
                group = handle.create_group(task.task_id)
                group.attrs["metadata"] = json.dumps(record)
                for key, array in arrays.items():
                    group.create_dataset(key, data=array, compression="gzip")
                if (i + 1) % 1000 == 0:
                    print(f"{split}: {i + 1}/{count}", flush=True)
        partial.replace(output / f"{split}.hdf5")
        write_json(output / f"{split}_manifest.json", {**settings, "split": split, "histories": records})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="delayed_adversarial_bandit")
    parser.add_argument("--output", default="datasets/delayed")
    parser.add_argument("--tasks", type=int, default=10000)
    parser.add_argument("--validation_tasks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delays", type=int, nargs="+")
    parser.add_argument("--pre_steps_choices", type=int, nargs="+")
    args = parser.parse_args()
    config = get_config(f"config/env/{args.env}.yaml")
    config.update(get_config("config/algorithm/ucb.yaml"))
    if args.delays is not None:
        config["train_delays"] = args.delays
    if args.pre_steps_choices is not None:
        config["pre_steps_choices"] = args.pre_steps_choices
    output = collect_dataset(project_path(args.output), config, args.tasks, args.validation_tasks, args.seed)
    print(f"Saved collection to {output}")


if __name__ == "__main__":
    main()
