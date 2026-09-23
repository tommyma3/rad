"""Isolated controlled old/recent evidence protocol, data, and model contracts.

Forced exploration is context, never an action-prediction target. Each task has
one query supervised by the empirical best arm using both evidence blocks.
All arms have equal counts, so this is also the existing DPT-UCB recommendation.
"""
from pathlib import Path

import numpy as np
import torch

from .model import make_model
from .utils import file_digest, get_config, load_config, write_json

PROTOCOL = "old-recent-evidence-v1"
DEFAULT_CONFIG = "config/experiments/old_recent.yaml"
METHODS = ("rad", "ad_short", "ad_long")
CONDITIONS = ("both", "early_only", "recent_only")
SPLITS = ("train", "validation", "test")


def geometry(study):
    """Finish the gap exactly at a RAD compression event, before recent data."""
    arms, pulls = study["num_arms"], study["pulls_per_arm"]
    context, keep = study["context_steps"], study["short_memory_keep"]
    rounds = study["gap_compressions"]
    for name, value in (("num_arms", arms), ("pulls_per_arm", pulls),
                        ("context_steps", context), ("gap_compressions", rounds)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if arms < 2 or arms % 2:
        raise ValueError("Equal disjoint arm subsets require an even num_arms >= 2")
    if type(keep) is not int or not 0 <= keep < context:
        raise ValueError("short_memory_keep must be in [0, context_steps)")
    block = arms // 2 * pulls
    recent_start = context + 1 + (rounds - 1) * (context + 1 - keep)
    delay = recent_start - block
    if block + keep > context or delay < keep or delay < 1:
        raise ValueError("Geometry must compress all early evidence and retain all recent evidence")
    return {"block_steps": block, "delay": delay, "recent_start": recent_start,
            "prefix_steps": recent_start + block, "rad_recent_steps": keep + block,
            "compression_count": rounds}


def validate_study(study):
    if study.get("experiment") != PROTOCOL:
        raise ValueError("Incompatible evidence experiment protocol")
    geometry(study)
    if not np.isfinite(study["reward_std"]) or study["reward_std"] <= 0:
        raise ValueError("reward_std must be positive and finite")
    if type(study["data_seed"]) is not int or study["data_seed"] < 0:
        raise ValueError("data_seed must be a nonnegative integer")
    for split in SPLITS:
        count = study[f"{split}_tasks"]
        if type(count) is not int or count < 2 or count % 2:
            raise ValueError(f"{split}_tasks must be positive and even for balanced strata")
    return study


def read_study(path=DEFAULT_CONFIG):
    return validate_study(get_config(path))


def collection_contract(study):
    return {key: study[key] for key in ("experiment", "num_arms", "reward_std",
            "pulls_per_arm", "context_steps", "short_memory_keep", "gap_compressions",
            "data_seed", "train_tasks", "validation_tasks", "test_tasks")}


def make_split(study, split):
    """Independent iid tasks; balance which block contains the true best arm.

    Each task's random partition is oriented to its assigned stratum. Means are
    used for that stratification and scoring only, never the supervised label.
    Reward potential outcomes are drawn per arm/sample independently of order.
    """
    validate_study(study)
    spec = geometry(study)
    count, arms = study[f"{split}_tasks"], study["num_arms"]
    block, start, length = spec["block_steps"], spec["recent_start"], spec["prefix_steps"]
    values = {
        "states": np.ones((count, length), dtype=np.int64),
        "actions": np.empty((count, length), dtype=np.int64),
        "rewards": np.zeros((count, length), dtype=np.float32),
        "filler_actions": np.empty((count, length), dtype=np.int64),
        "means": np.empty((count, arms), dtype=np.float64),
        "empirical_means": np.empty((count, arms), dtype=np.float64),
        "early_arms": np.empty((count, arms // 2), dtype=np.int64),
        "targets": np.empty(count, dtype=np.int64),
        "best_is_early": np.arange(count) % 2 == 0,
    }
    values["states"][:, :block] = 0
    values["states"][:, start:] = 0
    seeds = np.random.SeedSequence([study["data_seed"], SPLITS.index(split), 7301]).spawn(count)
    for index, seed in enumerate(seeds):
        task_seed, partition_seed, reward_seed, order_seed, filler_seed = seed.spawn(5)
        means = np.random.default_rng(task_seed).uniform(0, 1, arms)
        partition = np.random.default_rng(partition_seed).permutation(arms)
        half = arms // 2
        if (int(means.argmax()) in partition[:half]) != bool(values["best_is_early"][index]):
            partition = np.concatenate((partition[half:], partition[:half]))
        samples = np.random.default_rng(reward_seed).normal(
            means[:, None], study["reward_std"], (arms, study["pulls_per_arm"]))
        order = np.random.default_rng(order_seed)
        filler = np.random.default_rng(filler_seed).integers(arms, size=length)
        values["filler_actions"][index] = filler
        values["actions"][index] = filler
        for subset, offset in ((partition[:half], 0), (partition[half:], start)):
            # Each arm is sampled equally; within-block order carries no policy cue.
            pairs = [(int(arm), sample) for sample in range(study["pulls_per_arm"]) for arm in subset]
            order.shuffle(pairs)
            for step, (arm, sample) in enumerate(pairs, offset):
                values["actions"][index, step] = arm
                values["rewards"][index, step] = samples[arm, sample]
        # Compute labels from the actual float32 evidence supplied to the models.
        genuine = values["states"][index] == 0
        observed_actions = values["actions"][index, genuine]
        observed_rewards = values["rewards"][index, genuine]
        empirical = np.bincount(observed_actions, weights=observed_rewards, minlength=arms) / study["pulls_per_arm"]
        values["means"][index] = means
        values["empirical_means"][index] = empirical
        values["early_arms"][index] = partition[:half]
        values["targets"][index] = empirical.argmax()
    return values


def prepare_data(output, study):
    """Immutable shared data prepared once before GPU workers are launched."""
    import json

    output = Path(output)
    contract = collection_contract(validate_study(study))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Existing dataset uses a different evidence protocol/configuration")
        for split in SPLITS:
            if file_digest(output / f"{split}.npz") != manifest["digests"][split]:
                raise ValueError(f"Changed evidence data: {split}")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete/nonempty dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    digests = {}
    for split in SPLITS:
        target = output / f"{split}.npz"
        partial = output / f"{split}.npz.partial"
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **make_split(study, split))
        partial.replace(target)
        digests[split] = file_digest(target)
    manifest = {"protocol": PROTOCOL, "contract": contract, "geometry": geometry(study), "digests": digests}
    write_json(manifest_path, manifest)
    return manifest


class EvidenceDataset:
    def __init__(self, path, study):
        self.study, self.spec = study, geometry(study)
        with np.load(path, allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        self.size = len(self.arrays["targets"])
        if self.arrays["states"].shape != (self.size, self.spec["prefix_steps"]):
            raise ValueError("Evidence dataset geometry mismatch")

    def batch(self, indices, condition="both", context_steps=None, device="cpu"):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown evidence condition: {condition}")
        indices = np.asarray(indices)
        values = {key: self.arrays[key][indices].copy() for key in ("states", "actions", "rewards")}
        if condition != "both":
            removed = (slice(self.spec["recent_start"], None) if condition == "early_only"
                       else slice(0, self.spec["block_steps"]))
            # Replace the entire transition, preserving length/compression schedule.
            values["states"][:, removed] = 1
            values["actions"][:, removed] = self.arrays["filler_actions"][indices, removed]
            values["rewards"][:, removed] = 0
        if context_steps is not None:
            values = {key: value[:, -context_steps:] for key, value in values.items()}
        values.update(query_states=np.zeros(len(indices), dtype=np.int64),
                      targets=self.arrays["targets"][indices], loss_mask=np.ones(len(indices), dtype=bool))
        # No means, partition/stratum labels, or target action/reward in the prefix.
        return {key: torch.as_tensor(value, device=device) for key, value in values.items()}


def model_config(study, method, seed):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    config = load_config(method)
    config.update(study.get("training", {}))
    config.update(experiment=PROTOCOL, num_arms=study["num_arms"], seed=seed,
                  context_steps=geometry(study)["prefix_steps"] if method == "ad_long" else study["context_steps"],
                  short_memory_keep=study["short_memory_keep"], always_use_latent_prefix=False,
                  max_gradient_rounds=None)
    return config


def save_payload(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_evidence_checkpoint(path, device="cpu"):
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Expected an old/recent evidence checkpoint, not a legacy bandit checkpoint")
    expected = model_config(payload["study"], payload["method"], payload["seed"])
    if payload["config"] != expected:
        raise ValueError("Checkpoint model configuration differs from its study contract")
    model = make_model(payload["config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload
