"""Query-ended prefixes, sampled uniformly over genuine action targets."""
from collections import defaultdict
import json

import h5py
import numpy as np
import torch

from .utils import SCHEMA
from .env import BanditTask


class BanditDataset:
    def __init__(self, path, context_steps=None, expected_split=None):
        self.histories = []
        self.buckets = defaultdict(list)
        self.context_steps = context_steps
        self.task_ids = set()
        self.task_signatures = set()
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema") != SCHEMA:
                raise ValueError("Incompatible trajectory schema")
            if expected_split and handle.attrs.get("split") != expected_split:
                raise ValueError(f"Expected {expected_split} data")
            self.collection_config = json.loads(handle.attrs["config"])["config"]
            self.num_arms = self.collection_config["num_arms"]
            for task_id in sorted(handle.keys()):
                group = handle[task_id]
                metadata = json.loads(group.attrs["metadata"])
                task = BanditTask.from_dict(metadata["task"])
                if task.reward_std != self.collection_config["reward_std"]:
                    raise ValueError("Task noise does not match collection settings")
                history = {key: group[key][()] for key in ("states", "actions", "rewards", "loss_mask")}
                if len({len(v) for v in history.values()}) != 1:
                    raise ValueError("Unequal trajectory array lengths")
                if not np.array_equal(history["loss_mask"], history["states"] == 0):
                    raise ValueError("Supervision mask must select genuine bandit pulls")
                expected_states = np.concatenate((np.zeros(metadata["pre_steps"]),
                                                  np.ones(metadata["delay"]),
                                                  np.zeros(metadata["post_steps"])))
                if not np.array_equal(history["states"], expected_states):
                    raise ValueError("Trajectory does not match its phase boundaries")
                if (len(metadata["task"]["means"]) != self.num_arms or
                        np.any((history["actions"] < 0) | (history["actions"] >= self.num_arms))):
                    raise ValueError("Trajectory actions or task means do not match num_arms")
                if (not np.all(np.isfinite(history["rewards"])) or
                        np.any(history["rewards"][~history["loss_mask"]] != 0)):
                    raise ValueError("Nonfinite rewards or nonzero distractor rewards")
                history_index = len(self.histories)
                self.histories.append(history)
                self.task_ids.add(task_id)
                self.task_signatures.add(tuple(metadata["task"]["means"]))
                for target in np.flatnonzero(history["loss_mask"]):
                    length = int(target) if context_steps is None else min(int(target), context_steps)
                    self.buckets[length].append((history_index, int(target)))
        if not self.buckets:
            raise ValueError("Dataset contains no bandit targets")
        self.lengths = sorted(self.buckets)
        counts = np.asarray([len(self.buckets[n]) for n in self.lengths], dtype=float)
        self.length_probabilities = counts / counts.sum()
        self.num_targets = int(counts.sum())

    def sample_batch(self, batch_size, rng, pretrain=False):
        lengths = self.lengths
        probabilities = self.length_probabilities
        if pretrain:
            indices = [i for i, length in enumerate(lengths) if length > 0]
            if not indices:
                raise ValueError("Pretraining requires nonempty histories")
            lengths = [lengths[i] for i in indices]
            probabilities = probabilities[indices]
            probabilities = probabilities / probabilities.sum()
        length = int(rng.choice(lengths, p=probabilities))
        candidates = self.buckets[length]
        rows = [candidates[int(i)] for i in rng.integers(len(candidates), size=batch_size)]
        # Same-length buckets avoid padding entering a recurrent compressor.
        batch = {key: [] for key in ("states", "actions", "rewards", "query_states", "targets", "loss_mask")}
        for history_index, target in rows:
            history = self.histories[history_index]
            for key in ("states", "actions", "rewards"):
                batch[key].append(history[key][target-length:target])
            batch["query_states"].append(history["states"][target])
            batch["targets"].append(history["actions"][target])
            batch["loss_mask"].append(True)
        return {key: torch.as_tensor(np.asarray(value)) for key, value in batch.items()}


def assert_disjoint(train, validation):
    if train.task_ids & validation.task_ids or train.task_signatures & validation.task_signatures:
        raise ValueError("Training and validation tasks overlap")
