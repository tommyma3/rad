"""Lazy task-history datasets for image-based AD and RAD."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from artifacts import TRAJECTORY_FORMAT


@dataclass(frozen=True)
class _EpisodeRef:
    path: Path
    episode_key: str
    length: int


@dataclass(frozen=True)
class _TaskHistory:
    path: Path
    task_id: str
    split: str
    source_algorithm: str
    episodes: tuple[_EpisodeRef, ...]
    offsets: tuple[int, ...]
    length: int


def _discover_histories(
    root: str | Path,
    split: str,
    source_algorithm: str,
) -> list[_TaskHistory]:
    histories: list[_TaskHistory] = []
    for path in sorted(Path(root).glob(f"{split}/{source_algorithm}/*.hdf5")):
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format") != TRAJECTORY_FORMAT:
                raise ValueError(f"Unsupported trajectory format in {path}")
            task_spec = json.loads(handle.attrs["task_spec"])
            if task_spec["split"] != split:
                raise ValueError(f"Split mismatch in {path}")
            if handle.attrs["source_algorithm"] != source_algorithm:
                raise ValueError(f"Source-algorithm mismatch in {path}")
            streams = handle["streams"]
            for stream_key in sorted(streams.keys()):
                episodes: list[_EpisodeRef] = []
                offsets: list[int] = []
                total = 0
                episode_root = streams[stream_key]["episodes"]
                for key in sorted(episode_root.keys()):
                    length = int(episode_root[key]["actions"].shape[0])
                    if length <= 0:
                        continue
                    offsets.append(total)
                    episodes.append(_EpisodeRef(path, f"{stream_key}/{key}", length))
                    total += length
                if episodes:
                    histories.append(
                        _TaskHistory(
                            path=path,
                            task_id=f"{task_spec['task_id']}:stream-{stream_key}",
                            split=split,
                            source_algorithm=source_algorithm,
                            episodes=tuple(episodes),
                            offsets=tuple(offsets),
                            length=total,
                        )
                    )
    if not histories:
        raise ValueError(
            f"No {source_algorithm!r} task histories found under {Path(root) / split / source_algorithm}"
        )
    task_ids = [history.task_id for history in histories]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Each artifact file must contain a unique task_id")
    return histories


class _LazyHistoryReader:
    def __init__(self) -> None:
        self._handles: dict[Path, h5py.File] = {}

    def handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r", swmr=True)
            self._handles[path] = handle
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        self.close()


class TaskWindowDataset(Dataset):
    """Windows over complete source-learning histories without crossing tasks.

    Windows may cross episode boundaries of the same task. The ``dones`` field
    marks those boundaries while preserving the chronological learning history.
    """

    def __init__(
        self,
        config: dict,
        root: str | Path,
        split: str,
        *,
        context_length: int | None = None,
    ) -> None:
        self.config = config
        self.context_length = int(context_length or config["n_transit"])
        self.histories = _discover_histories(root, split, config["source_algorithm"])
        self._reader: _LazyHistoryReader | None = None
        self._windows: list[tuple[int, int]] = []
        stride = int(config.get("dataset_stride", 1))
        for history_index, history in enumerate(self.histories):
            for start in range(0, history.length - self.context_length + 1, stride):
                self._windows.append((history_index, start))

    @property
    def reader(self) -> _LazyHistoryReader:
        if self._reader is None:
            self._reader = _LazyHistoryReader()
        return self._reader

    def __len__(self) -> int:
        return len(self._windows)

    def _episode_for_offset(self, history: _TaskHistory, offset: int) -> tuple[int, int]:
        episode_index = int(np.searchsorted(history.offsets, offset, side="right") - 1)
        return episode_index, offset - history.offsets[episode_index]

    def read_window(self, history_index: int, start: int, length: int) -> dict[str, np.ndarray]:
        history = self.histories[history_index]
        remaining = length
        offset = start
        images: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        learner_steps: list[np.ndarray] = []
        while remaining:
            episode_index, local_start = self._episode_for_offset(history, offset)
            episode_ref = history.episodes[episode_index]
            take = min(remaining, episode_ref.length - local_start)
            stream_key, episode_key = episode_ref.episode_key.split("/", maxsplit=1)
            group = self.reader.handle(history.path)["streams"][stream_key]["episodes"][episode_key]
            images.append(group["images"][local_start : local_start + take])
            actions.append(group["actions"][local_start : local_start + take])
            rewards.append(group["rewards"][local_start : local_start + take])
            terminated = group["terminated"][local_start : local_start + take]
            truncated = group["truncated"][local_start : local_start + take]
            dones.append(np.logical_or(terminated, truncated))
            learner_steps.append(group["learner_steps"][local_start : local_start + take])
            offset += take
            remaining -= take
        return {
            "states": np.concatenate(images, axis=0),
            "actions": np.concatenate(actions, axis=0).astype(np.int64),
            "rewards": np.concatenate(rewards, axis=0).astype(np.float32),
            "dones": np.concatenate(dones, axis=0).astype(np.float32),
            "learner_steps": np.concatenate(learner_steps, axis=0).astype(np.int64),
            "task_id": history.task_id,
            "context_length": np.int64(length),
        }

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        history_index, start = self._windows[index]
        return self.read_window(history_index, start, self.context_length)


class ADDataset(TaskWindowDataset):
    pass


class RADDataset(TaskWindowDataset):
    """Variable-length task-history windows for recurrent compression."""

    def __init__(self, config: dict, root: str | Path, split: str) -> None:
        self.min_context_length = int(config.get("min_context_length", config["n_transit"]))
        self.max_context_length = int(config.get("max_context_length", self.min_context_length))
        super().__init__(config, root, split, context_length=self.min_context_length)
        self.max_compressions = config.get("max_compressions")

    def available_compression_buckets(self) -> list[int]:
        n_transit = int(self.config["n_transit"])
        short_keep = int(self.config["short_memory_keep"])
        refill = max(1, n_transit - short_keep)
        maximum = max(0, (self.max_context_length - n_transit + refill - 1) // refill)
        buckets = list(range(maximum + 1))
        if self.max_compressions is not None:
            buckets = [bucket for bucket in buckets if bucket <= int(self.max_compressions)]
        return buckets or [0]

    def length_for_bucket(self, bucket: int) -> int:
        n_transit = int(self.config["n_transit"])
        short_keep = int(self.config["short_memory_keep"])
        return min(self.max_context_length, n_transit + bucket * (n_transit - short_keep))

    def __getitem__(self, index) -> dict[str, np.ndarray]:
        if isinstance(index, tuple):
            base_index, bucket = index
        else:
            base_index = index
            bucket = random.choice(self.available_compression_buckets())
        history_index, nominal_start = self._windows[int(base_index)]
        length = self.length_for_bucket(int(bucket))
        history = self.histories[history_index]
        max_start = max(0, history.length - length)
        start = min(nominal_start, max_start)
        return self.read_window(history_index, start, min(length, history.length))


class CompressionPretrainDataset(RADDataset):
    pass


class CompressionBucketBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(self, dataset: RADDataset, batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        buckets = self.dataset.available_compression_buckets()
        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            bucket = random.choice(buckets)
            yield [(index, bucket) for index in indices[start : start + self.batch_size]]


def collate_trajectories(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    max_length = max(len(item["actions"]) for item in batch)
    batch_size = len(batch)
    height, width, channels = batch[0]["states"].shape[1:]
    states = np.zeros((batch_size, max_length, height, width, channels), dtype=np.uint8)
    actions = np.zeros((batch_size, max_length), dtype=np.int64)
    rewards = np.zeros((batch_size, max_length), dtype=np.float32)
    dones = np.zeros((batch_size, max_length), dtype=np.float32)
    valid = np.zeros((batch_size, max_length), dtype=np.bool_)
    learner_steps = np.zeros((batch_size, max_length), dtype=np.int64)
    for row, item in enumerate(batch):
        length = len(item["actions"])
        states[row, :length] = item["states"]
        actions[row, :length] = item["actions"]
        rewards[row, :length] = item["rewards"]
        dones[row, :length] = item["dones"]
        learner_steps[row, :length] = item["learner_steps"]
        valid[row, :length] = True
    return {
        "states": torch.from_numpy(states),
        "actions": torch.from_numpy(actions),
        "rewards": torch.from_numpy(rewards),
        "dones": torch.from_numpy(dones),
        "valid_mask": torch.from_numpy(valid),
        "learner_steps": torch.from_numpy(learner_steps),
    }
