"""Episode-bounded AD/RAD datasets for MiniGrid Memory histories."""

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

from .artifacts import TRAJECTORY_FORMAT


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    key: str
    length: int
    task_id: str
    split: str


def discover_episodes(
    root: str | Path,
    split: str,
    source_algorithm: str,
) -> list[EpisodeRef]:
    episodes: list[EpisodeRef] = []
    for path in sorted(Path(root).glob(f"{split}/{source_algorithm}/*.hdf5")):
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format") != TRAJECTORY_FORMAT:
                raise ValueError(f"Unsupported trajectory format in {path}")
            spec = json.loads(handle.attrs["task_spec"])
            if spec["split"] != split:
                raise ValueError(f"Split mismatch in {path}")
            if handle.attrs["source_algorithm"] != source_algorithm:
                raise ValueError(f"Source-algorithm mismatch in {path}")
            for key in sorted(handle["episodes"].keys()):
                group = handle["episodes"][key]
                length = int(group["actions"].shape[0])
                if length:
                    episodes.append(EpisodeRef(path, key, length, spec["task_id"], split))
    if not episodes:
        raise ValueError(f"No {split}/{source_algorithm} episodes found under {root}")
    return episodes


class _LazyFiles:
    def __init__(self) -> None:
        self.handles: dict[Path, h5py.File] = {}

    def get(self, path: Path) -> h5py.File:
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r", swmr=True)
        return self.handles[path]

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __del__(self):
        self.close()


class ADDataset(Dataset):
    """Fixed-capacity windows that never cross an episode boundary."""

    def __init__(self, config: dict, root: str | Path, split: str) -> None:
        self.config = config
        self.context_length = int(config["n_transit"])
        self.stride = int(config.get("dataset_stride", self.context_length))
        self.decision_window_repeats = int(config.get("decision_window_repeats", 1))
        if self.decision_window_repeats < 1:
            raise ValueError("decision_window_repeats must be at least one")
        self.episodes = discover_episodes(root, split, config["source_algorithm"])
        self._files: _LazyFiles | None = None
        self.windows: list[tuple[int, int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            if episode.length <= self.context_length:
                final_window = (episode_index, 0, episode.length)
                self.windows.extend([final_window] * self.decision_window_repeats)
                continue
            ends = list(range(self.context_length, episode.length + 1, self.stride))
            if ends[-1] != episode.length:
                ends.append(episode.length)
            episode_windows = [
                (episode_index, end - self.context_length, end) for end in ends
            ]
            self.windows.extend(episode_windows)
            self.windows.extend([episode_windows[-1]] * (self.decision_window_repeats - 1))

    @property
    def files(self) -> _LazyFiles:
        if self._files is None:
            self._files = _LazyFiles()
        return self._files

    def __len__(self) -> int:
        return len(self.windows)

    def close(self) -> None:
        if self._files is not None:
            self._files.close()
            self._files = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _read(self, episode_index: int, start: int, end: int) -> dict[str, np.ndarray | str]:
        episode = self.episodes[episode_index]
        group = self.files.get(episode.path)["episodes"][episode.key]
        keys = (
            "images",
            "directions",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "cue_ids",
            "cue_visible",
            "decision",
            "success",
            "learner_steps",
        )
        item = {key: group[key][start:end] for key in keys}
        item["task_id"] = episode.task_id
        item["episode_key"] = episode.key
        item["context_length"] = np.int64(end - start)
        return item

    def __getitem__(self, index):
        episode_index, start, end = self.windows[int(index)]
        return self._read(episode_index, start, end)


class RADDataset(ADDataset):
    """Variable history windows grouped by the number of compressions."""

    def __init__(self, config: dict, root: str | Path, split: str) -> None:
        self.max_context_length = int(config["max_context_length"])
        self.short_memory_keep = int(config["short_memory_keep"])
        self.max_compressions = config.get("max_compressions")
        super().__init__(config, root, split)

    def available_compression_buckets(self) -> list[int]:
        refill = max(1, self.context_length - self.short_memory_keep)
        maximum = max(
            0,
            (self.max_context_length - self.context_length + refill - 1) // refill,
        )
        if self.max_compressions is not None:
            maximum = min(maximum, int(self.max_compressions))
        return list(range(maximum + 1))

    def length_for_bucket(self, bucket: int) -> int:
        refill = max(1, self.context_length - self.short_memory_keep)
        return min(self.max_context_length, self.context_length + int(bucket) * refill)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            base_index, bucket = index
        else:
            base_index = int(index)
            bucket = random.choice(self.available_compression_buckets())
        episode_index, _, nominal_end = self.windows[int(base_index)]
        episode = self.episodes[episode_index]
        length = min(self.length_for_bucket(int(bucket)), episode.length)
        end = min(episode.length, max(length, nominal_end))
        start = end - length
        return self._read(episode_index, start, end)


class CompressionPretrainDataset(RADDataset):
    pass


class CompressionBucketBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(
        self,
        dataset: RADDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)
        buckets = self.dataset.available_compression_buckets()
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            bucket = rng.choice(buckets)
            yield [(index, bucket) for index in batch]


def collate_trajectories(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    max_length = max(len(item["actions"]) for item in batch)
    batch_size = len(batch)
    image_shape = batch[0]["images"].shape[1:]
    images = np.zeros((batch_size, max_length, *image_shape), dtype=np.uint8)
    directions = np.zeros((batch_size, max_length), dtype=np.int64)
    actions = np.zeros((batch_size, max_length), dtype=np.int64)
    rewards = np.zeros((batch_size, max_length), dtype=np.float32)
    terminated = np.zeros((batch_size, max_length), dtype=np.float32)
    truncated = np.zeros((batch_size, max_length), dtype=np.float32)
    valid = np.zeros((batch_size, max_length), dtype=np.bool_)
    decision = np.zeros((batch_size, max_length), dtype=np.bool_)
    cue_ids = np.full((batch_size, max_length), -1, dtype=np.int64)
    cue_visible = np.zeros((batch_size, max_length), dtype=np.bool_)
    success = np.zeros((batch_size, max_length), dtype=np.bool_)
    for row, item in enumerate(batch):
        length = len(item["actions"])
        selection = (row, slice(0, length))
        images[selection] = item["images"]
        directions[selection] = item["directions"]
        actions[selection] = item["actions"]
        rewards[selection] = item["rewards"]
        terminated[selection] = item["terminated"]
        truncated[selection] = item["truncated"]
        decision[selection] = item["decision"]
        cue_ids[selection] = item["cue_ids"]
        cue_visible[selection] = item["cue_visible"]
        success[selection] = item["success"]
        valid[selection] = True
    return {
        "images": torch.from_numpy(images),
        "directions": torch.from_numpy(directions),
        "actions": torch.from_numpy(actions),
        "rewards": torch.from_numpy(rewards),
        "terminated": torch.from_numpy(terminated),
        "truncated": torch.from_numpy(truncated),
        "valid_mask": torch.from_numpy(valid),
        "decision_mask": torch.from_numpy(decision),
        "cue_ids": torch.from_numpy(cue_ids),
        "cue_visible": torch.from_numpy(cue_visible),
        "success": torch.from_numpy(success),
        "task_ids": [str(item["task_id"]) for item in batch],
    }
