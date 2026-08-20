"""Shared AD/RAD training utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CompressionBucketBatchSampler, collate_trajectories


def make_loader(dataset, config: dict, batch_size_key: str, *, shuffle: bool = True):
    workers = int(config.get("num_workers", 4))
    batch_size = int(config[batch_size_key])
    if isinstance(dataset, __import__("dataset").RADDataset) and config.get(
        "rad_batching_strategy", "compression_buckets"
    ) == "compression_buckets":
        sampler = CompressionBucketBatchSampler(dataset, batch_size, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_trajectories,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_trajectories,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def infinite_batches(loader):
    while True:
        yield from loader


def latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = sorted(directory.glob("ckpt-*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint_atomic(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
