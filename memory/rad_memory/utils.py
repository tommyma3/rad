"""Configuration, reproducibility, and checkpoint helpers."""

from __future__ import annotations

import ast
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    include = config.pop("include", None)
    if include is None:
        return config
    include_path = Path(include)
    if not include_path.is_absolute():
        local = path.parent / include_path
        include_path = local if local.exists() else Path.cwd() / include_path
    merged = load_config(include_path)
    merged.update(config)
    return merged


def parse_override(value: str) -> tuple[str, Any]:
    key, separator, raw = value.partition("=")
    if not separator or not key:
        raise ValueError(f"Override must be KEY=VALUE, got {value!r}")
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = raw
    return key, parsed


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = dict(config)
    for value in overrides:
        key, parsed = parse_override(value)
        result[key] = parsed
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_step(path: str | Path) -> int:
    match = re.search(r"(?:ckpt|checkpoint)[-_](\d+)", Path(path).stem)
    return int(match.group(1)) if match else -1


def latest_checkpoint(directory: str | Path) -> Path | None:
    paths = list(Path(directory).glob("ckpt-*.pt"))
    return max(paths, key=checkpoint_step) if paths else None


def save_checkpoint_atomic(payload: dict, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
