from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import yaml


CHECKPOINT_FORMAT = "memory-maze-sar-v1"


def get_config(path: str | Path) -> dict:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    include = config.pop("include", None)
    if include is None:
        return config
    include_path = Path(include)
    if not include_path.is_absolute():
        candidate = path.parent / include_path
        include_path = candidate if candidate.exists() else include_path
    merged = get_config(include_path)
    merged.update(config)
    return merged


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_state_dict(state_dict: dict) -> dict:
    return {
        key.replace("._orig_mod.", ".").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def checkpoint_payload(model, config: dict, **state) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "config": config,
        "model": normalize_state_dict(model.state_dict()),
        **state,
    }
