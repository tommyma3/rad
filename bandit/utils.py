"""Project-relative configuration, manifests, and reproducible file contracts."""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCHEMA = "delayed-bandit-gaussian-sar-v2"


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_config(path, _seen=None):
    path = project_path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config include: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    include = value.pop("include", None)
    base = get_config(include, seen | {path}) if include else {}
    for key, item in value.items():
        if isinstance(item, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **item}
        else:
            base[key] = item
    return base


def load_config(model="ad_short", env="delayed_adversarial_bandit"):
    config = get_config(f"config/env/{env}.yaml")
    config.update(get_config("config/algorithm/ucb.yaml"))
    config.update(get_config(f"config/model/{model}.yaml"))
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
