from __future__ import annotations

import argparse

from .training import train_compression
from .utils import apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the MiniGrid Memory compressor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    train_compression(config, args.data_root, args.run_dir)


if __name__ == "__main__":
    main()
