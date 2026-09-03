from __future__ import annotations

import argparse

from .training import train_distillation
from .utils import apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RAD on MiniGrid Memory histories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pretrain-checkpoint")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    train_distillation(
        config,
        args.data_root,
        args.run_dir,
        model_kind="RAD",
        pretrain_checkpoint=args.pretrain_checkpoint,
    )


if __name__ == "__main__":
    main()
