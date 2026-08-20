"""Launch the four RAD latent-update variants for one source algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


MODES = ("replace", "residual", "multiplicative_gate", "gru_gate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("ppo", "dreamer_tbtt"), required=True)
    parser.add_argument("--pretrained-compression", type=Path, required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    for mode in MODES:
        config = Path("config/model") / f"rad_{args.source}_{mode}.yaml"
        command = [
            "accelerate",
            "launch",
            "train_rad.py",
            "--config",
            str(config),
            "--pretrained-compression",
            str(args.pretrained_compression),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
