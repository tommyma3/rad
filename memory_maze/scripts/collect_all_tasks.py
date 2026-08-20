"""Dispatch source learning independently for every task manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    task_paths = sorted(args.tasks.glob("*.json"))
    if not task_paths:
        raise ValueError(f"No task manifests found in {args.tasks}")
    for task_path in task_paths:
        command = [
            sys.executable,
            "collect.py",
            "--config",
            str(args.config),
            "--task",
            str(task_path),
            "--output",
            str(args.output),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
