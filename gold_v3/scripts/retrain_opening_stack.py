"""
Sequential retrain helper for the upgraded two-lens stack.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(mode: str, config_path: Path) -> None:
    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        mode,
        "--config",
        str(config_path),
    ]
    print(f"\n=== Running: {' '.join(cmd)} ===")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain Lens 1 and Lens 2 with the upgraded opening-intent stack.")
    parser.add_argument(
        "--config",
        default="config/retrain_fast.yaml",
        help="Config path to use for retraining.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    for mode in ("train-lens1", "analyse-lens1", "train-lens2"):
        _run(mode, config_path)


if __name__ == "__main__":
    main()
