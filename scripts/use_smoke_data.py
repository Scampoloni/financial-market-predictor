"""
use_smoke_data.py — Swap raw data with the smoke dataset for quick reproducible runs.

Modes:
  --activate  Move data/raw to data/raw_full (if not already), then copy data/smoke to data/raw
  --restore   Restore data/raw from data/raw_full

Usage:
    python scripts/use_smoke_data.py --activate
    python scripts/use_smoke_data.py --restore
"""

from __future__ import annotations

import shutil
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FULL_DIR = ROOT / "data" / "raw_full"
SMOKE_DIR = ROOT / "data" / "smoke"


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def activate() -> None:
    if not SMOKE_DIR.exists():
        raise FileNotFoundError("Smoke dataset not found. Run scripts/build_smoke_dataset.py first.")
    if RAW_DIR.exists() and not RAW_FULL_DIR.exists():
        RAW_DIR.rename(RAW_FULL_DIR)
        print(f"Backed up raw data to {RAW_FULL_DIR}")
    _copy_tree(SMOKE_DIR, RAW_DIR)
    print(f"Activated smoke dataset at {RAW_DIR}")


def restore() -> None:
    if not RAW_FULL_DIR.exists():
        raise FileNotFoundError("Backup not found: data/raw_full")
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_FULL_DIR.rename(RAW_DIR)
    print(f"Restored full raw data to {RAW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap smoke dataset into data/raw")
    parser.add_argument("--activate", action="store_true", help="Activate smoke dataset")
    parser.add_argument("--restore", action="store_true", help="Restore full dataset")
    args = parser.parse_args()

    if args.activate == args.restore:
        raise SystemExit("Specify exactly one of --activate or --restore")

    if args.activate:
        activate()
    else:
        restore()


if __name__ == "__main__":
    main()
