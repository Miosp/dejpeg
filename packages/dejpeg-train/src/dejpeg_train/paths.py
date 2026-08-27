"""Env-driven path configuration for training scripts.

Every script resolves data and output locations through this module, so the
tree is machine-independent and configured on the fly via two variables:

  DEJPEG_WORK_ROOT   outputs: checkpoints, logs, phase/run dirs   (default ~/dejpeg-work)
  DEJPEG_DATA_ROOT   datasets: shards, raw, manifest, eval sets   (default $DEJPEG_WORK_ROOT/data)

Keep large data on a native Linux filesystem (ext4) — never on drvfs mounts
like /mnt/c: random I/O there starves the dataloader.
"""
from __future__ import annotations

import os
from pathlib import Path


def work_root() -> Path:
    return Path(os.environ.get("DEJPEG_WORK_ROOT", str(Path.home() / "dejpeg-work")))


def data_root() -> Path:
    return Path(os.environ.get("DEJPEG_DATA_ROOT", str(work_root() / "data")))


def shards_dir() -> Path:
    return data_root() / "shards"


def raw_dir() -> Path:
    return data_root() / "raw"


def manifest_path() -> Path:
    return data_root() / "manifest.sqlite"


def eval_sets_dir() -> Path:
    return data_root() / "eval_sets"


def weights_dir() -> Path:
    return data_root() / "weights"


def testsets_dir(name: str) -> Path:
    """FBCNN upstream benchmark testsets (Classic5, LIVE1_color)."""
    return data_root() / "fbcnn-upstream" / "testsets" / name


def phase_dir(name: str) -> Path:
    """Experiment output dir ($DEJPEG_WORK_ROOT/<name>)."""
    return work_root() / name
