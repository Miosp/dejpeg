"""Pytest fixtures: skip weight-dependent tests gracefully in CI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def weights_dir() -> Path | None:
    d = os.environ.get("FBCNN_WEIGHTS_DIR")
    return Path(d) if d else None


@pytest.fixture(scope="session")
def testsets_dir() -> Path | None:
    d = os.environ.get("FBCNN_TESTSETS_DIR")
    return Path(d) if d else None


def skip_without_weights(weights_dir: Path | None) -> None:
    if weights_dir is None or not weights_dir.exists():
        pytest.skip("FBCNN_WEIGHTS_DIR unset or missing; skipping weight-dependent test")


def skip_without_testsets(testsets_dir: Path | None) -> None:
    if testsets_dir is None or not testsets_dir.exists():
        pytest.skip("FBCNN_TESTSETS_DIR unset or missing; skipping PSNR test")
