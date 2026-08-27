"""Phase 0 Task 0.4 (raw ingest front-end) -- tests.

``develop_raw`` is mocked so the dedup/idempotency/lossless logic is verifiable
without a real raw file. Real-file development is exercised once the user points
ingest at their raws on ext4.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dejpeg_train.data import ingest_raws as ir


def test_content_hash_deterministic_and_distinct(tmp_path):
    a = tmp_path / "a.raw"
    b = tmp_path / "b.raw"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert ir._content_hash(a) == ir._content_hash(a)
    assert ir._content_hash(a) != ir._content_hash(b)


def test_perceptual_hash_deterministic():
    rng = np.random.RandomState(0)
    arr = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
    assert ir.perceptual_hash(arr) == ir.perceptual_hash(arr)


def test_downscale_noop_when_small():
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    assert ir.downscale_to(arr, 4096).shape == (100, 200, 3)


def test_downscale_caps_longest_side():
    arr = np.zeros((8000, 4000, 3), dtype=np.uint8)
    out = ir.downscale_to(arr, 4096)
    assert max(out.shape[:2]) <= 4096


def test_ingest_idempotent_and_content_dedup(tmp_path, monkeypatch):
    src = tmp_path / "raws"
    out = tmp_path / "out"
    src.mkdir()
    (src / "a.dng").write_bytes(b"raw-a")
    (src / "b.dng").write_bytes(b"raw-b")
    (src / "dup.dng").write_bytes(b"raw-a")  # same bytes -> same content hash

    def fake_develop(path, *, sixteen_bit=False):
        r = np.random.RandomState(abs(hash(path.name)) % 100000)
        return (r.rand(256, 384, 3) * 255).astype(np.uint8)

    monkeypatch.setattr(ir, "develop_raw", fake_develop)
    s1 = ir.ingest(src, out, max_long_side=9999)
    assert s1["developed"] == 2          # a + b developed
    assert s1["skipped_existing"] == 1   # dup matched a's content hash
    assert (out / "raw_manifest.sqlite").exists()

    s2 = ir.ingest(src, out, max_long_side=9999)  # rerun -> nothing new
    assert s2["developed"] == 0
    assert s2["skipped_existing"] == 3


def test_ingest_perceptual_dedup(tmp_path, monkeypatch):
    src = tmp_path / "r"
    out = tmp_path / "o"
    src.mkdir()
    (src / "a.dng").write_bytes(b"X")   # distinct content hashes
    (src / "b.dng").write_bytes(b"Y")
    arr = (np.random.RandomState(1).rand(256, 256, 3) * 255).astype(np.uint8)
    monkeypatch.setattr(ir, "develop_raw", lambda path, *, sixteen_bit=False: arr.copy())
    s = ir.ingest(src, out, max_long_side=9999)
    assert s["developed"] == 1
    assert s["skipped_dup"] == 1


def test_ingest_output_is_lossless(tmp_path, monkeypatch):
    src = tmp_path / "r"
    out = tmp_path / "o"
    src.mkdir()
    (src / "a.dng").write_bytes(b"Z")
    arr = (np.random.RandomState(2).rand(300, 300, 3) * 255).astype(np.uint8)
    monkeypatch.setattr(ir, "develop_raw", lambda path, *, sixteen_bit=False: arr.copy())
    ir.ingest(src, out, max_long_side=9999)
    webps = list(out.glob("*.webp"))
    assert len(webps) == 1
    back = np.array(Image.open(webps[0]).convert("RGB"))
    assert back.shape == arr.shape
    assert np.array_equal(back, arr), "lossless WebP output must round-trip bit-identically"
