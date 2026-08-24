"""Phase 0 Task 0.4 -- corpus manifest + prepare tests."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dejpeg.data import manifest as M
from dejpeg.data.prepare import PrepareConfig, _ANNEX_K_LUMA_50, estimate_qf, prepare_source


def _gradient(size=700):
    idx = (np.arange(size, dtype=np.float32) / size * 255).astype(np.uint8)
    g = (idx[:, None] + idx[None, :]) // 2
    return np.stack([g, g, g], axis=-1).astype(np.uint8)


# ----------------------------------------------------------------- manifest


def test_manifest_add_has_and_phash_dedup(tmp_path):
    conn = M.open_manifest(tmp_path / "m.db")
    h16 = "0" * 16
    M.add_entry(conn, content_hash="a" * 64, perceptual_hash=h16, shard_id="s0", source_corpus="df2k", patch_count=3)
    assert M.has_content(conn, "a" * 64)
    assert not M.has_content(conn, "b" * 64)
    near = "0000000000000001"  # 1 bit from all-zeros; but our stored is all-zeros -> 1 bit
    M.add_entry(conn, content_hash="c" * 64, perceptual_hash="0" * 16, shard_id="s1", source_corpus="df2k", patch_count=1)
    assert M.find_dup_by_phash(conn, "0000000000000001", threshold=5) is not None
    assert M.find_dup_by_phash(conn, "ffffffffffffffff", threshold=5) is None


# ------------------------------------------------------------- QF estimation


def _annex_k(qf):
    base = np.array(_ANNEX_K_LUMA_50, dtype=float)
    scale = 5000 / qf if qf < 50 else 200 - 2 * qf
    return np.round((base * scale + 50) / 100)


def test_estimate_qf_recovers_annex_k_quality():
    for qf in (30, 50, 75, 90):
        est = estimate_qf(_annex_k(qf))
        assert abs(est - qf) < 2, f"QF{qf} estimated as {est:.1f}"


# ------------------------------------------------------------- prepare core


def test_prepare_idempotent_and_writes_lossless_shards(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.fromarray(_gradient()).save(src / "a.png")
    shards = tmp_path / "shards"
    conn = M.open_manifest(tmp_path / "m.db")

    s1 = prepare_source(src, "t", shards, conn)
    assert s1["accepted"] == 1 and s1["patches"] > 0
    tar_files = list(shards.glob("t-*.tar"))
    assert len(tar_files) >= 1

    # lossless 640 patch round-trip
    import tarfile

    with tarfile.open(tar_files[0]) as t:
        webp = next(n for n in t.getnames() if n.endswith(".webp"))
        arr = np.array(Image.open(io.BytesIO(t.extractfile(webp).read())).convert("RGB"))
    assert arr.shape == (640, 640, 3)

    # re-run adds nothing
    s2 = prepare_source(src, "t", shards, conn)
    assert s2["accepted"] == 0 and s2["skipped_existing"] == 1


def test_prepare_jpeg_cleaning_rejects_low_qf(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    img = _gradient()
    Image.fromarray(img).save(src / "bad.jpg", quality=60)
    Image.fromarray(img).save(src / "good.jpg", quality=98)
    conn = M.open_manifest(tmp_path / "m.db")
    stats = prepare_source(src, "j", tmp_path / "shards", conn)
    assert stats["rejected"] == 1, f"expected bad.jpg rejected, got {stats}"
    assert stats["accepted"] == 1, f"expected good.jpg accepted, got {stats}"


def test_prepare_rejects_low_variance_image(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (700, 700), (128, 128, 128)).save(src / "flat.png")
    conn = M.open_manifest(tmp_path / "m.db")
    stats = prepare_source(src, "f", tmp_path / "shards", conn)
    assert stats["patches"] == 0


def test_prepare_phash_dedup_across_dirs_and_formats(tmp_path):
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir(); d2.mkdir()
    img = _gradient()
    Image.fromarray(img).save(d1 / "a.png")
    Image.fromarray(img).save(d2 / "a.tif")  # different bytes, same pixels -> same phash
    shards = tmp_path / "shards"
    conn = M.open_manifest(tmp_path / "m.db")
    s1 = prepare_source(d1, "s1", shards, conn)
    s2 = prepare_source(d2, "s2", shards, conn)
    assert s1["accepted"] == 1
    assert s2["skipped_dup"] == 1 and s2["accepted"] == 0
