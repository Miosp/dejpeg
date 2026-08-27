"""Corpus preparation (spec §2.3).

Idempotent and resumable (keyed by source content hash -- already-processed images
are skipped). Emits NEW webdataset-style .tar shards with monotonically increasing
keys (existing shards are never modified). Lossless WebP patches (~40% smaller than
PNG). Never JPEG. Global perceptual-hash dedup against the manifest.

Cleaning applies only to JPEG sources: estimate QF from the DQT, reject QF<95;
also reject below a bytes-per-pixel floor. PNG/RAW/TIFF/WebP-native sources skip
cleaning (DF2K is PNG-native). Low-variance 640 patches are rejected.
"""
from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import manifest as M
from .jpegmeta import parse_jpeg

# Standard JPEG Annex-K luminance quantization table at QF50 (natural/raster order).
_ANNEX_K_LUMA_50 = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

_DEFAULT_EXTS = {".png", ".tif", ".tiff", ".webp", ".bmp", ".jpg", ".jpeg"}
_JPEG_EXTS = {".jpg", ".jpeg"}


@dataclass
class PrepareConfig:
    patch_size: int = 640
    overlap: float = 0.25
    min_std: float = 8.0
    clean_min_qf: int = 95
    min_bytes_per_pixel: float = 0.1
    patches_per_shard: int = 256


def estimate_qf(luma_values) -> float:
    """Estimate JPEG quality from a luma quant table (Annex-K inverse mapping)."""
    base = np.asarray(_ANNEX_K_LUMA_50, dtype=np.float64)
    t = np.asarray(luma_values, dtype=np.float64)
    nz = base > 0
    if not np.any(nz):
        return 100.0
    scale = float(np.median((t[nz] * 100.0) / base[nz]))
    if scale >= 100:
        q = 5000.0 / scale
    else:
        q = (200.0 - scale) / 2.0
    return float(np.clip(q, 1.0, 100.0))


def _read_image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        return None


def _clean_jpeg_source(path: Path, cfg: PrepareConfig) -> np.ndarray | None:
    """Return decoded RGB if the JPEG source passes cleaning, else None."""
    size = path.stat().st_size
    try:
        with open(path, "rb") as f:
            meta = parse_jpeg(f.read())
    except Exception:
        return None
    arr = _read_image(path)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    if size / (w * h) < cfg.min_bytes_per_pixel:
        return None
    qt = meta.quant_tables.get(0) or meta.quant_tables.get("0")
    if qt is None:
        # no luma table -> cannot certify cleanliness -> reject
        return None
    qf = estimate_qf(qt.values)
    if qf < cfg.clean_min_qf:
        return None
    return arr


def _crop_patches(arr: np.ndarray, size: int, overlap: float):
    h, w = arr.shape[:2]
    step = max(1, int(size * (1 - overlap)))
    for y in range(0, max(1, h - size + 1), step):
        for x in range(0, max(1, w - size + 1), step):
            patch = arr[y : y + size, x : x + size]
            if patch.shape[0] < size or patch.shape[1] < size:
                continue
            yield patch


class _ShardWriter:
    def __init__(self, shards_dir: Path, source_name: str, start_index: int, patches_per_shard: int):
        self.dir = shards_dir
        self.source = source_name
        self.per_shard = patches_per_shard
        self.global_key = start_index
        self.in_shard = 0
        self.tar = None
        self.current_name: str | None = None
        self.dir.mkdir(parents=True, exist_ok=True)

    def _open_new(self):
        if self.tar is not None:
            self.tar.close()
        shard_idx = self.global_key // self.per_shard
        self.current_name = f"{self.source}-{shard_idx:06d}.tar"
        self.tar = tarfile.open(self.dir / self.current_name, "w")
        self.in_shard = 0

    def add(self, patch: np.ndarray, source_content_hash: str) -> str:
        if self.tar is None or self.in_shard >= self.per_shard:
            self._open_new()
        key = f"{self.global_key:08d}"
        buf = io.BytesIO()
        Image.fromarray(patch).save(buf, format="WEBP", lossless=True)
        data = buf.getvalue()
        ti = tarfile.TarInfo(f"{key}.webp")
        ti.size = len(data)
        self.tar.addfile(ti, io.BytesIO(data))
        meta = {"source": self.source, "source_hash": source_content_hash, "shard": self.current_name}
        mb = json.dumps(meta).encode()
        mti = tarfile.TarInfo(f"{key}.json")
        mti.size = len(mb)
        self.tar.addfile(mti, io.BytesIO(mb))
        self.global_key += 1
        self.in_shard += 1
        return self.current_name

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def prepare_source(
    source_dir: str | Path,
    source_name: str,
    shards_dir: str | Path,
    conn,
    config: PrepareConfig | None = None,
    extensions: set[str] | None = None,
) -> dict:
    cfg = config or PrepareConfig()
    exts = extensions or _DEFAULT_EXTS
    source_dir = Path(source_dir)
    shards_dir = Path(shards_dir)

    existing = sorted(shards_dir.glob(f"{source_name}-*.tar"))
    start_index = 0
    for s in existing:
        # resume from highest shard index
        try:
            idx = int(s.stem.rsplit("-", 1)[-1])
            start_index = max(start_index, (idx + 1) * cfg.patches_per_shard)
        except ValueError:
            pass

    writer = _ShardWriter(shards_dir, source_name, start_index, cfg.patches_per_shard)
    stats = {"scanned": 0, "accepted": 0, "skipped_existing": 0, "skipped_dup": 0, "rejected": 0, "patches": 0}

    paths = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in exts and p.is_file())
    for path in paths:
        stats["scanned"] += 1
        ch = M.content_hash_file(path)
        if M.has_content(conn, ch):
            stats["skipped_existing"] += 1
            continue
        if path.suffix.lower() in _JPEG_EXTS:
            arr = _clean_jpeg_source(path, cfg)
            if arr is None:
                stats["rejected"] += 1
                continue
        else:
            arr = _read_image(path)
            if arr is None:
                stats["rejected"] += 1
                continue
        ph = M.perceptual_hash(arr)
        if M.find_dup_by_phash(conn, ph) is not None:
            stats["skipped_dup"] += 1
            continue
        shard_used = None
        patch_count = 0
        for patch in _crop_patches(arr, cfg.patch_size, cfg.overlap):
            if patch.std() < cfg.min_std:
                continue
            shard_used = writer.add(patch, ch)
            patch_count += 1
        h, w = arr.shape[:2]
        M.add_entry(
            conn,
            content_hash=ch,
            perceptual_hash=ph,
            shard_id=shard_used or "",
            source_corpus=source_name,
            patch_count=patch_count,
            width=w,
            height=h,
        )
        stats["accepted"] += 1
        stats["patches"] += patch_count

    writer.close()
    return stats
