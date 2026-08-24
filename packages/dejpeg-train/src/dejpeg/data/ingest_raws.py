"""Raw ingest preprocessor (Task 0.4 front-end for user raws).

Develops from-camera raws (DNG/CR2/NEF/ARW/RAF/RW2/ORF/PEF/SRW...) into clean
lossless RGB, ready to enter the prepare.py corpus pipeline as a PNG/lossless-
native source. This is the "optimize the raws" step: many GB of raws become a
managed, deduped, resolution-capped lossless store.

Defaults (sensible for matching the sRGB-JPEG deployment distribution):
  * develop to 8-bit sRGB, camera white balance, no auto-brightness
  * cap longest side at 4096 (Lanczos) -- saves space, keeps patch diversity
  * lossless WebP output (16-bit falls back to PNG)
  * raws left untouched (read-only); output is a separate clean corpus dir
  * idempotent + resumable keyed by raw content hash; global perceptual-hash dedup

``develop_raw`` is a module-level function so tests can monkeypatch it without a
real raw file on disk.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

RAW_EXTS = frozenset(
    {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".pef", ".srw", ".raw", ".iiq"}
)


def develop_raw(path: Path, *, sixteen_bit: bool = False) -> np.ndarray:
    """LibRaw-develop a raw file to an RGB array (uint8 by default)."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_bps=16 if sixteen_bit else 8,
            output_color=rawpy.ColorSpace.sRGB,
            use_camera_wb=True,
            no_auto_bright=True,
        )
    return rgb


def downscale_to(arr: np.ndarray, max_long_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_long_side:
        return arr
    scale = max_long_side / longest
    new = (max(1, round(h * scale)), max(1, round(w * scale)))
    img = Image.fromarray(arr)
    return np.array(img.resize((new[1], new[0]), Image.LANCZOS))


def _content_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def perceptual_hash(arr: np.ndarray) -> str:
    """16-byte dHash (64-bit) of an 8x8 grayscale thumbnail."""
    gray = np.asarray(Image.fromarray(arr).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return "{:016x}".format(int(np.packbits(bits).tobytes().hex(), 16))


def _open_manifest(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    # IF NOT EXISTS: the shared corpus DB may already hold the `corpus` table
    # (from prepare.py) but not the `raws` table -- create idempotently.
    con.execute(
        "CREATE TABLE IF NOT EXISTS raws (content_hash TEXT PRIMARY KEY, phash TEXT, "
        "source TEXT, out_path TEXT)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_phash ON raws(phash)")
    con.commit()
    return con


def _save_lossless(arr: np.ndarray, out_path: Path, sixteen_bit: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if sixteen_bit:
        img.save(str(out_path), format="PNG")  # lossless 16-bit
    else:
        img.save(str(out_path), format="WEBP", lossless=True)


def ingest(
    source_dir,
    output_dir,
    *,
    manifest_db=None,
    max_long_side: int = 4096,
    sixteen_bit: bool = False,
    extensions=RAW_EXTS,
):
    """Walk ``source_dir`` for raws, develop + dedup into ``output_dir``.

    Returns a stats dict: scanned, developed, skipped_existing, skipped_dup,
    failed. Idempotent: re-running adds nothing new.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    manifest_db = Path(manifest_db) if manifest_db else output_dir / "raw_manifest.sqlite"
    out_ext = ".png" if sixteen_bit else ".webp"

    con = _open_manifest(manifest_db)
    stats = {"scanned": 0, "developed": 0, "skipped_existing": 0, "skipped_dup": 0, "failed": 0}

    for raw_path in sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in extensions):
        stats["scanned"] += 1
        try:
            chash = _content_hash(raw_path)
            row = con.execute("SELECT out_path FROM raws WHERE content_hash=?", (chash,)).fetchone()
            if row:
                stats["skipped_existing"] += 1
                continue

            arr = develop_raw(raw_path, sixteen_bit=sixteen_bit)
            arr = downscale_to(arr, max_long_side)
            ph = perceptual_hash(arr)
            dup = con.execute("SELECT 1 FROM raws WHERE phash=? LIMIT 1", (ph,)).fetchone()
            if dup:
                stats["skipped_dup"] += 1
                continue

            out_path = output_dir / f"{chash[:16]}{out_ext}"
            _save_lossless(arr, out_path, sixteen_bit)
            con.execute(
                "INSERT INTO raws VALUES (?,?,?,?)",
                (chash, ph, str(raw_path), str(out_path)),
            )
            con.commit()
            stats["developed"] += 1
        except Exception:
            stats["failed"] += 1
    con.close()
    return stats
