"""Global corpus manifest (spec §2.3).

SQLite store: content_hash -> perceptual_hash -> shard_id -> source_corpus ->
patch_count. Dedup is GLOBAL (not per-run): a perceptual hash close to any existing
entry is rejected no matter which source directory it came from.

This is the authoritative dedup store. The raw preprocessor (ingest_raws.py) keeps
its own lighter manifest for the develop step; unifying them is minor future debt.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np
from PIL import Image

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus (
    content_hash   TEXT PRIMARY KEY,
    perceptual_hash TEXT NOT NULL,
    shard_id       TEXT,
    source_corpus  TEXT NOT NULL,
    patch_count    INTEGER DEFAULT 0,
    width          INTEGER,
    height         INTEGER,
    added_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_phash  ON corpus(perceptual_hash);
CREATE INDEX IF NOT EXISTS idx_source ON corpus(source_corpus);
"""

_POPCOUNT = bytearray(256)
for _i in range(256):
    _POPCOUNT[_i] = bin(_i).count("1")


def open_manifest(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    return conn


def perceptual_hash(arr: np.ndarray, hash_size: int = 8) -> str:
    """dHash: 8x8 gradient bits -> 16 hex chars. Deterministic, cross-run stable."""
    img = Image.fromarray(arr).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    px = np.asarray(img, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    bits = int(np.packbits(diff.flatten()).tobytes().hex(), 16)
    return f"{bits:0{(hash_size * hash_size) // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    va, vb = int(a, 16), int(b, 16)
    x = va ^ vb
    return _POPCOUNT[x & 0xFF] + _POPCOUNT[(x >> 8) & 0xFF] + _POPCOUNT[(x >> 16) & 0xFF] + _POPCOUNT[(x >> 24) & 0xFF]


def has_content(conn: sqlite3.Connection, content_hash: str) -> bool:
    row = conn.execute("SELECT 1 FROM corpus WHERE content_hash=?", (content_hash,)).fetchone()
    return row is not None


def find_dup_by_phash(conn: sqlite3.Connection, phash: str, threshold: int = 3) -> str | None:
    """Return the content_hash of a near-duplicate (hamming <= threshold), else None."""
    rows = conn.execute("SELECT content_hash, perceptual_hash FROM corpus").fetchall()
    for ch, ph in rows:
        if hamming_hex(phash, ph) <= threshold:
            return ch
    return None


def add_entry(
    conn: sqlite3.Connection,
    *,
    content_hash: str,
    perceptual_hash: str,
    shard_id: str,
    source_corpus: str,
    patch_count: int,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    cur = conn.execute(
        "INSERT OR IGNORE INTO corpus VALUES (?,?,?,?,?,?,?,?)",
        (content_hash, perceptual_hash, shard_id, source_corpus, patch_count, width, height, time.time()),
    )
    conn.commit()
    return cur.rowcount > 0


def source_summary(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT source_corpus, COUNT(*), SUM(patch_count) FROM corpus GROUP BY source_corpus"
    ).fetchall()


def content_hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
