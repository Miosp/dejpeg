"""Ingest user raws (Sony ARW) -> developed lossless RGB on ext4.

Reads ONCE from /mnt/c (drvfs) during this one-time ingest; writes developed
lossless WebP to ext4 so the training dataloader never touches /mnt/c.
Defaults: sRGB 8-bit, cap longest side 4096, lossless WebP, raws read-only/untouched.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dejpeg.data.ingest_raws import ingest
from dejpeg.paths import raw_dir, manifest_path

# Default to the ext4 copy (fast reads); the /mnt/c drvfs path is 10-50x slower.
SRC = os.environ.get(
    "RAWS_SRC", str(raw_dir() / "user_raws_src")
)
OUT = str(raw_dir() / "user_raws")
MANIFEST = str(manifest_path())


def main() -> None:
    print(f"[ingest] src={SRC}", flush=True)
    print(f"[ingest] out={OUT}", flush=True)
    print(f"[ingest] manifest={MANIFEST}", flush=True)
    print(f"[ingest] defaults: sRGB 8-bit, cap longest side 4096, lossless WebP", flush=True)
    t0 = time.time()
    stats = ingest(
        SRC,
        OUT,
        manifest_db=MANIFEST,
        max_long_side=4096,
        sixteen_bit=False,
        extensions={".arw"},
    )
    dt = time.time() - t0
    print(f"[ingest] DONE in {dt:.0f}s ({dt/60:.1f}min)", flush=True)
    print(f"[ingest] stats={stats}", flush=True)


if __name__ == "__main__":
    main()
