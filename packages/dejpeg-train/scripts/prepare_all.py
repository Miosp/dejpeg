"""Shard the three remaining corpus sources into webdataset shards.

Runs serially (one shared manifest.sqlite connection) to avoid sqlite write
contention. Idempotent: already-sharded sources skip via content-hash.
  - DIV2K re-prep at threshold 3 (re-evaluates the 198 false-dups)
  - LIU4K-v2 (2000 PNGs, internet-sourced but PNG-native -> skip cleaning)
  - user_raws (883 developed lossless WebP)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dejpeg_train.data.manifest import open_manifest
from dejpeg_train.data.prepare import PrepareConfig, prepare_source
from dejpeg_train.paths import raw_dir, shards_dir, manifest_path

RAW = raw_dir()
SHARDS = shards_dir()
MANIFEST = manifest_path()

SOURCES = [
    ("div2k", RAW / "div2k/DIV2K_train_HR", {".png"}),
    ("liu4k_v2", RAW / "liu4k_v2", {".png"}),
    ("user_raws", RAW / "user_raws", {".webp"}),
]


def main() -> None:
    conn = open_manifest(MANIFEST)
    cfg = PrepareConfig()
    results = {}
    for name, src, exts in SOURCES:
        if not src.exists():
            print(f"[skip] {name}: {src} missing", flush=True)
            continue
        t0 = time.time()
        print(f"[prepare] {name} <- {src}", flush=True)
        stats = prepare_source(src, name, SHARDS, conn, cfg, extensions=exts)
        stats["wall_s"] = round(time.time() - t0, 1)
        results[name] = stats
        print(f"[prepare] {name} -> {stats}", flush=True)
    conn.close()
    print("[prepare] ALL DONE", flush=True)
    print("RESULTS=" + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
