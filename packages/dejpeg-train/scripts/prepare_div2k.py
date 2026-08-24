"""Prepare DIV2K into webdataset shards (Phase 0.5 corpus).

DIV2K is PNG-native -> cleaning is skipped; just crop 640px patches + dedup.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dejpeg.data.manifest import open_manifest
from dejpeg.data.prepare import PrepareConfig, prepare_source
from dejpeg.paths import manifest_path, raw_dir, shards_dir

DIV2K = raw_dir() / "div2k" / "DIV2K_train_HR"
SHARDS = shards_dir()
MANIFEST = manifest_path()


def main() -> None:
    if not DIV2K.is_dir():
        raise SystemExit(f"DIV2K not found at {DIV2K}")
    conn = open_manifest(MANIFEST)
    stats = prepare_source(DIV2K, "div2k", SHARDS, conn, PrepareConfig())
    print("DIV2K prepare:", stats)
    n_shards = len(list(SHARDS.glob("div2k-*.tar")))
    print(f"shards: {n_shards}")


if __name__ == "__main__":
    main()
