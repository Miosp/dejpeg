"""Process Flickr2K + LIU-4K-v2 into corpus shards.

Extracts PNGs from the Windows-side downloads to ext4 raw dirs (one slow /mnt/c
read), then prepares lossless-WebP 640 shards ext4->ext4 (fast). Idempotent via
content-hash + global phash dedup. NO cleanup here -- run clean_corpora.sh after
verifying shard counts.
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dejpeg_train.paths import manifest_path, raw_dir, shards_dir  # noqa: E402

RAW = raw_dir()
SHARDS = shards_dir()
MANIFEST = manifest_path()
WIN_DL = Path(os.environ.get("DEJPEG_DOWNLOADS", str(Path.home() / "Downloads")))
FLICKR_ZIP = WIN_DL / "Flickr2K.zip"
LIU_DIR = WIN_DL / "LIU-4K"

from dejpeg_train.data import manifest as M  # noqa: E402
from dejpeg_train.data.prepare import PrepareConfig, prepare_source  # noqa: E402


def extract_pngs(zip_path: Path, out_dir: Path, log_prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".png"):
                continue
            target = out_dir / Path(info.filename).name
            if target.exists():
                continue
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            n += 1
            if n % 200 == 0:
                print(f"  {log_prefix}: {n} PNGs...", flush=True)
    return n


def main() -> int:
    t0 = time.time()

    # 1. Flickr2K (skip the .txt aux files)
    print(f"[extract] Flickr2K -> {RAW / 'flickr2k'}", flush=True)
    n = extract_pngs(FLICKR_ZIP, RAW / "flickr2k", "flickr2k")
    print(f"[extract] Flickr2K: {n} PNGs ({time.time() - t0:.0f}s)", flush=True)

    # 2. LIU-4K (8 zips, PNGs only)
    t1 = time.time()
    total = 0
    for z in sorted(LIU_DIR.glob("*.zip")):
        out = RAW / "liu4k_v2" / z.stem
        try:
            k = extract_pngs(z, out, z.stem)
            total += k
            print(f"[extract] {z.name}: {k} PNGs", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[extract] {z.name} FAILED: {e}", flush=True)
    print(f"[extract] LIU-4K total: {total} PNGs ({time.time() - t1:.0f}s)", flush=True)

    # 3. Prepare both -> shards
    conn = M.open_manifest(MANIFEST)
    cfg = PrepareConfig()
    for name in ("flickr2k", "liu4k_v2"):
        src = RAW / name
        if not src.exists():
            print(f"[prepare] {name}: source dir missing, skipping", flush=True)
            continue
        t2 = time.time()
        stats = prepare_source(src, name, SHARDS, conn, cfg, {".png"})
        print(f"[prepare] {name}: {stats} ({time.time() - t2:.0f}s)", flush=True)

    # 4. Summary
    nshards = {
        "div2k": len(list(SHARDS.glob("div2k-*.tar"))),
        "flickr2k": len(list(SHARDS.glob("flickr2k-*.tar"))),
        "liu4k_v2": len(list(SHARDS.glob("liu4k_v2-*.tar"))),
    }
    print(f"[done] shards={nshards} total={time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
