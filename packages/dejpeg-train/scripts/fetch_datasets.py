"""Fetch DF2K + LIU4K-v2 to ext4. Resilient: per-source try/except, continues on failure.

Logs to $DEJPEG_DATA_ROOT/fetch.log. Idempotent: skips files already downloaded.
Run via: uv run --with gdown python scripts/fetch_datasets.py
"""
from __future__ import annotations

import os
import pathlib
import sys

from dejpeg.paths import data_root
import tarfile
import time
import urllib.request
import zipfile

DATA = data_root()
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "div2k": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "flickr2k": "http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
}
LIU4K_FOLDERS = {
    "LIU4K_v2_train": "1FtVQtY2t_ecuy_gzJqZ-CatqrJBAdq_d",
    "LIU4K_v2_val": "1OCSXbWAlZ_im9oVIocOlr8tTjKFlOYM-",
}


def log(msg: str) -> None:
    print(f"[fetch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_http(url: str, dest: pathlib.Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  exists, skip: {dest.name} ({dest.stat().st_size / 1e6:.0f}MB)")
        return True
    log(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            t0 = time.time()
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got % (50 * 1024 * 1024) < 1024 * 1024:
                    pct = (100 * got / total) if total else 0
                    rate = got / 1e6 / max(time.time() - t0, 1e-6)
                    log(f"    {dest.name}: {got / 1e6:.0f}/{total / 1e6:.0f}MB ({pct:.0f}%) {rate:.1f}MB/s")
        tmp.replace(dest)
        log(f"  done {dest.name}: {dest.stat().st_size / 1e6:.0f}MB")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"  FAILED {dest.name}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def extract(path: pathlib.Path, into: pathlib.Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(into)
    else:
        with tarfile.open(path) as t:
            t.extractall(into)
    log(f"  extracted {path.name} -> {into}")


def main() -> int:
    failures = []

    log("=== DF2K ===")
    for name, url in SOURCES.items():
        ext = ".zip" if url.endswith(".zip") else ".tar"
        dest = RAW / f"{name}{ext}"
        ok = download_http(url, dest)
        if ok:
            try:
                extract(dest, RAW / name)
            except Exception as e:  # noqa: BLE001
                log(f"  extract FAILED {name}: {e}")
                failures.append(f"extract:{name}")
        else:
            failures.append(f"download:{name}")

    log("=== LIU4K-v2 (Google Drive) ===")
    try:
        import gdown

        for name, fid in LIU4K_FOLDERS.items():
            out = RAW / name
            if out.exists() and any(out.iterdir()):
                log(f"  exists, skip: {name}")
                continue
            log(f"  gdown folder {name} ({fid})")
            try:
                gdown.download_folder(id=fid, output=str(out), quiet=False, use_cookies=False, remaining_ok=True)
                log(f"  done {name}")
            except Exception as e:  # noqa: BLE001
                log(f"  FAILED {name}: {e}")
                failures.append(f"gdown:{name}")
    except ImportError:
        log("  gdown not available (run with --with gdown)")
        failures.append("gdown:import")

    log("=== SUMMARY ===")
    for p in sorted(RAW.glob("**/*")):
        if p.is_file():
            log(f"  {p.relative_to(RAW)} ({p.stat().st_size / 1e6:.1f}MB)")
    if failures:
        log(f"FAILURES: {failures}")
        return 1
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
