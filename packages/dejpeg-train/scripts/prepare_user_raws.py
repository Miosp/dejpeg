"""Shard only the user_raws corpus (idempotent; new webps only)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dejpeg_train.data.manifest import open_manifest
from dejpeg_train.data.prepare import PrepareConfig, prepare_source
from dejpeg_train.paths import raw_dir, shards_dir, manifest_path

RAW = raw_dir() / "user_raws"
SHARDS = shards_dir()
MANIFEST = manifest_path()


def main() -> None:
    t0 = time.time()
    with open_manifest(MANIFEST) as man:
        stats = prepare_source(
            RAW, "user_raws", SHARDS, man,
            PrepareConfig(),
            extensions={".webp"},
        )
    print(f"[prepare] user_raws -> {stats}", flush=True)
    print(f"[prepare] wall_s={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
