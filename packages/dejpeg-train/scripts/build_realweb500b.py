"""Fill realweb500 to TARGET using picsum.photos (resize+re-encode pipeline).

Appends to the existing manifest left by build_realweb500.py.
"""
import hashlib
import io
import json
import random
import time
import urllib.request
from pathlib import Path

from PIL import Image
from dejpeg_train.paths import eval_sets_dir

OUT = eval_sets_dir() / "realweb500"
MANIFEST = OUT / "manifest.jsonl"
TARGET = 500
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) dejpeg-research/1.0"}
DIMS = [(800, 600), (1024, 768), (1200, 800), (1280, 854), (640, 480),
        (1600, 900), (900, 1200), (1440, 960)]

have = {json.loads(l)["sha1"] for l in MANIFEST.read_text().splitlines()} if MANIFEST.exists() else set()
n_have = len(list(OUT.glob("rw_*.jpg")))
print(f"[start] have {n_have} files", flush=True)

rng = random.Random(11)
idx = n_have
got = 0
ids = list(range(0, 1084))
rng.shuffle(ids)
backoff = 1.0

for pid in ids:
    if n_have + got >= TARGET:
        break
    w, h = rng.choice(DIMS)
    try:
        req = urllib.request.Request(f"https://picsum.photos/id/{pid}/{w}/{h}.jpg", headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            blob = r.read()
        backoff = 1.0
        if len(blob) < 8 * 1024 or len(blob) > 2_500_000:
            continue
        sha = hashlib.sha1(blob).hexdigest()
        if sha in have:
            continue
        im = Image.open(io.BytesIO(blob))
        im.verify()
        im = Image.open(io.BytesIO(blob))
        if im.format != "JPEG" or min(im.size) < 256:
            continue
        idx += 1
        path = OUT / f"rw_{idx:03d}.jpg"
        path.write_bytes(blob)
        with MANIFEST.open("a") as f:
            f.write(json.dumps({"file": path.name, "source": f"picsum:{pid}",
                                "bytes": len(blob), "width": im.size[0],
                                "height": im.size[1], "sha1": sha}) + "\n")
        have.add(sha)
        got += 1
        if got % 50 == 0:
            print(f"[got] total {n_have + got}", flush=True)
        time.sleep(0.2)
    except Exception as e:
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)

total = n_have + got
sizes = sorted(p.stat().st_size for p in OUT.glob("rw_*.jpg"))
print(f"[done] total {total} imgs, median {sizes[len(sizes)//2]//1024}KB -> {OUT}", flush=True)
