"""Build a real-web JPEG corpus (~500 imgs) from Wikimedia Commons thumbnails.

Thumbnails are produced by the site's own resize+re-encode pipeline, which is
exactly the "resized and recompressed for the web" degradation our model must
handle. Bytes are stored verbatim - no local re-encoding.
"""
import hashlib
import io
import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image
from dejpeg_train.paths import eval_sets_dir

OUT = eval_sets_dir() / "realweb500"
MANIFEST = OUT / "manifest.jsonl"
TARGET = 500
UA = {"User-Agent": "dejpeg-research-corpus/1.0 (local research use)"}
API = "https://commons.wikimedia.org/w/api.php"
WIDTHS = [640, 800, 1024, 1280, 1600]

OUT.mkdir(parents=True, exist_ok=True)


def api(params):
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    rng = random.Random(7)
    done, seen_sha, titles, afrom = [], set(), [], None
    while len(titles) < TARGET * 3:
        q = {"action": "query", "list": "categorymembers", "cmtitle": "Category:Quality images",
             "cmtype": "file", "cmlimit": "500"}
        if afrom:
            q["cmcontinue"] = afrom
        try:
            data = api(q)
        except Exception as e:
            print(f"[api] {e}", flush=True)
            time.sleep(5)
            continue
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont = data.get("continue")
        if not cont:
            break
        afrom = cont["cmcontinue"]

    print(f"[pool] {len(titles)} candidate titles", flush=True)
    rng.shuffle(titles)

    idx = 0
    for title in titles:
        if len(done) >= TARGET:
            break
        try:
            w = rng.choice(WIDTHS)
            info = api({"action": "query", "titles": title, "prop": "imageinfo",
                        "iiprop": "url|size|mime|sha1", "iiurlwidth": str(w)})
            page = next(iter(info["query"]["pages"].values()))
            ii = page["imageinfo"][0]
            if ii.get("mime") not in ("image/jpeg",):
                continue
            url = ii.get("thumburl") or ii["url"]
            blob = fetch(url)
            if len(blob) < 8 * 1024 or len(blob) > 2_500_000:
                continue
            sha = hashlib.sha1(blob).hexdigest()
            if sha in seen_sha:
                continue
            im = Image.open(io.BytesIO(blob))
            im.verify()
            im = Image.open(io.BytesIO(blob))
            if min(im.size) < 256 or im.format != "JPEG":
                continue
            seen_sha.add(sha)
            idx += 1
            path = OUT / f"rw_{idx:03d}.jpg"
            path.write_bytes(blob)
            with MANIFEST.open("a") as f:
                f.write(json.dumps({"file": path.name, "source": title, "thumb_width": w,
                                    "bytes": len(blob), "width": im.size[0],
                                    "height": im.size[1], "sha1": sha}) + "\n")
            done.append(path.name)
            if len(done) % 50 == 0:
                print(f"[got] {len(done)}", flush=True)
            time.sleep(0.15)
        except Exception as e:
            print(f"[skip] {title}: {type(e).__name__}: {e}", flush=True)
            time.sleep(1)

    sizes = [os.path.getsize(OUT / n) for n in done]
    print(f"[done] {len(done)} imgs, median {sorted(sizes)[len(sizes)//2]//1024}KB -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
