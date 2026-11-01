#!/usr/bin/env python
"""Build the E2 (benchmark matrix) and E3 (robustness) eval sets.

E2: clean sources -> JPEG via libjpeg (Pillow) and mozjpeg (jpeglib, trellis)
at QF 5/10/20/30/40/60, 4:2:0. Sets: live1, classic5, bsds500test, kodak,
div2kval, urban100 (+ genuine-gray variants for Y-PSNR comparability).

E3: misaligned double JPEG (q1,shift,q2), JPEG->0.75x resize->JPEG,
near-identity (clean), two-halves (QF10|QF90), flat/gradient/dark synthetic
probes. Sources: live1 + kodak.

Everything is deterministic (fixed seed, shift = index % 7 + 1). Outputs land
under $DEJPEG_DATA_ROOT/eval_sets/{e2,e3}; a manifest JSONL is rewritten per
run from what exists on disk, so re-runs are incremental and idempotent.

Usage:
    python scripts/build_eval_sets.py [--sets live1,kodak] [--qfs 30,40]
        [--encoders libjpeg,mozjpeg] [--limit N] [--skip-e2] [--skip-e3]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dejpeg_train.paths import data_root  # noqa: E402

E2_QFS = [5, 10, 20, 30, 40, 60]
DBL_PAIRS = [(10, 30), (30, 10), (50, 50)]


def _set_sources(root: Path, name: str) -> list[Path]:
    bench = root / "eval_sets" / "raw_bench"
    fbcnn = root / "fbcnn-upstream" / "testsets"
    table = {
        "live1": fbcnn / "LIVE1_color",
        "classic5": fbcnn / "Classic5",
        "bsds500test": bench / "bsds500test" / "BSR" / "BSDS500" / "data" / "images" / "test",
        "kodak": bench / "kodak",
        "div2kval": root / "raw" / "div2k_valid" / "DIV2K_valid_HR",
        "urban100": bench / "Urban100_HR",
        "live1gray": fbcnn / "LIVE1_gray",
    }
    d = table[name]
    if not d.is_dir():
        raise FileNotFoundError(f"source dir for {name} missing: {d}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg", ".tif"})


def encode_libjpeg(arr: np.ndarray, qf: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=qf)  # default 4:2:0 below q=95
    return buf.getvalue()


def encode_mozjpeg(arr: np.ndarray, qf: int, tmp: Path) -> bytes:
    import jpeglib

    jpeglib.version.set("mozjpeg403")
    im = jpeglib.from_spatial(np.ascontiguousarray(arr), in_color_space=jpeglib.JCS_RGB)
    im.write_spatial(str(tmp), qt=qf)
    return tmp.read_bytes()


def decode(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def build_e2(out: Path, sets: list[str], encoders: list[str], qfs: list[int], limit: int | None,
             tmp: Path) -> list[dict]:
    rows = []
    for name in sets:
        srcs = _set_sources(data_root(), name)
        if limit:
            srcs = srcs[:limit]
        for enc in encoders:
            for qf in qfs:
                d = out / name / f"{enc}-qf{qf}"
                d.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(srcs):
                    dst = d / f"{src.stem}.jpg"
                    arr = np.asarray(Image.open(src).convert("RGB"))
                    if not dst.exists():
                        data = encode_libjpeg(arr, qf) if enc == "libjpeg" else encode_mozjpeg(arr, qf, tmp)
                        dst.write_bytes(data)
                    rows.append({"set": name, "encoder": enc, "qf": qf, "tag": "e2",
                                 "clean": str(src), "degraded": str(dst)})
        print(f"e2 {name}: {sum(1 for r in rows if r['set'] == name)} pairs", flush=True)
    return rows


def build_e3(out: Path, sets: list[str], limit: int | None, tmp: Path) -> list[dict]:
    rows = []
    srcs = []
    for name in sets:
        s = _set_sources(data_root(), name)
        if limit:
            s = s[:limit]
        srcs.extend((name, p) for p in s)

    # misaligned double JPEG
    for q1, q2 in DBL_PAIRS:
        d = out / f"dbl-{q1}-{q2}"
        d.mkdir(parents=True, exist_ok=True)
        for i, (name, src) in enumerate(srcs):
            shift = i % 7 + 1
            gt_d, jpg_d = d / f"{src.stem}-s{shift}.png", d / f"{src.stem}-s{shift}.jpg"
            if not jpg_d.exists():
                arr = np.asarray(Image.open(src).convert("RGB"))
                j1 = decode(encode_libjpeg(arr, q1))
                mis = j1[shift:, shift:, :]  # misalign the 8px grid
                gt = arr[shift:, shift:, :]
                Image.fromarray(gt).save(gt_d, lossless=True)
                jpg_d.write_bytes(encode_libjpeg(mis, q2))
            rows.append({"set": f"e3-dbl-{q1}-{q2}", "encoder": "libjpeg", "qf": q2, "tag": "double",
                         "clean": str(gt_d), "degraded": str(jpg_d)})

    # JPEG -> 0.75x resize -> JPEG
    d = out / "resize075"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in srcs:
        gt_d, jpg_d = d / f"{src.stem}.png", d / f"{src.stem}.jpg"
        if not jpg_d.exists():
            arr = np.asarray(Image.open(src).convert("RGB"))
            h, w = arr.shape[:2]
            gt = np.asarray(Image.fromarray(arr).resize((int(w * 0.75), int(h * 0.75)), Image.LANCZOS))
            j1 = Image.fromarray(decode(encode_libjpeg(arr, 30))).resize(
                (int(w * 0.75), int(h * 0.75)), Image.LANCZOS)
            Image.fromarray(gt).save(gt_d, lossless=True)
            jpg_d.write_bytes(encode_libjpeg(np.asarray(j1), 30))
        rows.append({"set": "e3-resize075", "encoder": "libjpeg", "qf": 30, "tag": "resize",
                     "clean": str(gt_d), "degraded": str(jpg_d)})

    # near-identity: model(clean) vs clean (restoration must not move it)
    d = out / "nearidentity"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in srcs:
        dst = d / f"{src.stem}.png"
        if not dst.exists():
            arr = np.asarray(Image.open(src).convert("RGB"))
            Image.fromarray(arr).save(dst, lossless=True)
        rows.append({"set": "e3-nearidentity", "encoder": "none", "qf": 0, "tag": "nearid",
                     "clean": str(src), "degraded": str(dst)})

    # two halves: left QF10, right QF90
    d = out / "twohalves"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in srcs:
        dst = d / f"{src.stem}.png"
        if not dst.exists():
            arr = np.asarray(Image.open(src).convert("RGB"))
            h, w = arr.shape[:2]
            left = decode(encode_libjpeg(arr[:, : w // 2], 10))
            right = decode(encode_libjpeg(arr[:, w // 2:], 90))
            Image.fromarray(np.concatenate([left, right], axis=1)).save(dst, lossless=True)
        rows.append({"set": "e3-twohalves", "encoder": "mixed", "qf": 0, "tag": "twohalves",
                     "clean": str(src), "degraded": str(dst)})

    # synthetic probes
    rng = np.random.default_rng(0)
    H = W = 512
    synth = {
        "flat128": np.full((H, W, 3), 128, np.uint8),
        "flat255": np.full((H, W, 3), 255, np.uint8),
        "gradient_h": np.tile(np.linspace(0, 255, W, dtype=np.uint8)[None, :, None], (H, 1, 3)),
        "gradient_v": np.tile(np.linspace(0, 255, H, dtype=np.uint8)[:, None, None], (1, W, 3)),
        "dark_gradient": np.tile((np.linspace(0, 40, H, dtype=np.uint8)[:, None, None]), (1, W, 3)),
        "checker8": (np.indices((H, W)).sum(0) // 8 % 2 * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
        "noise": rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
    }
    for label, arr in synth.items():
        d = out / f"synth-{label}"
        d.mkdir(parents=True, exist_ok=True)
        gt_d = d / "gt.png"
        if not gt_d.exists():
            Image.fromarray(arr).save(gt_d, lossless=True)
        for qf in (10, 30, 60):
            jpg_d = d / f"qf{qf}.jpg"
            if not jpg_d.exists():
                jpg_d.write_bytes(encode_libjpeg(arr, qf))
            rows.append({"set": f"e3-synth-{label}", "encoder": "libjpeg", "qf": qf, "tag": "synthetic",
                         "clean": str(gt_d), "degraded": str(jpg_d)})
    print(f"e3: {len(rows)} pairs", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="live1,classic5,bsds500test,kodak,div2kval,urban100")
    ap.add_argument("--e3-sets", default="live1,kodak")
    ap.add_argument("--encoders", default="libjpeg,mozjpeg")
    ap.add_argument("--qfs", default=",".join(str(q) for q in E2_QFS))
    ap.add_argument("--limit", type=int, default=None, help="first N images per set (smoke runs)")
    ap.add_argument("--skip-e2", action="store_true")
    ap.add_argument("--skip-e3", action="store_true")
    args = ap.parse_args()

    root = data_root()
    e2_dir, e3_dir = root / "eval_sets" / "e2", root / "eval_sets" / "e3"
    e2_dir.mkdir(parents=True, exist_ok=True)
    e3_dir.mkdir(parents=True, exist_ok=True)
    tmp = e2_dir / ".tmp-enc.jpg"

    rows = []
    if not args.skip_e2:
        rows += build_e2(e2_dir, args.sets.split(","), args.encoders.split(","),
                         [int(q) for q in args.qfs.split(",")], args.limit, tmp)
    if not args.skip_e3:
        rows += build_e3(e3_dir, args.e3_sets.split(","), args.limit, tmp)
    if tmp.exists():
        tmp.unlink()

    # merge with any previous manifest so partial rebuilds keep other rows
    manifest = root / "eval_sets" / "manifest.jsonl"
    seen: set[str] = set()
    merged: list[dict] = []
    for r in rows:
        if r["degraded"] not in seen:
            seen.add(r["degraded"])
            merged.append(r)
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["degraded"] not in seen:
                seen.add(r["degraded"])
                merged.append(r)
    with manifest.open("w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {manifest} ({len(merged)} rows: {len(rows)} from this run)", flush=True)


if __name__ == "__main__":
    main()
