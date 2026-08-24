"""Benchmark a checkpoint against ground-truth image suites.

For every image and quality factor: re-encode the pristine file as JPEG,
restore it, score against the pristine original. Reports PSNR always, and
LPIPS/DISTS when pyiqa is installed (``pip install .[eval]``).

Expects suites laid out as one folder of images each, e.g.:
    data/benchmarks/classic5/*.bmp
    data/benchmarks/live1/*.png

Example:
    python scripts/evaluate.py --weights weights/dejpeg-c40-fp16.pt \
        --suite data/benchmarks/classic5 --qf 10 20 30 40
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dejpeg.infer import load_model, restore_array

EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def jpeg_roundtrip(pil: Image.Image, qf: int) -> Image.Image:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=qf, subsampling=2)  # 4:2:0, like most web JPEGs
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def to_tensor(pil: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None)
    p.add_argument("--suite", required=True, action="append",
                   help="folder of ground-truth images; repeatable")
    p.add_argument("--qf", type=int, nargs="+", default=[10, 20, 30, 40])
    p.add_argument("--sharpness", type=float, default=0.10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    model = load_model(args.weights, args.device)
    try:
        import pyiqa

        metrics = {"lpips": pyiqa.create_metric("lpips", device=args.device),
                   "dists": pyiqa.create_metric("dists", device=args.device)}
    except ImportError:
        metrics = {}
        print("(pyiqa not installed -- reporting PSNR only; pip install .[eval])")

    results = {}
    for suite in args.suite:
        paths = sorted(f for f in Path(suite).iterdir() if f.suffix.lower() in EXTENSIONS)
        for qf in args.qf:
            rows = []
            for path in paths:
                gt = Image.open(path).convert("RGB")
                degraded = jpeg_roundtrip(gt, qf)
                out = restore_array(to_tensor(degraded, args.device), model, args.sharpness)
                row = {"input_psnr": psnr(np.asarray(gt), np.asarray(degraded)),
                       "psnr": psnr(np.asarray(gt),
                                    (out.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))}
                x = out.unsqueeze(0)
                gt_t = to_tensor(gt, args.device).unsqueeze(0)
                for name, metric in metrics.items():
                    row[name] = float(metric(x, gt_t))
                rows.append(row)

            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            results[f"{Path(suite).name}@QF{qf}"] = agg
            pretty = " ".join(f"{k}={v:.4f}" if k != "psnr" else f"{k}={v:.2f}" for k, v in agg.items())
            print(f"[{Path(suite).name} QF{qf}] n={len(paths)} {pretty}", flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
