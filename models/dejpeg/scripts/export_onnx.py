"""Export a checkpoint to FP16 ONNX.

    python scripts/export_onnx.py --weights weights/dejpeg-c40-fp16.pt --out dejpeg.onnx --dynamic
"""
from __future__ import annotations

import argparse

import torch

from dejpeg.export import export_onnx
from dejpeg.infer import load_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=None)
    p.add_argument("--out", default="dejpeg.onnx")
    p.add_argument("--size", type=int, default=256, help="trace size; irrelevant when --dynamic")
    p.add_argument("--dynamic", action="store_true", help="export dynamic H/W axes")
    p.add_argument("--no-fp16", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    path = export_onnx(model, args.out, size=args.size, dynamic=args.dynamic, fp16=not args.no_fp16)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
