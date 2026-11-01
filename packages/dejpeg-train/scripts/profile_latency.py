#!/usr/bin/env python
"""T7 latency profile for the deployed artifact + torch student.

Measures restore latency at deployment-relevant tile sizes, both runtimes:
  --onnx PATH  : deployed fp16 artifact, CUDA EP and CPU EP
  --weights    : train-package student (fp32, CUDA) for comparison
Sizes: 256, 512, 1024 (square) + 512x768 photo aspect. Reports mean/ms
over --iters runs after --warmup, and per-megapixel normalize. The web app
runs the ONNX graph single-threaded on the client device; CPU-EP numbers
here are an upper-bound proxy for high-end desktops, not mobile truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--weights")
    ap.add_argument("--sliding", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cond-mode", default="prompt", choices=["prompt", "scalar", "none"])
    ap.add_argument("--c0", type=int, default=30)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sizes = [(256, 256), (512, 512), (768, 512), (1024, 1024)]
    results: dict[str, dict] = {}

    if args.onnx:
        import onnxruntime as ort

        ort.preload_dlls()  # pip nvidia-* libs (cudnn etc.) for the CUDA EP
        for prov_name in [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                          if p in ort.get_available_providers()]:
            sess = ort.InferenceSession(args.onnx, providers=[prov_name])
            name = sess.get_inputs()[0].name
            key = f"onnx:{prov_name[:-17]}"  # CUDA / CPU
            results[key] = {}
            for h, w in sizes:
                x = np.random.rand(1, 3, h, w).astype(np.float32)
                results[key][f"{h}x{w}"] = bench(
                    lambda: sess.run(None, {name: x}), args.warmup, args.iters)
            print(key, {k: f"{v:.1f}ms" for k, v in results[key].items()}, flush=True)

    if args.weights:
        import torch

        from dejpeg_train.model.student import DeJPEGNetS, build_ctx
        from dejpeg_train.model.conditioning import quant_table_to_condition

        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model = DeJPEGNetS(cond_mode=args.cond_mode, c0=args.c0).eval()
        model.load_state_dict(sd)
        if args.sliding:
            from dejpeg_train.model.blocks import swap_lsca_sliding

            swap_lsca_sliding(model)  # creates fresh CPU modules; move after
        model = model.to(args.device, memory_format=torch.channels_last)
        cond = quant_table_to_condition([0] * 64, 0.0).to(args.device)
        ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=args.device))
        key = f"torch:{args.device}{'-sliding' if args.sliding else ''}"
        results[key] = {}
        with torch.no_grad():
            for h, w in sizes:
                x = torch.rand(1, 3, h, w, device=args.device).contiguous(
                    memory_format=torch.channels_last)
                results[key][f"{h}x{w}"] = bench(
                    lambda: model(x, ctx), args.warmup, args.iters)
        print(key, {k: f"{v:.1f}ms" for k, v in results[key].items()}, flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
