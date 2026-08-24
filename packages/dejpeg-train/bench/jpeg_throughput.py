"""JPEG encode/decode throughput benchmark (Task 0.6).

Measures CPU paths (PIL vs simplejpeg) on the common case (Annex-K table,
baseline, standard subsampling). simplejpeg wraps libjpeg-turbo directly and is
the fast-path candidate; PIL supports progressive + custom tables but is slower.

The nvjpeg GPU path needs a CUDA JPEG binding (nvidia.dali or a custom extension)
and is deferred until Phase 2, when the actual training throughput is known and
the loader >= 2x GPU-consumption gate can be evaluated. If CPU cannot sustain
that ratio at batch 8, GPU JPEG becomes mandatory.

Usage: uv run python bench/jpeg_throughput.py
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

import simplejpeg


def make_patch(seed: int = 0, size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, size, dtype=np.float32)
    img = (base[None, :] + base[:, None]) / 2
    img = np.stack([img, img, img], axis=-1)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255)
    return np.ascontiguousarray(img.astype(np.uint8))


def bench_pil(patch: np.ndarray, qf: int, subsampling: int, n: int) -> float:
    img = Image.fromarray(patch)
    t0 = time.perf_counter()
    for _ in range(n):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=qf, subsampling=subsampling, optimize=False)
        Image.open(buf).convert("RGB").load()
    return (time.perf_counter() - t0) / n


def bench_simplejpeg(patch: np.ndarray, qf: int, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        enc = simplejpeg.encode_jpeg(patch, qf, colorspace="rgb")
        simplejpeg.decode_jpeg(enc, colorspace="rgb")
    return (time.perf_counter() - t0) / n


def main():
    patch = make_patch(size=512)
    qf = 50
    n = 200
    print(f"JPEG encode+decode, 512x512 RGB, QF {qf}, {n} iters\n")
    # simplejpeg has no per-call subsampling in 1.9 (RGB default ~420); compare
    # against PIL 420 (same family) for an apples-to-apples fast-path read.
    t_pil = bench_pil(patch, qf, 2, n)
    t_sj = bench_simplejpeg(patch, qf, n)
    print(f"  PIL(420)     : {t_pil*1e3:7.2f} ms/img  ({1/t_pil:7.1f} img/s)")
    print(f"  simplejpeg   : {t_sj*1e3:7.2f} ms/img  ({1/t_sj:7.1f} img/s)")
    print(f"  speedup      : {t_pil/t_sj:5.2f}x\n")
    print(
        "Phase-2 gate: loader throughput (img/s) must be >= 2x the training\n"
        "consumption rate (batch 8 + accum x2 = 16 samples/step). Measure again\n"
        "once the training loop reports samples/s; switch to nvjpeg if CPU falls short."
    )


if __name__ == "__main__":
    main()
