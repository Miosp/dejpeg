"""Export verification: torch/ORT parity and tile-invariance on the exported graph.

The parity test (torch vs ORT on CPU EP, max-abs < 1e-3) is the Phase-0 export
gate. The real-browser no-CPU-fallback assertion lives in Phase 0.5 / 0.11-browser
via Playwright headless Chrome WebGPU; here we provide the ORT-side harness.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def ort_session(path: str, providers=None):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def ort_parity(onnx_path: str, model, sample_inputs: Sequence, *, atol: float = 1e-3):
    """Run torch and ORT (CPU EP) on the same inputs. Return (max_abs_diff, per_output)."""
    model = model.eval()
    with torch.no_grad():
        out = model(*sample_inputs)
    torch_outs = out if isinstance(out, (tuple, list)) else (out,)
    sess = ort_session(onnx_path)
    feeds = {inp.name: _np(x) for inp, x in zip(sess.get_inputs(), sample_inputs)}
    ort_outs = sess.run(None, feeds)
    diffs = [float(np.max(np.abs(_np(t) - o))) for t, o in zip(torch_outs, ort_outs)]
    return max(diffs), diffs


def tile_invariance_onnx(
    onnx_path: str,
    image: np.ndarray,
    ctx: np.ndarray,
    *,
    tile: int = 128,
    overlap: int = 64,
) -> float:
    """Full-image vs halo-tiled inference through the exported graph; max-abs.

    Tile origins are multiples of (tile-overlap) and the LSCA grid is 32, so an
    LSCA student graph must reconstruct near-exactly. ctx is a per-image constant
    reused for every tile.
    """
    sess = ort_session(onnx_path)
    in_tile, in_ctx = sess.get_inputs()[0].name, sess.get_inputs()[1].name

    def run(img):
        return sess.run(None, {in_tile: img.astype(np.float32), in_ctx: ctx.astype(np.float32)})[0]

    full = run(image)
    h, w = image.shape[-2:]
    stride = tile - overlap
    starts_h = list(range(0, h - tile + 1, stride))
    starts_w = list(range(0, w - tile + 1, stride))
    if starts_h[-1] != h - tile:
        starts_h.append(h - tile)
    if starts_w[-1] != w - tile:
        starts_w.append(w - tile)

    acc = np.zeros_like(image)
    wsum = np.zeros((1, 1, h, w), dtype=np.float32)
    for sh in starts_h:
        for sw in starts_w:
            view = image[:, :, sh : sh + tile, sw : sw + tile]
            out = run(view)
            acc[:, :, sh : sh + tile, sw : sw + tile] += out
            wsum[:, :, sh : sh + tile, sw : sw + tile] += 1.0
    tiled = acc / wsum
    return float(np.max(np.abs(full - tiled)))
