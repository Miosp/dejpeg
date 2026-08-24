"""Deployment export: reparameterize, trace to ONNX, simplify, cast FP16.

Order matters: fuse the parallel depthwise branches first so the graph carries a
single conv per site, then export with constant folding, then (optionally) halve
the weights. FP16 halves file size and roughly doubles WebGPU throughput; keep
I/O in FP32 so callers never think about dtype.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .model import DeJPEGNet


def export_onnx(model: DeJPEGNet, path: str | Path, size: int = 256,
                dynamic: bool = False, fp16: bool = True) -> Path:
    device = next(model.parameters()).device
    fused = model.fuse()
    sample = torch.rand(1, 3, size, size, device=device)
    torch.onnx.export(
        fused, sample, str(path), opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"}, "output": {0: "N", 2: "H", 3: "W"}}
        if dynamic else None,
        do_constant_folding=True,
    )
    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(str(path))
        simplified, check = simplify(onnx_model)
        if check:
            onnx.save(simplified, str(path))
    except ImportError:
        pass

    if fp16:
        import onnx
        from onnxconverter_common import float16

        onnx_model = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
        onnx.save(onnx_model, str(path))
    return Path(path)
