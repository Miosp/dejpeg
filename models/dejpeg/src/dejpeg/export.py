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
    # torch>=2.14 (dynamo exporter) writes initializers to an external
    # `<path>.data` sidecar and cannot downconvert to opset 17; we emit 18 and
    # re-inline below so the artifact stays a single self-contained file.
    torch.onnx.export(
        fused, sample, str(path), opset_version=18,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"}, "output": {0: "N", 2: "H", 3: "W"}}
        if dynamic else None,
        do_constant_folding=True,
    )
    sidecar = Path(str(path) + ".data")
    if sidecar.exists():
        # merge the external tensor file back into the proto
        import onnx

        onnx.save_model(onnx.load(str(path), load_external_data=True), str(path))
        sidecar.unlink()
    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(str(path))
        simplified, check = simplify(onnx_model)
        if check:
            onnx.save(simplified, str(path))
    except ImportError:
        pass

    _floor_average_pool(path)

    if fp16:
        import onnx
        from onnxconverter_common import float16

        onnx_model = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
        onnx.save(onnx_model, str(path))
    return Path(path)


def _floor_average_pool(path: str | Path) -> None:
    """Rewrite AveragePool ceil_mode 1 -> 0 in the exported graph.

    onnxruntime-web's symbolic shape inference (required for the WebGPU
    execution provider) cannot express ceil() in AveragePool output dims and
    fails session creation outright. Torch semantics keep ceil_mode=True
    because spatial extents below the 32px window must still produce one
    window; the deployment contract instead pads inputs to multiples of 32
    (see ModelDef alignment notes), where floor == ceil. For a directly-fed
    non-aligned size the exported graph drops the trailing partial window
    instead of averaging it; the Resize back to (h, w) hides the difference.
    """
    import onnx
    from onnx import AttributeProto

    model = onnx.load(str(path))
    changed = 0
    for node in model.graph.node:
        if node.op_type != "AveragePool":
            continue
        for attr in node.attribute:
            if attr.name == "ceil_mode" and attr.i == 1:
                attr.i = 0
                changed += 1
    if changed:
        onnx.save(model, str(path))
