"""ONNX export pipeline (spec §3.3 / Phase 3.3).

Order: reparameterize (fuse RepDWConv -> single DW3x3) -> export -> simplify ->
(constant fold) -> optional FP16 cast. Phase 0 ships the FP32 path + parity gate;
FP16 deploy hardening is Phase 3.2.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..model.blocks import RepDWConv3x3
from ..model.reparam import fuse_rep_dwconv


def fuse_for_export(model: nn.Module) -> nn.Module:
    """Return an eval copy with every RepDWConv3x3 fused to a single DW3x3.

    The original (training) model is untouched.
    """
    fused = copy.deepcopy(model).eval()
    _fuse_in_place(fused)
    return fused


def _fuse_in_place(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, RepDWConv3x3):
            setattr(module, name, fuse_rep_dwconv(child))
        else:
            _fuse_in_place(child)


def export_onnx(
    model: nn.Module,
    sample_inputs,
    path: str,
    *,
    opset: int = 17,
    simplify: bool = True,
    fp16: bool = False,
    dynamic: bool = False,
    input_names=("tile", "ctx"),
    output_names=("output",),
) -> str:
    model = model.eval()
    input_names = list(input_names)
    output_names = list(output_names)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            input_names[0]: {0: "N", 2: "H", 3: "W"},
            output_names[0]: {0: "N", 2: "H", 3: "W"},
        }
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_inputs,
            path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    if simplify:
        _simplify(path)
    if fp16:
        _to_fp16(path)
    return path


def _simplify(path: str) -> None:
    import onnx
    from onnxsim import simplify

    model = onnx.load(path)
    simplified, check = simplify(model)
    if check:
        onnx.save(simplified, path)


def _to_fp16(path: str) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, path)
