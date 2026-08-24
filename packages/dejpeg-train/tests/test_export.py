"""Phase 0 Task 0.11 -- export stub tests.

Gate: ONNX export parity vs torch on CPU EP, max-abs < 1e-3.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from dejpeg.export.onnx import export_onnx, fuse_for_export
from dejpeg.export.verify import ort_parity, tile_invariance_onnx
from dejpeg.model.blocks import RepDWConv3x3
from dejpeg.model.student import DeJPEGNetS


def _small_student():
    return DeJPEGNetS(c0=8, enc_blocks=(1, 1, 1), mid_blocks=1, dec_blocks=(1, 1, 1))


def test_fuse_for_export_replaces_all_repdwconv():
    train_model = _small_student()
    fused = fuse_for_export(train_model)
    assert any(isinstance(m, RepDWConv3x3) for m in train_model.modules()), "train model has RepDWConv"
    assert not any(isinstance(m, RepDWConv3x3) for m in fused.modules()), "fused model still has RepDWConv"


def test_export_fp32_ort_parity(tmp_path):
    torch.manual_seed(0)
    model = _small_student().eval()
    tile = torch.randn(1, 3, 32, 32)
    ctx = torch.randn(1, 97)
    path = str(tmp_path / "student_fp32.onnx")
    export_onnx(model, (tile, ctx), path, fp16=False, simplify=True)
    onnx.checker.check_model(onnx.load(path))  # simplify output must be valid
    max_diff, _ = ort_parity(path, model, (tile, ctx), atol=1e-3)
    assert max_diff < 1e-3, f"ORT parity {max_diff:.2e} >= 1e-3"


def test_fp16_export_is_valid_onnx(tmp_path):
    torch.manual_seed(0)
    model = _small_student().eval()
    tile = torch.randn(1, 3, 32, 32)
    ctx = torch.randn(1, 97)
    path = str(tmp_path / "student_fp16.onnx")
    export_onnx(model, (tile, ctx), path, fp16=True, simplify=True)
    onnx.checker.check_model(onnx.load(path))


def test_fused_model_matches_unfused_output(tmp_path):
    torch.manual_seed(0)
    model = _small_student().eval()
    fused = fuse_for_export(model)
    tile = torch.randn(1, 3, 32, 32)
    ctx = torch.randn(1, 97)
    with torch.no_grad():
        a = model(tile, ctx)
        b = fused(tile, ctx)
    assert torch.max((a - b).abs()).item() < 1e-5, "fusion changed outputs"


def test_exported_graph_tile_invariance(tmp_path):
    torch.manual_seed(1)
    model = _small_student().eval()
    np.random.seed(1)
    img = np.random.randn(1, 3, 64, 64).astype(np.float32)
    ctx = np.random.randn(1, 97).astype(np.float32)
    path = str(tmp_path / "student_dyn.onnx")
    sample = (torch.randn(1, 3, 32, 32), torch.randn(1, 97))
    export_onnx(model, sample, path, simplify=False, dynamic=True)
    err = tile_invariance_onnx(path, img, ctx, tile=32, overlap=16)
    assert err < 5e-2, f"ONNX tile-invariance err {err:.2e}"
