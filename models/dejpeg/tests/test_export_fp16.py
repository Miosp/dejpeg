"""D5 regression: the FP16 ONNX graph must survive saturated inputs.

Report 2026-10-13 §5/D5: LayerNorm2d's variance computation squared deviations
of up to ~255 levels per channel -> ~65k, overflowing IEEE half (65504) and
producing inf/NaN on saturated blocks (11 pixels on a real QF30 crop, parity
down to 37.1 dB). The fix pre-scales the LN input by 1/16 (exact identity in
fp32, squared deviations capped at ~256). This test is the CI gate: FP16 vs
FP32 ONNX parity must stay >= 50 dB on worst-case inputs.
"""
import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime", reason="export extra not installed")

from dejpeg.export import export_onnx
from dejpeg.model import DeJPEGNet, LayerNorm2d

SIZES = (256,)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _saturated_inputs(size: int) -> dict[str, np.ndarray]:
    """Worst cases for LN variance: max channel spread per pixel."""
    n = size // 8
    rng = np.random.default_rng(0)
    # per-8x8-block random channel-diverse constants (R/G/B independent)
    blocks = rng.random((n, n, 3))
    x = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1).transpose(2, 0, 1)[None]
    out = {"blockwise-rgb": x.astype(np.float32)}
    # hard saturation: 0/1 checkerboard, channels offset
    yy, xx = np.mgrid[0:size, 0:size]
    cb = ((xx // 8 + yy // 8) % 2).astype(np.float32)
    out["checkerboard"] = np.stack([cb, 1.0 - cb, np.full_like(cb, 0.5)])[None]
    out["white"] = np.ones((1, 3, size, size), np.float32)
    out["black-white-columns"] = np.stack(
        [np.zeros_like(cb), np.ones_like(cb), (yy // 8 % 2).astype(np.float32)]
    )[None]
    return out


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    torch.manual_seed(0)
    model = DeJPEGNet(c0=16)  # tiny: LN depths/widths match the real topology
    d = tmp_path_factory.mktemp("fp16")
    p32 = export_onnx(model, d / "m32.onnx", size=SIZES[0], fp16=False)
    p16 = export_onnx(model, d / "m16.onnx", size=SIZES[0], fp16=True)
    return (
        onnxruntime.InferenceSession(str(p32), providers=["CPUExecutionProvider"]),
        onnxruntime.InferenceSession(str(p16), providers=["CPUExecutionProvider"]),
    )


def test_ln_prescale_is_identity_in_fp64():
    ln = LayerNorm2d(64).double()
    x = torch.randn(2, 64, 16, 16, dtype=torch.float64) * 300.0
    u = x.mean(1, keepdim=True)
    d = (x - u).pow(2).mean(1, keepdim=True)
    ref = (x - u) / torch.sqrt(d + 1e-6)
    ref = ref * ln.weight.view(1, -1, 1, 1) + ln.bias.view(1, -1, 1, 1)
    assert torch.allclose(ln(x), ref, atol=1e-8, rtol=1e-10)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", ["blockwise-rgb", "checkerboard", "white", "black-white-columns"])
def test_fp16_parity_on_saturated_inputs(sessions, size, name):
    s32, s16 = sessions
    x = _saturated_inputs(size)[name]
    y32 = s32.run(None, {"input": x})[0]
    y16 = s16.run(None, {"input": x})[0]
    assert np.isfinite(y16).all(), "FP16 graph produced non-finite values (D5 overflow)"
    psnr = _psnr(np.clip(y16, 0, 1), np.clip(y32, 0, 1))
    assert psnr >= 50.0, f"{name}: FP16 parity {psnr:.1f} dB < 50 dB (D5 regression)"
