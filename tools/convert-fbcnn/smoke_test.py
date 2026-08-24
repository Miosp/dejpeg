"""Compare PyTorch and ONNX Runtime outputs on identical random input.

Tolerance: 1e-2 absolute max (FP16 dynamic quantize introduces some drift;
FP32 would match to 1e-5 but we ship FP16).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

SMOKE_TOLERANCE = 1e-2


def smoke_test(
    onnx_path: Path,
    torch_net: torch.nn.Module,
    variant_in_nc: int,
) -> None:
    torch_net.eval()
    dummy_image = torch.randn(1, variant_in_nc, 64, 64)
    dummy_qf = torch.tensor([[0.6]])

    with torch.no_grad():
        out_torch_e, out_torch_qf = torch_net(dummy_image, dummy_qf)
    out_torch_e_np = out_torch_e[0].numpy()
    out_torch_qf_np = out_torch_qf[0].numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_ort = sess.run(None, {
        "input":    dummy_image.numpy(),
        "qf_input": dummy_qf.numpy(),
    })
    out_ort_e_np = out_ort[0][0]
    out_ort_qf_np = out_ort[1][0]

    assert out_torch_e_np.shape == out_ort_e_np.shape, (
        f"shape mismatch: torch {out_torch_e_np.shape} vs ort {out_ort_e_np.shape}"
    )

    max_abs_e = float(np.abs(out_torch_e_np - out_ort_e_np).max())
    assert max_abs_e < SMOKE_TOLERANCE, f"image diverged: max_abs={max_abs_e} > {SMOKE_TOLERANCE}"

    max_abs_qf = float(np.abs(out_torch_qf_np - out_ort_qf_np).max())
    assert max_abs_qf < SMOKE_TOLERANCE, f"qf diverged: max_abs={max_abs_qf} > {SMOKE_TOLERANCE}"

    print(f"smoke ok: image max_abs={max_abs_e:.2e}, qf max_abs={max_abs_qf:.2e}")
