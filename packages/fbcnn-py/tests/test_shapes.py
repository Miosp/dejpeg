"""Shape correctness for every variant — no weights needed."""

from __future__ import annotations

import pytest
import torch

from fbcnn import VARIANTS, build_model


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.id for v in VARIANTS])
def test_shape_with_qf_input(variant):
    net = build_model(variant)
    net.eval()
    x = torch.randn(1, variant.in_nc, 64, 64)
    qf = torch.tensor([[0.6]])
    with torch.no_grad():
        out_e, out_qf = net(x, qf)
    assert out_e.shape == (1, variant.out_nc, 64, 64)
    assert out_qf.shape == (1, 1)


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.id for v in VARIANTS])
def test_shape_without_qf_input(variant):
    """Auto-predict path: model must still return (E, QF)."""
    net = build_model(variant)
    net.eval()
    x = torch.randn(1, variant.in_nc, 64, 64)
    with torch.no_grad():
        out_e, out_qf = net(x)
    assert out_e.shape == (1, variant.out_nc, 64, 64)
    assert out_qf.shape == (1, 1)


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.id for v in VARIANTS])
def test_shape_non_square(variant):
    """Non-square input — ONNX dynamic axes must handle this."""
    net = build_model(variant)
    net.eval()
    x = torch.randn(1, variant.in_nc, 48, 72)
    qf = torch.tensor([[0.4]])
    with torch.no_grad():
        out_e, _ = net(x, qf)
    assert out_e.shape == (1, variant.out_nc, 48, 72)
