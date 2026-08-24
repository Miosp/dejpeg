"""Conditioning transform: JPEG luma quant table -> 65-D model input vector.

Layout: [log-normalized 64 quant values, validity flag].
  log(x + 1) / log(256) maps the [1, 255] quant range into [0, 1].
  validity = 1 when a real table was parsed, 0 when dropped (non-JPEG path) or
  the table could not be recovered.

This module is the SINGLE SOURCE OF TRUTH for the python side. The TypeScript
twin (inference-core/src/codec/conditioning.ts) must be byte-identical; the
Phase-0.5 cross-language parity test enforces that.
"""
from __future__ import annotations

import math

import torch

LOG256 = math.log(256.0)


def quant_table_to_condition(table_64, validity: float = 1.0) -> torch.Tensor:
    """64 quant values (natural order, 1-255) + validity -> (65,) float32 tensor."""
    t = torch.as_tensor(table_64, dtype=torch.float32).flatten()
    assert t.numel() == 64, f"expected 64 quant values, got {t.numel()}"
    log = torch.log(t + 1.0) / LOG256
    return torch.cat([log, torch.tensor([validity], dtype=torch.float32)])


def record_to_condition(record, dropout_p: float = 0.0, rng=None) -> torch.Tensor:
    """Build a (65,) condition from a DegradationRecord's luma quant table.

    With probability ``dropout_p`` the table is zeroed and validity cleared (the
    non-JPEG / unknown-table path). Phase 0.5 passes dropout_p=0 (GT table always).
    """
    qt = record.quant_tables.get(0)
    if qt is not None:
        values = qt["values"] if isinstance(qt, dict) else getattr(qt, "values", None)
    else:
        qt = record.quant_tables.get("0")
        values = (qt["values"] if isinstance(qt, dict) else getattr(qt, "values", None)) if qt is not None else None
    if values is None:
        return quant_table_to_condition(torch.zeros(64), validity=0.0)
    drop = dropout_p > 0 and rng is not None and rng.random() < dropout_p
    if drop:
        return quant_table_to_condition(torch.zeros(64), validity=0.0)
    return quant_table_to_condition(values, validity=1.0)
