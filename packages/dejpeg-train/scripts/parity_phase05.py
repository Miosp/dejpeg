#!/usr/bin/env python3
"""Phase 0.5.6 golden-image parity: torch (EMA, fp32) vs ORT-Web (WebGPU, fp16)
on a byte-identical deterministic input. Reports max-abs + verdict."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO + "/src")
from dejpeg_train.paths import phase_dir, work_root

OUT = str(phase_dir("phase05"))
BIN = os.environ.get("PARITY_BIN", str(work_root() / "tmp" / "parity_out.bin"))
THRESH = 0.05  # fp16 weights/activations + WebGPU numerics vs torch fp32

from dejpeg_train.model.student import DeJPEGNetS
from dejpeg_train.train.schedule import EMA

# Deterministic tile (matches parity.html exactly).
W = 256
tile = np.zeros((1, 3, W, W), dtype=np.float32)
for c in range(3):
    for h in range(W):
        for w in range(W):
            tile[0, c, h, w] = ((c * 17 + h * 3 + w * 7) % 256) / 255.0
ctx = np.full((1, 97), 0.5, dtype=np.float32)

ckpt = torch.load(f"{OUT}/student.pt", map_location="cpu", weights_only=False)
model = DeJPEGNetS().eval()
ema = EMA(model)
ema.load_state_dict(ckpt["ema"])

with ema.swap(model), torch.no_grad():
    out_torch = model(torch.from_numpy(tile), torch.from_numpy(ctx)).numpy()

out_browser = np.fromfile(BIN, dtype=np.float32).reshape(1, 3, W, W)

diff = np.abs(out_torch - out_browser)
mx = float(diff.max())
mean = float(diff.mean())
# residual magnitude for context
res = float(np.abs(out_torch - tile).max())
verdict = "PASS" if mx < THRESH else "FAIL"
print(f"[parity] torch-vs-ORTWeb max-abs={mx:.5f} mean={mean:.6f} (thresh {THRESH})")
print(f"[parity] residual max-abs (torch out - input)={res:.5f}")
print(f"[parity] verdict: {verdict}")
