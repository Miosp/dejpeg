"""Near-identity gate on a checkpoint: clean DIV2K-valid in -> out must stay ~identity."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dejpeg_train.model.student import DeJPEGNetS  # noqa: E402
from dejpeg_train.paths import data_root, phase_dir, raw_dir

CKPT = Path(os.environ.get("GATE_CKPT", phase_dir("phase2") / "student_p2_latest.pt"))
DIV2K = data_root() / "DIV2K_valid_HHR"  # fallback below if missing
for cand in (raw_dir() / "div2k_valid/DIV2K_valid_HR",
             DIV2K,
             data_root() / "DIV2K/DIV2K_valid_HR"):
    if cand.is_dir():
        DIV2K = cand
        break

m = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "30"))).cuda()
ck = torch.load(CKPT, map_location="cuda", weights_only=False)
m.load_state_dict(ck.get("ema", ck["model"]))
m.eval()

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)

vals = []
imgs = sorted(DIV2K.glob("*.png"))[:10]
print(f"[gate] {len(imgs)} imgs from {DIV2K}")
for p in imgs:
    bgr = cv2.imread(str(p))
    c = 512
    h, w = bgr.shape[:2]
    y0, x0 = (h - c) // 2, (w - c) // 2
    crop = bgr[y0:y0 + c, x0:x0 + c]
    x = torch.from_numpy(crop).permute(2, 0, 1)[None].float().cuda() / 255.0
    with torch.no_grad():
        o = m(x.contiguous(memory_format=torch.channels_last), torch.zeros(1, 97, device="cuda"))
    out = (o[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    v = psnr(crop, out)
    vals.append(v)
    print(f"  {p.stem}: {v:.2f} dB")
print(f"[gate] worst={min(vals):.2f} mean={float(np.mean(vals)):.2f} "
      f"{'PASS' if min(vals) >= 45 else 'FAIL'} (gate >=45dB)")
