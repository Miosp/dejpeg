"""Chroma-hallucination probe: grayscale Classic5 #1 @QF30 through a ckpt.

Prints invented chroma (mean |A|,|B| deviation from neutral in LAB) and PSNR.
Env: PROBE_CKPT (default phase2c latest).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dejpeg.model.student import DeJPEGNetS  # noqa: E402
from dejpeg.paths import phase_dir, testsets_dir

CKPT = Path(os.environ.get(
    "PROBE_CKPT", phase_dir("phase2c") / "student_p2_latest.pt"))

net = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "30"))).cuda()
ck = torch.load(CKPT, map_location="cuda", weights_only=False)
net.load_state_dict(ck.get("ema", ck["model"]))
net.eval()

gt = cv2.imread(str(testsets_dir("Classic5") / "1.bmp"))
ok, enc = cv2.imencode(".jpg", gt, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
inp = cv2.imdecode(enc, cv2.IMREAD_COLOR)

x = torch.from_numpy(inp).permute(2, 0, 1)[None].float().cuda() / 255.0
with torch.no_grad():
    o = net(x.contiguous(memory_format=torch.channels_last), torch.zeros(1, 97, device="cuda"))
out = (o[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

lab_o = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(float)
chroma = float(np.abs(lab_o[..., 1:] - 128).mean())
psnr_v = 10 * np.log10(255 ** 2 / np.mean((out.astype(float) - gt.astype(float)) ** 2))
print(f"[probe] {CKPT.parent.name}: inventedChroma={chroma:.2f} PSNR={psnr_v:.2f} "
      f"({'OK' if chroma <= 1.7 else 'DRIFTED'})")
