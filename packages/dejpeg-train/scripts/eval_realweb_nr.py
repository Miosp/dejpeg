"""No-GT IQA comparison on the real-web corpus: input vs FBCNN vs our student.

Usage:
  python scripts/eval_realweb_nr.py --variants input --device cpu   # anytime
  python scripts/eval_realweb_nr.py --variants all   --device cuda   # post-training

MUSIQ/CLIPIQA higher=better; NIQE lower=better.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "fbcnn-py" / "src"))

import os
from dejpeg_train.paths import eval_sets_dir, weights_dir, phase_dir

CORPUS = eval_sets_dir() / "realweb500"
OUT_JSON = CORPUS / "nr_scores.json"
WEIGHTS_DIR = weights_dir()
CKPT = Path(os.environ.get(
    "NR_CKPT", phase_dir("phase2b") / "student_p2_latest.pt"))


def load_student(device):
    import torch
    from dejpeg_train.model.student import DeJPEGNetS

    m = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "30"))).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    sd = ck.get("ema", ck["model"])
    m.load_state_dict(sd)
    m.eval()
    return m


def tiled_restore(model, img_u8, device, tile=256, overlap=64):
    import cv2
    import torch

    x = torch.from_numpy(img_u8[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    _, _, H, W = x.shape
    if max(H, W) <= int(tile * 1.4):
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            o = model(xp.contiguous(memory_format=torch.channels_last),
                      torch.zeros(1, 97, device=device))[:, :, :H, :W]
        return (o[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[:, :, ::-1]

    step = tile - overlap
    vy = list(range(0, max(H - tile, 0) + 1, step))
    vx = list(range(0, max(W - tile, 0) + 1, step))
    if vy[-1] != H - tile:
        vy.append(H - tile)
    if vx[-1] != W - tile:
        vx.append(W - tile)

    wy = torch.hann_window(tile, periodic=False).to(device)
    wx = torch.hann_window(tile, periodic=False).to(device)
    win = (wy.unsqueeze(1) @ wx.unsqueeze(0)).clamp_min(1e-4)[None, None]

    acc = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    with torch.no_grad():
        for y0 in vy:
            for x0 in vx:
                t = x[:, :, y0:y0 + tile, x0:x0 + tile]
                o = model(t.contiguous(memory_format=torch.channels_last),
                          torch.zeros(1, 97, device=device))
                acc[:, :, y0:y0 + tile, x0:x0 + tile] += o * win
                wsum[:, :, y0:y0 + tile, x0:x0 + tile] += win
    out = (acc / wsum).clamp(0, 1)
    return (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[:, :, ::-1]


class Metrics:
    def __init__(self, device):
        import torch
        import pyiqa
        self.torch = torch
        self.m = {n: pyiqa.create_metric(n, device=device) for n in ("musiq", "clipiqa", "niqe")}

    def __call__(self, bgr_u8):
        x = self.torch.from_numpy(bgr_u8[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255.0
        x = x.to(next(self.m["musiq"].parameters()).device)
        return {n: float(fn(x)) for n, fn in self.m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="input", choices=["input", "student", "fbcnn", "all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    variants = ["input", "student", "fbcnn"] if args.variants == "all" else [args.variants]
    files = sorted(CORPUS.glob("rw_*.jpg"))
    if args.limit:
        files = files[:args.limit]

    import cv2
    model = load_student(args.device) if "student" in variants else None
    if "fbcnn" in variants:
        import fbcnn.config as fb_cfg
        import fbcnn.inference as fb_inf
    met = Metrics(args.device)

    results = {v: {} for v in variants}
    t0 = time.time()
    for i, p in enumerate(files, 1):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        if max(bgr.shape[:2]) > 2400:
            s = 2400 / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        for v in variants:
            img = bgr if v == "input" else (
                tiled_restore(model, bgr, args.device) if v == "student" else
                fb_inf.run(bgr, None, fb_cfg.COLOR_REAL, str(WEIGHTS_DIR))[0])
            try:
                sc = met(img)
            except Exception as e:
                print(f"[{i}/{len(files)}] {v} metric fail: {e}", flush=True)
                continue
            for k, val in sc.items():
                results[v].setdefault(k, []).append(val)
        if i % 25 == 0 or i == len(files):
            el = time.time() - t0
            print(f"[{i}/{len(files)}] {el:.0f}s eta {el / i * (len(files) - i):.0f}s", flush=True)

    summary = {}
    for v in variants:
        summary[v] = {k: {"mean": float(np.mean(a)), "median": float(np.median(a)), "n": len(a)}
                      for k, a in results[v].items()}
    OUT_JSON.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
