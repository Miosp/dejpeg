"""Perceptual comparison matrix: degraded-input vs FBCNN vs our 110k student.

Computes LPIPS-Alex + DISTS (and PSNR for reference) for all three on
Classic5 + LIVE1 at QF {10,20,30,40}. Answers: does the cond-none student beat
FBCNN on the perceptual axis even while trailing on PSNR?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
REPO_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_PKG.parent / "fbcnn-py" / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import cv2

import fbcnn  # noqa: E402  (our port)
from dejpeg_train.eval.dists import DISTS
from dejpeg_train.eval.metrics import psnr
from dejpeg_train.model.student import DeJPEGNetS, build_ctx  # noqa: F401
from dejpeg_train.model.conditioning import quant_table_to_condition  # noqa: F401

import os

DEV = "cuda"
CKPT = Path(os.environ.get(
    "MATRIX_CKPT",
    phase_dir("phase2") / "student_p2_it110k_backup.pt"))
WEIGHTS_DIR = weights_dir()
SETS = {
    "classic5": testsets_dir("Classic5"),
    "live1": testsets_dir("LIVE1_color"),
}
OUT = phase_dir("phase07") / "perceptual_matrix.json"

import lpips  # noqa: E402
from dejpeg_train.paths import phase_dir, weights_dir, testsets_dir
lalex = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
dists_m = DISTS().to(DEV).eval()


def lpips_alex(a_u8, b_u8):
    x = torch.from_numpy(a_u8).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0 * 2 - 1
    y = torch.from_numpy(b_u8).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0 * 2 - 1
    return float(lalex(x, y))


def dists(a_u8, b_u8):
    x = torch.from_numpy(a_u8).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
    y = torch.from_numpy(b_u8).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
    return float(dists_m(x, y))


def main():
    import fbcnn.inference as fb_inf
    import fbcnn.config as fb_cfg

    # student (cond-none, 110k EMA)
    student = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "30"))).to(DEV, memory_format=torch.channels_last)
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    sd = ck["ema"] if "ema" in ck else ck["model"]
    student.load_state_dict(sd)
    student.eval()

    def restore_student(rgb_u8, jpeg_bytes):
        with torch.no_grad():
            t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
            _, _, h, w = t.shape
            t32 = F.pad(t, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))
            o = student(t32.contiguous(memory_format=torch.channels_last),
                        torch.zeros(1, 97, device=DEV))[:, :, :h, :w].clamp(0, 1)
        return (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    results = {}
    for set_name, root in SETS.items():
        imgs = sorted(root.glob("*.bmp")) + sorted(root.glob("*.png"))
        rows = []
        for qf in (10, 20, 30, 40):
            agg = {k: [] for k in (
                "in_lpips", "fb_lpips", "st_lpips", "in_dists", "fb_dists", "st_dists",
                "in_psnr", "fb_psnr", "st_psnr")}
            for p in imgs:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                rgb = bgr[:, :, ::-1].copy()
                ok, enc = cv2.imencode(".jpg", bgr[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), qf])
                dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                dec = dec_bgr[:, :, ::-1].copy()
                jb = enc.tobytes()
                fb_out, _ = fb_inf.run(dec, None, fb_cfg.COLOR_REAL, str(WEIGHTS_DIR))
                st_out = restore_student(dec, jb)
                agg["in_lpips"].append(lpips_alex(dec, rgb))
                agg["in_dists"].append(dists(dec, rgb))
                agg["fb_lpips"].append(lpips_alex(fb_out, rgb))
                agg["fb_dists"].append(dists(fb_out, rgb))
                agg["st_lpips"].append(lpips_alex(st_out, rgb))
                agg["st_dists"].append(dists(st_out, rgb))
                agg["in_psnr"].append(psnr(rgb, dec))
                agg["fb_psnr"].append(psnr(rgb, fb_out))
                agg["st_psnr"].append(psnr(rgb, st_out))
            row = {k: float(np.mean(v)) for k, v in agg.items()}
            rows.append((qf, row))
            print(f"[{set_name} QF{qf}] "
                  f"LPIPS in={row['in_lpips']:.4f} fb={row['fb_lpips']:.4f} st={row['st_lpips']:.4f} | "
                  f"DISTS in={row['in_dists']:.4f} fb={row['fb_dists']:.4f} st={row['st_dists']:.4f} | "
                  f"PSNR in={row['in_psnr']:.2f} fb={row['fb_psnr']:.2f} st={row['st_psnr']:.2f}", flush=True)
        results[set_name] = {str(qf): r for qf, r in rows}

    import json
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n[matrix] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
