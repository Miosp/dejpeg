"""Phase 0.7.4 -- SSIM loss arm: L1 + LPIPS + W_SSIM*(1-SSIM) vs L1 + LPIPS at 10k.

Run as a PAIR (W_SSIM=0 baseline, then W_SSIM=1) at c0=30 cond-none (the 0.7.2
winner) so the loss question is settled at matched budget/iters. Judged on
LIVE1 QF10-40 mean PSNR (+ gain over input). Env: W_SSIM, TAG.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from dejpeg.data.jpegmeta import parse_jpeg
from dejpeg.data.loader import make_dataloader
from dejpeg.eval.metrics import psnr, psnr_b
from dejpeg.loss.perceptual import PerceptualLoss
from dejpeg.model.conditioning import quant_table_to_condition
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import shards_dir, phase_dir, testsets_dir
from dejpeg.train.schedule import (
    EMA, bf16_autocast, clip_grad_norm, cosine_lr, prepare_model_for_training, set_seed,
)

W_SSIM = float(os.environ.get("W_SSIM", "1.0"))
TAG = os.environ.get("TAG", f"ssim{W_SSIM:g}")
ITERS = 10_000
SHARDS = [str(shards_dir() / "div2k-*.tar"),
          str(shards_dir() / "flickr2k-*.tar")]
OUT = phase_dir("phase07")
OUT.mkdir(parents=True, exist_ok=True)
LIVE1 = testsets_dir("LIVE1_color")
DEV = "cuda"
_cond = quant_table_to_condition([0] * 64, validity=0.0).to(DEV)


def _gauss_window(k=11, sigma=1.5, c=3, dev=DEV):
    x = torch.arange(k, dtype=torch.float32, device=dev) - (k - 1) / 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    w = (g[:, None] * g[None, :]); w = w / w.sum()
    return w.view(1, 1, k, k).repeat(c, 1, 1, 1)


_WIN = _gauss_window()
_K1, _K2 = 0.01, 0.03


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable SSIM on [0,1] NCHW; returns mean 1-SSIM."""
    mu_x = F.conv2d(x, _WIN, groups=x.shape[1])
    mu_y = F.conv2d(y, _WIN, groups=y.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x = F.conv2d(x * x, _WIN, groups=x.shape[1]) - mu_x2
    sigma_y = F.conv2d(y * y, _WIN, groups=y.shape[1]) - mu_y2
    sigma_xy = F.conv2d(x * y, _WIN, groups=x.shape[1]) - mu_xy
    c1, c2 = (_K1 * 1.0) ** 2, (_K2 * 1.0) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
    return 1.0 - ssim_map.mean()


@torch.no_grad()
def eval_live1(model):
    import cv2
    ps, pins = [], []
    for p in sorted(LIVE1.glob("*.bmp")):
        bgr = cv2.imread(str(p))
        rgb = bgr[:, :, ::-1].copy()
        cond = quant_table_to_condition(parse_jpeg(cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 20])[1].tobytes()).quant_tables[0].values, 1.0).to(DEV)
        dec = cv2.imdecode(cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 20])[1], cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=DEV))
        t = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
        _, _, h, w = t.shape
        t32 = F.pad(t, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))
        o = model(t32.contiguous(memory_format=torch.channels_last), ctx)[:, :, :h, :w].clamp(0, 1)
        o = (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        ps.append(psnr(rgb, o)); pins.append(psnr(rgb, dec))
    return float(np.mean(ps)), float(np.mean(pins))


def main():
    set_seed(0)
    gen = DeJPEGNetS(cond_mode="none").to(DEV, memory_format=torch.channels_last).train()
    gen_fwd = prepare_model_for_training(gen)
    opt = torch.optim.AdamW(gen.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.9), fused=True)
    ema = EMA(gen, decay=0.999)
    percept = PerceptualLoss(net="vgg", crop=128).to(DEV).eval()
    lr_at = cosine_lr(1e-3, ITERS, warmup=500)
    loader = make_dataloader(SHARDS, batch_size=8, num_workers=6, patch=256, seed=42)
    it = iter(loader)
    print(f"[ssim:{TAG}] w_ssim={W_SSIM} iters={ITERS}", flush=True)
    t0 = time.time()
    for step in range(ITERS):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        jpeg, target, c65 = next(it)
        jpeg = jpeg.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        target = target.to(DEV, non_blocking=True)
        ctx = build_ctx(c65, torch.zeros(len(c65), 32)).to(DEV, non_blocking=True)
        with bf16_autocast(True, device_type="cuda"):
            pred = gen_fwd(jpeg, ctx)
            pred_f = pred.float()
            loss = (pred_f - target).abs().mean() + percept(pred_f, target)
            if W_SSIM > 0:
                loss = loss + W_SSIM * ssim_loss(pred_f, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(gen.parameters(), 1.0)
        opt.step()
        ema.update(gen)
        if step % 2000 == 0 or step == ITERS - 1:
            print(f"[ssim:{TAG}] it={step:5d} loss={loss.item():.4f} {1000*(time.time()-t0)/(step+1):.0f}ms/it", flush=True)
    with ema.swap(gen):
        torch.save({"model": gen.state_dict()}, OUT / f"student_ssim_{TAG}.pt")
        gen.eval()
        p20, pin20 = eval_live1(gen)
        print(f"[ssim:{TAG}] LIVE1 QF20: PSNR={p20:.2f} (in {pin20:.2f}, gain {p20-pin20:+.2f})", flush=True)
    print(f"[ssim:{TAG}] DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
