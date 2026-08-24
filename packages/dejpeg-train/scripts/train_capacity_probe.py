"""Phase 0.7.3 -- capacity probe: C0 in {40, 48} at 30k iters, winner's recipe
(cond_mode=none per the 0.7.2 verdict, L1+LPIPS, grad-checkpoint so both fit VRAM).
Measures the dB-per-MB slope vs C0=30 (30.81 Classic5 / 28.73 LIVE1 @30k).
Env overrides: C0, ITERS, TAG.
"""
import os
import random
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

C0 = int(os.environ.get("C0", "40"))
ITERS = int(os.environ.get("ITERS", "30000"))
TAG = os.environ.get("TAG", f"c0{C0}")
SHARDS = [str(shards_dir() / "div2k-*.tar"),
          str(shards_dir() / "flickr2k-*.tar")]
OUT = phase_dir("phase07")
OUT.mkdir(parents=True, exist_ok=True)
CLASSIC5 = testsets_dir("LIVE1_color")  # placeholder, fixed below
CLASSIC5 = testsets_dir("Classic5")
LIVE1 = testsets_dir("LIVE1_color")
BATCH, WORKERS, PATCH = 8, 6, 256
LR, WARMUP, GRAD_CLIP = 1e-3, 500, 1.0
DEV = "cuda"
_cond = quant_table_to_condition([0] * 64, validity=0.0).to(DEV)


@torch.no_grad()
def eval_set(model, imgs, qf):
    ps, pbs, pins = [], [], []
    import cv2
    for p in imgs:
        bgr = cv2.imread(str(p))
        rgb = bgr[:, :, ::-1].copy()
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        cond = quant_table_to_condition(parse_jpeg(enc.tobytes()).quant_tables[0].values, 1.0).to(DEV)
        ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=DEV))
        t = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
        _, _, h, w = t.shape
        t32 = F.pad(t, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))
        o = model(t32.contiguous(memory_format=torch.channels_last), ctx)[:, :, :h, :w].clamp(0, 1)
        o = (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        ps.append(psnr(rgb, o)); pbs.append(psnr_b(rgb, o)); pins.append(psnr(rgb, dec))
    return float(np.mean(ps)), float(np.mean(pbs)), float(np.mean(pins))


def main():
    set_seed(0)
    gen = DeJPEGNetS(cond_mode="none", c0=C0, grad_checkpoint=True).to(DEV, memory_format=torch.channels_last).train()
    gen_fwd = prepare_model_for_training(gen)
    n_params = sum(p.numel() for p in gen.parameters())
    opt = torch.optim.AdamW(gen.parameters(), lr=LR, weight_decay=1e-3, betas=(0.9, 0.9), fused=True)
    ema = EMA(gen, decay=0.999)
    percept = PerceptualLoss(net="vgg", crop=128).to(DEV).eval()
    lr_at = cosine_lr(LR, ITERS, warmup=WARMUP)
    loader = make_dataloader(SHARDS, batch_size=BATCH, num_workers=WORKERS, patch=PATCH, seed=42)
    it = iter(loader)
    print(f"[cap:{TAG}] params={n_params:,} iters={ITERS}", flush=True)
    t0 = time.time()
    for step in range(ITERS):
        g = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = g
        jpeg, target, _cond65 = next(it)
        jpeg = jpeg.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        target = target.to(DEV, non_blocking=True)
        ctx = build_ctx(_cond65, torch.zeros(len(_cond65), 32)).to(DEV, non_blocking=True)
        with bf16_autocast(True, device_type="cuda"):
            pred = gen_fwd(jpeg, ctx)
            l1 = (pred.float() - target).abs().mean()
            lp = percept(pred.float(), target)
            loss = l1 + lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(gen.parameters(), GRAD_CLIP)
        opt.step()
        ema.update(gen)
        if step % 1000 == 0 or step == ITERS - 1:
            print(f"[cap:{TAG}] it={step:5d} l1={l1.item():.4f} lpips={lp.item():.4f} "
                  f"{1000*(time.time()-t0)/(step+1):.0f}ms/it", flush=True)
    with ema.swap(gen):
        torch.save({"model": gen.state_dict()}, OUT / f"student_cap_{TAG}.pt")
        gen.eval()
        c5 = {q: eval_set(gen, sorted(CLASSIC5.glob('*.bmp')), q) for q in (10, 20, 30, 40)}
        l1r = {q: eval_set(gen, sorted(LIVE1.glob('*.bmp')), q) for q in (10, 20, 30, 40)}
        for name, r in (("classic5", c5), ("live1", l1r)):
            for q, (p_, pb, pin) in r.items():
                print(f"[cap:{TAG}] {name} QF{q}: PSNR={p_:.2f} (in {pin:.2f}, gain {p_-pin:+.2f}) PSNR-B={pb:.2f}", flush=True)
    print(f"[cap:{TAG}] DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
