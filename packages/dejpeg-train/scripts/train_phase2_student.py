"""Phase 2 -- final student run (recipe locked by Phase 0.6/0.7 evidence).

Architecture: DeJPEGNetS cond_mode=none, C0=30 (1.49M params, ~3.0MB FP16) --
conditioning deleted (0.7.2), capacity flat (0.7.3), browser-safe.
Losses: L1 + LPIPS-VGG(128-crop) -- GAN (0.6), SSIM (0.7.4), LDL/blockiness
(spec-weight-inert in 0.6) all excluded.
Data: FULL corpus, per-source weights (user_raws 0.35 / div2k 0.25 / flickr2k
0.20 / liu4k_v2 0.20 -- H4), 10% controls, patch 256.
Optim: AdamW lr 1e-3 wd 1e-3, warmup 2000, cosine->0, effective batch 16 via
grad-accum x2 (H6), grad-clip 1.0, EMA 0.999 (eval/ship), bf16 + channels_last
+ torch.compile, 6-worker loader.

Env: ITERS (default 250000), OUT_DIR, RESUME (ckpt path), EVAL_EVERY (50000),
CKPT_EVERY (10000).
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
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import phase_dir, shards_dir, testsets_dir
from dejpeg.train.schedule import (
    EMA, bf16_autocast, clip_grad_norm, cosine_lr, prepare_model_for_training,
    save_checkpoint, set_seed,
)

ITERS = int(os.environ.get("ITERS", "250000"))
OUT = Path(os.environ.get("OUT_DIR", str(phase_dir("phase2"))))
RESUME = os.environ.get("RESUME", "")
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "50000"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "10000"))
OUT.mkdir(parents=True, exist_ok=True)

SHARDS = [str(shards_dir() / f"{s}-*.tar")
          for s in ("div2k", "flickr2k", "liu4k_v2", "user_raws")]
WEIGHTS = {"user_raws": 0.35, "div2k": 0.25, "flickr2k": 0.20, "liu4k_v2": 0.20}
CLASSIC5 = testsets_dir("Classic5")
LIVE1 = testsets_dir("LIVE1_color")
BATCH, ACCUM, WORKERS, PATCH = 8, 2, 6, 256
LR, WD, WARMUP, GRAD_CLIP = float(os.environ.get("LR", "1e-3")), 1e-3, int(os.environ.get("WARMUP", "2000")), 1.0
LOG_EVERY = 500
DEV = "cuda"
FBCNN_C5 = {10: (29.80, 29.55), 20: (32.01, 31.59), 30: (33.27, 32.70), 40: (34.12, 33.46)}
FBCNN_L1 = {10: (27.77, 27.51), 20: (30.11, 29.70), 30: (31.43, 30.93), 40: (32.34, 31.80)}


@torch.no_grad()
def eval_set(model, imgs, qf):
    import cv2
    ps, pbs = [], []
    for p in imgs:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = bgr[:, :, ::-1].copy()
        _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        t = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
        _, _, h, w = t.shape
        t32 = F.pad(t, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))
        o = model(t32.contiguous(memory_format=torch.channels_last),
                  torch.zeros(1, 97, device=DEV))[:, :, :h, :w].clamp(0, 1)
        o = (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        ps.append(psnr(rgb, o)); pbs.append(psnr_b(rgb, o))
    return float(np.mean(ps)), float(np.mean(pbs))


def report(model, tag):
    gen_eval = model
    gen_eval.eval()
    for name, imgs, base in (("classic5", sorted(CLASSIC5.glob("*.bmp")), FBCNN_C5),
                             ("live1", sorted(LIVE1.glob("*.bmp")), FBCNN_L1)):
        for qf in (10, 20, 30, 40):
            p_, pb = eval_set(gen_eval, imgs, qf)
            fb_p, fb_pb = base[qf]
            print(f"[{tag}] {name} QF{qf}: PSNR={p_:.2f} ({p_-fb_p:+.2f} vs FBCNN) "
                  f"PSNR-B={pb:.2f} ({pb-fb_pb:+.2f})", flush=True)
    gen_eval.train()


def main():
    set_seed(0)
    gen = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "30"))).to(DEV, memory_format=torch.channels_last).train()
    gen_fwd = prepare_model_for_training(gen)
    opt = torch.optim.AdamW(gen.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.9), fused=True)
    ema = EMA(gen, decay=0.999)
    percept = PerceptualLoss(net="vgg", crop=128).to(DEV).eval()
    lr_at = cosine_lr(LR, ITERS, warmup=WARMUP)
    loader = make_dataloader(SHARDS, batch_size=BATCH, num_workers=WORKERS, patch=PATCH,
                             seed=42, source_weights=WEIGHTS,
                             gray_frac=float(os.environ.get("GRAY_FRAC", "0")))
    it = iter(loader)
    start = 0
    if RESUME:
        ck = torch.load(RESUME, map_location=DEV, weights_only=False)
        gen.load_state_dict(ck["model"])
        if "ema" in ck and ck["ema"]:
            ema.load_state_dict(ck["ema"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        start = int(os.environ.get("STEP0", ck.get("step", 0)))
        print(f"[p2] resumed from {RESUME} at step {start}", flush=True)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"[p2] params={n_params:,} iters={ITERS} accum={ACCUM} weights={WEIGHTS}", flush=True)

    t0 = time.time()
    for step in range(start, ITERS):
        g = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = g
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            jpeg, target, c65 = next(it)
            jpeg = jpeg.to(DEV, memory_format=torch.channels_last, non_blocking=True)
            target = target.to(DEV, non_blocking=True)
            ctx = build_ctx(c65, torch.zeros(len(c65), 32)).to(DEV, non_blocking=True)
            with bf16_autocast(True, device_type="cuda"):
                pred = gen_fwd(jpeg, ctx)
                l1 = (pred.float() - target).abs().mean()
                lp = percept(pred.float(), target)
                (l1 + lp).div(ACCUM).backward()
        clip_grad_norm(gen.parameters(), GRAD_CLIP)
        opt.step()
        ema.update(gen)
        el = time.time() - t0
        if step % LOG_EVERY == 0 or step == ITERS - 1:
            eta_h = (ITERS - step - 1) * (el / max(1, step - start + 1)) / 3600
            print(f"[p2] it={step:6d} lr={g:.2e} l1={l1.item():.4f} lpips={lp.item():.4f} "
                  f"{1000*el/max(1,step-start+1):.0f}ms/it eta {eta_h:.1f}h", flush=True)
        if step > start and step % CKPT_EVERY == 0:
            with ema.swap(gen):
                save_checkpoint(OUT / "student_p2_latest.pt", model=gen, ema=ema,
                                optimizer=opt, disc=None, step=step, manifest_hash="",
                                config={"phase": "2", "recipe": "cond-none C0=30 L1+LPIPS eff16"},
                                rng_states=None)
        if step > start and step % EVAL_EVERY == 0:
            with ema.swap(gen):
                report(gen, f"it{step}")
            gen.train()

    with ema.swap(gen):
        torch.save({"model": gen.state_dict(), "ema": ema.state_dict()}, OUT / "student_p2_final.pt")
        report(gen, "FINAL")
        from dejpeg.export.onnx import export_onnx, fuse_for_export
        fused = fuse_for_export(gen)
        sample = (torch.rand(1, 3, PATCH, PATCH, device=DEV).contiguous(memory_format=torch.channels_last),
                  torch.zeros(1, 97, device=DEV))
        export_onnx(fused, sample, str(OUT / "student_p2_fp16.onnx"), opset=17, simplify=True, fp16=True)
        print("[p2] exported student_p2_fp16.onnx", flush=True)
    print(f"[p2] DONE {(time.time()-t0)/3600:.2f}h -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
