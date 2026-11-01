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

from dejpeg_train.data.jpegmeta import parse_jpeg
from dejpeg_train.data.loader import make_dataloader
from dejpeg_train.eval.metrics import psnr, psnr_b
from dejpeg_train.loss.noise import flat_hf_loss, peak_bin_loss, ssim_loss
from dejpeg_train.loss.perceptual import PerceptualLoss
from dejpeg_train.model.student import DeJPEGNetS, build_ctx
from dejpeg_train.paths import phase_dir, shards_dir, testsets_dir
from dejpeg_train.train.schedule import (
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
BATCH = int(os.environ.get("BATCH", "8"))
ACCUM = int(os.environ.get("ACCUM", "2"))
PATCH = int(os.environ.get("PATCH", "256"))
LAMBDA_L2 = float(os.environ.get("LAMBDA_L2", "0"))
CUTBLUR_P = float(os.environ.get("CUTBLUR_P", "0"))
WORKERS = int(os.environ.get("WORKERS", "6"))
SEQUENTIAL = os.environ.get("SEQUENTIAL", "1") == "1"  # HDD-friendly shard streaming
LAMBDA_LPIPS = float(os.environ.get("LAMBDA_LPIPS", "1.0"))
LAMBDA_SSIM = float(os.environ.get("LAMBDA_SSIM", "0"))
LAMBDA_FLATHF = float(os.environ.get("LAMBDA_FLATHF", "0"))
LAMBDA_PEAK = float(os.environ.get("LAMBDA_PEAK", "0"))
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
    SEED = int(os.environ.get("SEED", "0"))
    set_seed(SEED)
    gen = DeJPEGNetS(cond_mode="none", c0=int(os.environ.get("C0", "30"))).to(DEV, memory_format=torch.channels_last).train()
    gen_fwd = prepare_model_for_training(gen)
    opt = torch.optim.AdamW(gen.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.9), fused=True)
    ema = EMA(gen, decay=0.999)
    percept = PerceptualLoss(net="vgg", crop=128).to(DEV).eval()
    lr_at = cosine_lr(LR, ITERS, warmup=WARMUP)
    loader = make_dataloader(SHARDS, batch_size=BATCH, num_workers=WORKERS, patch=PATCH,
                             seed=42 + SEED, source_weights=WEIGHTS,
                             gray_frac=float(os.environ.get("GRAY_FRAC", "0")),
                             mozjpeg_frac=float(os.environ.get("MOZJPEG_FRAC", "0")),
                             sequential=SEQUENTIAL)
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
    print(f"[p2] params={n_params:,} iters={ITERS} accum={ACCUM} weights={WEIGHTS} "
          f"seed={SEED} lam: lpips={LAMBDA_LPIPS} ssim={LAMBDA_SSIM} flathf={LAMBDA_FLATHF} peak={LAMBDA_PEAK}", flush=True)

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
            with torch.autocast(device_type="cuda", enabled=False):
                p32 = pred.float()
                lp = percept(p32, target)
                # X5 CutBlur (report 6.7): paste target boxes into prediction so the
                # network sees real high-frequency content inside its own output
                # distribution; boxes land on the 8px JPEG grid, side uniform in
                # [patch/8, patch/2], per-sample.
                if CUTBLUR_P:
                    n, _, h, w = p32.shape
                    for bi in range(n):
                        if torch.rand(()).item() < CUTBLUR_P:
                            s = int(round(float(torch.empty(()).uniform_(h / 8, h / 2)) / 8)) * 8
                            s = max(8, min(s, h - 8))
                            y0 = int(torch.randint(0, h - s + 1, ())) // 8 * 8
                            x0 = int(torch.randint(0, w - s + 1, ())) // 8 * 8
                            p32[bi, :, y0:y0 + s, x0:x0 + s] = target[bi, :, y0:y0 + s, x0:x0 + s]
                l1 = (p32 - target).abs().mean()
                loss = l1 + LAMBDA_L2 * (p32 - target).pow(2).mean() + LAMBDA_LPIPS * lp
                if LAMBDA_SSIM:
                    ss = ssim_loss(p32, target)
                    loss = loss + LAMBDA_SSIM * ss
                else:
                    ss = torch.zeros((), device=DEV)
                if LAMBDA_FLATHF:
                    fh = flat_hf_loss(p32, target)
                    loss = loss + LAMBDA_FLATHF * fh
                else:
                    fh = torch.zeros((), device=DEV)
                if LAMBDA_PEAK:
                    pk = peak_bin_loss(p32, target)
                    loss = loss + LAMBDA_PEAK * pk
                else:
                    pk = torch.zeros((), device=DEV)
                loss.div(ACCUM).backward()
        clip_grad_norm(gen.parameters(), GRAD_CLIP)
        opt.step()
        ema.update(gen)
        el = time.time() - t0
        if step % LOG_EVERY == 0 or step == ITERS - 1:
            eta_h = (ITERS - step - 1) * (el / max(1, step - start + 1)) / 3600
            print(f"[p2] it={step:6d} lr={g:.2e} l1={l1.item():.4f} lpips={lp.item():.4f} "
                  f"ssim={ss.item():.4f} flathf={fh.item():.6f} peak={pk.item():.6f} "
                  f"{1000*el/max(1,step-start+1):.0f}ms/it eta {eta_h:.1f}h", flush=True)
        if step > start and step % CKPT_EVERY == 0:
            with ema.swap(gen):
                save_checkpoint(OUT / "student_p2_latest.pt", model=gen, ema=ema,
                                optimizer=opt, disc=None, step=step, manifest_hash="",
                                config={"phase": "2", "recipe": f"cond-none C0={int(os.environ.get('C0', '30'))} "
                                                                  f"lam_lpips={LAMBDA_LPIPS} lam_ssim={LAMBDA_SSIM} "
                                                                  f"lam_flathf={LAMBDA_FLATHF} lam_peak={LAMBDA_PEAK} seed={SEED}"},
                                rng_states=None)
        if step > start and step % EVAL_EVERY == 0:
            with ema.swap(gen):
                report(gen, f"it{step}")
            gen.train()

    with ema.swap(gen):
        torch.save({"model": gen.state_dict(), "ema": ema.state_dict()}, OUT / "student_p2_final.pt")
        report(gen, "FINAL")
        from dejpeg_train.export.onnx import export_onnx, fuse_for_export
        fused = fuse_for_export(gen)
        sample = (torch.rand(1, 3, PATCH, PATCH, device=DEV).contiguous(memory_format=torch.channels_last),
                  torch.zeros(1, 97, device=DEV))
        export_onnx(fused, sample, str(OUT / "student_p2_fp16.onnx"), opset=18, simplify=True, fp16=True)
        print("[p2] exported student_p2_fp16.onnx", flush=True)
    print(f"[p2] DONE {(time.time()-t0)/3600:.2f}h -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
