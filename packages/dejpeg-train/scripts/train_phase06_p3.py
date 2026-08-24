"""Phase 0.6 P3 -- conditioning ablation: none vs scalar-FiLM vs prompt.

Resolves the architecture decision before Phase 1/2: is the PromptCIR prompt
module (per-position conditioning) worth its params+complexity over simpler
scalar-FiLM (one gamma,beta per image) or no conditioning? Trains all three with
the strong P1 recipe (L1+LPIPS, compile + 6 workers) and compares Classic5 PSNR.
Spec hypothesis: prompt should win (esp. for spatially-varying JPEG).
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from dejpeg.data.jpegmeta import parse_jpeg
from dejpeg.data.loader import make_dataloader, sample_condition
from dejpeg.data.sources import DegradedBatchSource
from dejpeg.eval.metrics import psnr, psnr_b
from dejpeg.eval.panel import contact_sheet
from dejpeg.loss.perceptual import PerceptualLoss
from dejpeg.model.conditioning import quant_table_to_condition
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import shards_dir, phase_dir, testsets_dir
from dejpeg.train.schedule import (
    EMA, bf16_autocast, clip_grad_norm, cosine_lr, prepare_model_for_training, set_seed,
)

SHARDS = [str(shards_dir() / "div2k-*.tar"),
          str(shards_dir() / "flickr2k-*.tar")]
OUT = phase_dir("phase06") / "p3"
OUT.mkdir(parents=True, exist_ok=True)
CLASSIC5 = testsets_dir("Classic5")
ITERS = 30_000
BATCH = 8
WORKERS = 6
PATCH = 256
LR = 1e-3
WARMUP = 500
GRAD_CLIP = 1.0
LOG_EVERY = 1000
DEV = "cuda"
FBCNN = {20: (32.01, 31.59), 30: (33.27, 32.70)}


@torch.no_grad()
def eval_classic5(model):
    import cv2
    model.eval()
    imgs = sorted(list(CLASSIC5.glob("*.bmp")) + list(CLASSIC5.glob("*.png")))
    out = {}
    for qf in (20, 30):
        psnrs, psnrbs = [], []
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            rgb = bgr[:, :, ::-1].copy()
            _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
            meta = parse_jpeg(enc.tobytes())
            cond = quant_table_to_condition(meta.quant_tables[0].values, 1.0).to(DEV)
            ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=DEV))
            inp = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.0
            _, _, h, w = inp.shape
            inp32 = F.pad(inp, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))
            o = model(inp32.contiguous(memory_format=torch.channels_last), ctx)[:, :, :h, :w].clamp(0, 1)
            o = (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            psnrs.append(psnr(rgb, o)); psnrbs.append(psnr_b(rgb, o))
        if psnrs:
            out[qf] = (float(np.mean(psnrs)), float(np.mean(psnrbs)))
    model.train()
    return out


def run_variant(variant):
    set_seed(0)
    gen = DeJPEGNetS(cond_mode=variant).to(DEV, memory_format=torch.channels_last).train()
    gen_fwd = prepare_model_for_training(gen)
    n_params = sum(p.numel() for p in gen.parameters())
    opt = torch.optim.AdamW(gen.parameters(), lr=LR, weight_decay=1e-3, betas=(0.9, 0.9), fused=True)
    ema = EMA(gen, decay=0.999)
    percept = PerceptualLoss(net="vgg", crop=128).to(DEV).eval()
    lr_at = cosine_lr(LR, ITERS, warmup=WARMUP)
    loader = make_dataloader(SHARDS, batch_size=BATCH, num_workers=WORKERS, patch=PATCH, seed=42)
    it = iter(loader)
    print(f"\n[p3:{variant}] params={n_params:,}  iters={ITERS}", flush=True)
    t0 = time.time()
    for step in range(ITERS):
        g = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = g
        jpeg, target, cond = next(it)
        jpeg = jpeg.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        target = target.to(DEV, non_blocking=True)
        ctx = build_ctx(cond, torch.zeros(len(cond), 32)).to(DEV, non_blocking=True)
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
        if step % LOG_EVERY == 0 or step == ITERS - 1:
            el = time.time() - t0
            print(f"[p3:{variant}] it={step:5d} l1={l1.item():.4f} lpips={lp.item():.4f} "
                  f"{1000*el/max(1,step+1):.0f}ms/it", flush=True)
    with ema.swap(gen):
        torch.save({"model": gen.state_dict()}, OUT / f"student_p3_{variant}.pt")
        res = eval_classic5(gen) if CLASSIC5.exists() else {}
        for qf, (p_, pb) in res.items():
            fb_p, fb_pb = FBCNN[qf]
            print(f"[p3:{variant}] Classic5 QF{qf}: PSNR={p_:.2f} ({p_-fb_p:+.2f}) "
                  f"PSNR-B={pb:.2f} ({pb-fb_pb:+.2f})", flush=True)
        with torch.no_grad():
            csrc = DegradedBatchSource(SHARDS, seed=7)
            rng = random.Random(7)
            pairs = []
            while len(pairs) < 6:
                s = csrc.draw(15, 35, rng)
                if not s["is_control"]:
                    pairs.append(s)
            sheet = []
            for s in pairs:
                j = torch.from_numpy(s["jpeg"]).permute(2, 0, 1).unsqueeze(0).float().to(DEV).contiguous(memory_format=torch.channels_last)
                c = sample_condition(s).to(DEV)
                ctx = build_ctx(c.unsqueeze(0), torch.zeros(1, 32, device=DEV))
                o = gen(j, ctx).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                sheet.append((s["jpeg"] * 255).astype(np.uint8))
                sheet.append((o * 255).astype(np.uint8))
            contact_sheet(sheet, cols=2, thumb=320, path=str(OUT / f"contact_p3_{variant}.png"))
    elapsed = (time.time() - t0) / 60
    print(f"[p3:{variant}] DONE in {elapsed:.1f}min  params={n_params:,}", flush=True)
    return variant, n_params, res, elapsed


def main():
    print("[p3] conditioning ablation: none vs scalar-FiLM vs prompt (30k iters each)", flush=True)
    results = []
    for v in ("none", "scalar", "prompt"):
        try:
            results.append(run_variant(v))
        except Exception as e:
            print(f"[p3:{v}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("\n=== P3 COMPARISON ===", flush=True)
    print(f"{'variant':<10} {'params':>10} {'QF20 PSNR':>10} {'QF20 PSNR-B':>12} {'QF30 PSNR':>10} {'min':>6}", flush=True)
    for v, np_, res, el in results:
        q20 = res.get(20, (float("nan"), float("nan")))
        q30 = res.get(30, (float("nan"), float("nan")))
        print(f"{v:<10} {np_:>10,} {q20[0]:>10.2f} {q20[1]:>12.2f} {q30[0]:>10.2f} {el:>6.1f}", flush=True)


if __name__ == "__main__":
    main()
