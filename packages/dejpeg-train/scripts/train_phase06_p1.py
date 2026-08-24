"""Phase 0.6 P1 -- 10k iters L1 + LPIPS-VGG on DF2K (optimized pipeline).

Architecture locked by P0 (C0=30 + Simple-FFN, AdamW + grad-clip). Pipeline locked
by the GPU-util work: torch.compile + 6 process workers + prefetch -> 96% GPU util,
~104 ms/iter. First real-deblocking signal + contact sheet + Classic5 vs FBCNN.
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
from dejpeg.export.onnx import export_onnx, fuse_for_export
from dejpeg.loss.perceptual import PerceptualLoss
from dejpeg.model.conditioning import quant_table_to_condition
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import shards_dir, phase_dir, testsets_dir
from dejpeg.train.schedule import (
    EMA, bf16_autocast, clip_grad_norm, cosine_lr, prepare_model_for_training,
    save_checkpoint, set_seed,
)

SHARDS = [
    str(shards_dir() / "div2k-*.tar"),
    str(shards_dir() / "flickr2k-*.tar"),
]
OUT = phase_dir("phase06") / "p1"
OUT.mkdir(parents=True, exist_ok=True)
CLASSIC5 = testsets_dir("Classic5")

ITERS = 10_000
BATCH = 8
WORKERS = 6
PATCH = 256
LR = 1e-3
WD = 1e-3
WARMUP = 500
GRAD_CLIP = 1.0
CKPT_EVERY = 2000
LOG_EVERY = 200
DEV = "cuda"

# FBCNN baseline reference (Classic5 QF20/30) for the post-train comparison
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
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
            meta = parse_jpeg(enc.tobytes())
            cond = quant_table_to_condition(meta.quant_tables[0].values, 1.0).to(DEV)
            ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=DEV))
            inp = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).float().to(DEV)
            _, _, h, w = inp.shape
            inp32 = F.pad(inp, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))
            o = model(inp32.contiguous(memory_format=torch.channels_last), ctx)
            o = o[:, :, :h, :w].clamp(0, 1)
            o = (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            psnrs.append(psnr(rgb, o))
            psnrbs.append(psnr_b(rgb, o))
        if psnrs:
            out[qf] = (float(np.mean(psnrs)), float(np.mean(psnrbs)))
    model.train()
    return out


def main():
    set_seed(0)
    model = DeJPEGNetS().to(DEV, memory_format=torch.channels_last).train()
    model_fwd = prepare_model_for_training(model)  # tf32 + cudnn.benchmark + torch.compile
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.9), fused=True)
    ema = EMA(model, decay=0.999)
    percept = PerceptualLoss(net="vgg", crop=128).to(DEV).eval()
    lr_at = cosine_lr(LR, ITERS, warmup=WARMUP)
    loader = make_dataloader(SHARDS, batch_size=BATCH, num_workers=WORKERS, patch=PATCH, seed=42)
    it = iter(loader)
    print(f"[p1] params={sum(p.numel() for p in model.parameters()):,}  "
          f"iters={ITERS}  workers={WORKERS}  (compiled + worker-fed)", flush=True)

    t0 = time.time()
    for it_step in range(ITERS):
        g = lr_at(it_step)
        for pg in opt.param_groups:
            pg["lr"] = g
        jpeg, target, cond = next(it)
        jpeg = jpeg.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        target = target.to(DEV, non_blocking=True)
        ctx = build_ctx(cond, torch.zeros(len(cond), 32)).to(DEV, non_blocking=True)
        with bf16_autocast(True, device_type="cuda"):
            out = model_fwd(jpeg, ctx)
        out_f = out.float()
        l1 = (out_f - target).abs().mean()
        lp = percept(out_f, target)
        loss = l1 + lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(model.parameters(), GRAD_CLIP)
        opt.step()
        ema.update(model)
        if it_step % LOG_EVERY == 0 or it_step == ITERS - 1:
            el = time.time() - t0
            eta = (ITERS - it_step - 1) * (el / max(1, it_step + 1))
            print(f"[p1] it={it_step:5d} lr={g:.2e} l1={l1.item():.5f} lpips={lp.item():.5f} "
                  f"loss={loss.item():.5f}  {1000*el/max(1,it_step+1):.0f}ms/it  eta {eta/60:.1f}min",
                  flush=True)
        if it_step > 0 and it_step % CKPT_EVERY == 0:
            with ema.swap(model):
                save_checkpoint(OUT / f"student_p1_it{it_step}.pt", model=model, ema=ema,
                                optimizer=opt, disc=None, step=it_step, manifest_hash="",
                                config={"phase": "0.6-p1"}, rng_states=None)

    # finalize under EMA weights
    with ema.swap(model):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, OUT / "student_p1.pt")
        # contact sheet (8 fixed pairs)
        csrc = DegradedBatchSource(SHARDS, seed=7)
        rng = random.Random(7)
        pairs = []
        while len(pairs) < 8:
            s = csrc.draw(15, 35, rng)
            if not s["is_control"]:
                pairs.append(s)
        restored = []
        for s in pairs:
            j = torch.from_numpy(s["jpeg"]).permute(2, 0, 1).unsqueeze(0).float().to(DEV)
            cond = sample_condition(s).to(DEV)
            ctx = build_ctx(cond.unsqueeze(0), torch.zeros(1, 32, device=DEV))
            o = model(j.contiguous(memory_format=torch.channels_last), ctx).clamp(0, 1)[0]
            restored.append((s["jpeg"], o.permute(1, 2, 0).cpu().numpy()))
        sheet = []
        for jq, o in restored:
            sheet.append((jq * 255).astype(np.uint8))
            sheet.append((o * 255).astype(np.uint8))
        contact_sheet(sheet, cols=2, thumb=320, path=str(OUT / "contact_sheet_p1.png"))
        print("[p1] contact sheet written", flush=True)
        if CLASSIC5.exists():
            res = eval_classic5(model)
            for qf, (p_, pb) in res.items():
                fb_p, fb_pb = FBCNN[qf]
                d_p = p_ - fb_p
                d_pb = pb - fb_pb
                print(f"[p1] Classic5 QF{qf}: PSNR={p_:.2f} ({d_p:+.2f} vs FBCNN)  "
                      f"PSNR-B={pb:.2f} ({d_pb:+.2f} vs FBCNN)", flush=True)
        fused = fuse_for_export(model)
        sample = (torch.rand(1, 3, PATCH, PATCH, device=DEV, memory_format=torch.channels_last),
                  torch.rand(1, 97, device=DEV))
        export_onnx(fused, sample, str(OUT / "student_p1_fp16.onnx"), opset=17, simplify=True, fp16=True)
        print("[p1] exported student_p1_fp16.onnx", flush=True)
    print(f"[p1] DONE in {(time.time()-t0)/60:.1f}min  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
