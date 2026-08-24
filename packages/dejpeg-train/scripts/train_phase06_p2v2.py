"""Phase 0.6 P2 -- 30k iters full loss stack on DF2K (L1+LPIPS+blockiness+LDL+PatchGAN).

Built on the optimized P1 pipeline (compile + 6 workers). Answers: do the losses
cohere or fight, and is the discriminator balanced? Loss weights per spec §3:
L1 1.0, LPIPS-VGG 1.0, relative blockiness 0.5, LDL 1.0, adversarial 0.15 (gen),
disc 1.0. Generator (student) gets EMA; discriminator does not.
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
from dejpeg.loss.blockiness import blockiness_loss
from dejpeg.loss.gan import PatchDiscriminator, discriminator_hinge_loss, generator_hinge_loss
from dejpeg.loss.ldl import ldl_loss
from dejpeg.loss.perceptual import PerceptualLoss
from dejpeg.model.conditioning import quant_table_to_condition
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import shards_dir, phase_dir, testsets_dir
from dejpeg.train.schedule import (
    EMA, bf16_autocast, clip_grad_norm, cosine_lr, prepare_model_for_training, set_seed,
)

SHARDS = [str(shards_dir() / "div2k-*.tar"),
          str(shards_dir() / "flickr2k-*.tar")]
OUT = phase_dir("phase06") / "p2v2"
OUT.mkdir(parents=True, exist_ok=True)
CLASSIC5 = testsets_dir("Classic5")
ITERS = 30_000
BATCH = 8
WORKERS = 6
PATCH = 256
LR = 1e-3
LR_D = 5e-4              # raised from P2 (1e-4 -> inert disc)
WD = 1e-3
WARMUP = 500
GRAD_CLIP = 1.0
LOG_EVERY = 500
W_L1, W_LP, W_BLK, W_LDL, W_ADV = 1.0, 1.0, 0.5, 1.0, 0.15
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


def main():
    set_seed(0)
    gen = DeJPEGNetS().to(DEV, memory_format=torch.channels_last).train()
    gen_fwd = prepare_model_for_training(gen)
    disc = PatchDiscriminator(in_ch=6, n_layers=4).to(DEV).train()
    g_opt = torch.optim.AdamW(gen.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.9), fused=True)
    d_opt = torch.optim.AdamW(disc.parameters(), lr=LR_D, weight_decay=0, betas=(0.9, 0.9), fused=True)
    ema = EMA(gen, decay=0.999)
    percept = PerceptualLoss(net="vgg", crop=128).to(DEV).eval()
    lr_at = cosine_lr(LR, ITERS, warmup=WARMUP)
    loader = make_dataloader(SHARDS, batch_size=BATCH, num_workers=WORKERS, patch=PATCH, seed=42)
    it = iter(loader)
    print(f"[p2v2] gen params={sum(p.numel() for p in gen.parameters()):,}  disc params="
          f"{sum(p.numel() for p in disc.parameters()):,}  iters={ITERS}", flush=True)

    t0 = time.time()
    for step in range(ITERS):
        g = lr_at(step)
        for pg in g_opt.param_groups:
            pg["lr"] = g
        jpeg, target, cond = next(it)
        jpeg = jpeg.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        target = target.to(DEV, non_blocking=True)
        ctx = build_ctx(cond, torch.zeros(len(cond), 32)).to(DEV, non_blocking=True)

        # ---- generator step ----
        with bf16_autocast(True, device_type="cuda"):
            pred = gen_fwd(jpeg, ctx)
            pred_f = pred.float()
            l1 = (pred_f - target).abs().mean()
            lp = percept(pred_f, target)
            blk = blockiness_loss(pred_f, target, offset=(0, 0))
            ldl = ldl_loss(pred_f, target)
            fake_logits = disc(torch.cat([jpeg, pred_f], dim=1))
            adv = generator_hinge_loss(fake_logits)
            g_loss = W_L1 * l1 + W_LP * lp + W_BLK * blk + W_LDL * ldl + W_ADV * adv
        g_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        clip_grad_norm(gen.parameters(), GRAD_CLIP)
        g_opt.step()
        ema.update(gen)

        # ---- discriminator step ----
        with bf16_autocast(True, device_type="cuda"):
            real_logits = disc(torch.cat([jpeg, target], dim=1))
            fake_logits = disc(torch.cat([jpeg, pred.detach().float()], dim=1))
            d_loss = discriminator_hinge_loss(real_logits, fake_logits)
        d_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        d_opt.step()

        if step % LOG_EVERY == 0 or step == ITERS - 1:
            el = time.time() - t0
            eta = (ITERS - step - 1) * (el / max(1, step + 1))
            print(f"[p2v2] it={step:5d} l1={l1.item():.4f} lpips={lp.item():.4f} "
                  f"blk={blk.item():.4f} ldl={ldl.item():.4f} adv={adv.item():.4f} "
                  f"d={d_loss.item():.4f}  {1000*el/max(1,step+1):.0f}ms/it  eta {eta/60:.1f}min",
                  flush=True)

    with ema.swap(gen):
        torch.save({"model": gen.state_dict(), "ema": ema.state_dict()}, OUT / "student_p2v2.pt")
        with torch.no_grad():
            if CLASSIC5.exists():
                res = eval_classic5(gen)
                for qf, (p_, pb) in res.items():
                    fb_p, fb_pb = FBCNN[qf]
                    print(f"[p2v2] Classic5 QF{qf}: PSNR={p_:.2f} ({p_-fb_p:+.2f} vs FBCNN)  "
                          f"PSNR-B={pb:.2f} ({pb-fb_pb:+.2f} vs FBCNN)", flush=True)
            # contact sheet
            csrc = DegradedBatchSource(SHARDS, seed=7)
            rng = random.Random(7)
            pairs = []
            while len(pairs) < 8:
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
            contact_sheet(sheet, cols=2, thumb=320, path=str(OUT / "contact_sheet_p2v2.png"))
            print("[p2v2] contact sheet -> contact_sheet_p2v2.png", flush=True)
        fused = fuse_for_export(gen)
        sample = (torch.rand(1, 3, PATCH, PATCH, device=DEV, memory_format=torch.channels_last),
                  torch.rand(1, 97, device=DEV))
        export_onnx(fused, sample, str(OUT / "student_p2v2_fp16.onnx"), opset=17, simplify=True, fp16=True)
        print("[p2v2] exported student_p2v2_fp16.onnx", flush=True)
    print(f"[p2v2] DONE in {(time.time()-t0)/60:.1f}min  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
