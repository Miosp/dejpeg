"""Phase 0.5 thin vertical slice (BLOCKING).

L1-only student sanity on DIV2K: 5k iters, batch 8, patch 256, bf16, channels_last.
Conditioning = GROUND-TRUTH quant table from the degradation record (deg_emb zeros,
no encoder yet). Exports FP16 ONNX for student + deg-encoder (early WebGPU op check)
and runs a QF-20 PSNR sanity (restored must beat input).

This is the architecture/browser-risk gate: if export or training reveals a problem,
we change the architecture HERE, before the long Phase-2 run.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dejpeg_train.data.batcher import QFBatcher
from dejpeg_train.data.controls import ControlConfig, ControlSampler
from dejpeg_train.data.degrade import DegradationConfig, DegradationSampler
from dejpeg_train.data.sources import DegradedBatchSource
from dejpeg_train.export.onnx import export_onnx, fuse_for_export
from dejpeg_train.model.conditioning import record_to_condition
from dejpeg_train.model.degencoder import DeJPEGNetE
from dejpeg_train.model.student import DeJPEGNetS
from dejpeg_train.paths import phase_dir, shards_dir
from dejpeg_train.train.schedule import EMA, bf16_autocast, cosine_lr, set_seed

SHARD_GLOB = str(shards_dir() / "div2k-*.tar")
OUT = phase_dir("phase05")

ITERS = 5000
BATCH = 8
PATCH = 256
LR = 2e-4
WARMUP = 200
SEED = 42


def hwc_to_chw_t(a: np.ndarray) -> torch.Tensor:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(a))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    dev = "cuda"
    source = DegradedBatchSource(
        SHARD_GLOB,
        degrade_config=DegradationConfig(),
        control_config=ControlConfig(),
        seed=SEED,
    )
    batcher = QFBatcher(source, batch_size=BATCH, accum_steps=1, seed=SEED)
    print(f"[info] shards indexed: {len(source)} patches", flush=True)

    model = DeJPEGNetS().to(dev, memory_format=torch.channels_last)
    ema = EMA(model, decay=0.999)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = cosine_lr(LR, ITERS, warmup=WARMUP)
    py_rng = random.Random(SEED)

    t0 = time.time()
    for step in range(1, ITERS + 1):
        samples = [s for micro in batcher.step() for s in micro]
        # Per-sample 256 crop: post-resize can leave a patch at 511 (sampler's
        # 512 guarantee has a gap on undershoot), so crop each sample independently
        # with its own valid offset rather than stack-then-crop.
        jc_list, tc_list, cond_list = [], [], []
        for s in samples:
            j = hwc_to_chw_t(s["jpeg"])
            t = hwc_to_chw_t(s["target"])
            h, w = j.shape[-2], j.shape[-1]
            dy = py_rng.randint(0, h - PATCH)
            dx = py_rng.randint(0, w - PATCH)
            jc_list.append(j[:, dy : dy + PATCH, dx : dx + PATCH])
            tc_list.append(t[:, dy : dy + PATCH, dx : dx + PATCH])
            cond_list.append(record_to_condition(s["record"], dropout_p=0.0))
        jc = torch.stack(jc_list).to(dev, memory_format=torch.channels_last)
        tc = torch.stack(tc_list).to(dev)
        degz = torch.zeros(BATCH, 32)
        ctx = torch.cat([torch.stack(cond_list), degz], dim=1).to(dev)
        for g in opt.param_groups:
            g["lr"] = sched(step - 1)
        with bf16_autocast(True, device_type="cuda"):
            out = model(jc, ctx)
            loss = F.l1_loss(out, tc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        if step % 250 == 0 or step == 1:
            print(
                f"[{step}/{ITERS}] loss={loss.item():.5f} lr={sched(step-1):.2e} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    # ---- EMA swap for eval/export ----
    ckpt = {
        "step": ITERS,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
    }
    torch.save(ckpt, OUT / "student.pt")
    print(f"[info] checkpoint saved -> {OUT/'student.pt'}", flush=True)

    with ema.swap(model):
        run_qf20_sanity(model, source, dev)
        export_graphs(model)

    print("[done] Phase 0.5 training + export complete", flush=True)


def run_qf20_sanity(model: torch.nn.Module, source, dev: str) -> None:
    """Restored PSNR must beat input PSNR on QF~20 pairs."""
    # Only REAL degraded pairs: source.draw injects ~10% passthrough controls
    # (jpeg==target -> mse=0 -> PSNR=inf) which would poison the average.
    rng = random.Random(0)
    psnr_in = psnr_out = 0.0
    n = 0
    target = 16
    attempts = 0
    model.eval()
    with torch.no_grad():
        while n < target and attempts < 400:
            attempts += 1
            s = source.draw(18, 22, rng)
            if s.get("is_control"):
                continue
            j = hwc_to_chw_t(s["jpeg"]).unsqueeze(0).to(dev)
            t = hwc_to_chw_t(s["target"]).unsqueeze(0).to(dev)
            cond = record_to_condition(s["record"], dropout_p=0.0).unsqueeze(0).to(dev)
            ctx = torch.cat([cond, torch.zeros(1, 32).to(dev)], dim=1)
            with bf16_autocast(True, device_type="cuda"):
                out = model(j, ctx).float()
            pi = psnr(j, t).item()
            po = psnr(out, t).item()
            if pi == float("inf") or po == float("inf"):
                continue
            psnr_in += pi
            psnr_out += po
            n += 1
    psnr_in /= max(n, 1)
    psnr_out /= max(n, 1)
    verdict = "PASS" if psnr_out > psnr_in else "FAIL"
    print(f"[sanity] QF20 PSNR input={psnr_in:.2f} restored={psnr_out:.2f} "
          f"margin={psnr_out - psnr_in:+.2f}dB n={n} -> {verdict}", flush=True)
    (OUT / "qf20_sanity.json").write_text(
        json.dumps({"psnr_input": psnr_in, "psnr_restored": psnr_out, "verdict": verdict}, indent=2)
    )


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = ((a - b) ** 2).mean()
    if mse <= 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(1.0 / mse)


def export_graphs(model: torch.nn.Module) -> None:
    fused = fuse_for_export(model).cpu().eval()
    tile = torch.randn(1, 3, PATCH, PATCH)
    ctx = torch.randn(1, 97)
    export_onnx(fused, (tile, ctx), OUT / "student_fp16.onnx", opset=17, simplify=True, fp16=True)
    print(f"[info] student FP16 ONNX -> {OUT/'student_fp16.onnx'}", flush=True)
    enc = DeJPEGNetE().cpu().eval()
    img = torch.randn(1, 3, 256, 256)
    export_onnx(enc, (img,), OUT / "degencoder_fp16.onnx", opset=17, simplify=True, fp16=True)
    print(f"[info] deg-encoder FP16 ONNX -> {OUT/'degencoder_fp16.onnx'}", flush=True)


if __name__ == "__main__":
    main()
