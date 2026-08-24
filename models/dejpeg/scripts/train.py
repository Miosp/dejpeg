"""Train DeJPEGNet on a folder of clean images.

Recipe (matches the shipped checkpoint):
    L1 + LPIPS-VGG(128-crop), AdamW lr 1e-3 wd 1e-3 betas (0.9, 0.9),
    effective batch 16 (8 x grad-accum 2), grad-clip 1.0, warmup 2000,
    cosine anneal to 0, EMA 0.999 (EMA weights are what get saved/shipped),
    bf16 autocast + channels_last + torch.compile.

Example:
    python scripts/train.py --data ~/datasets/my-images --out runs/v1 --iters 150000
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from dejpeg.dataset import make_loader
from dejpeg.export import export_onnx
from dejpeg.losses import PerceptualLoss
from dejpeg.model import DeJPEGNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder of clean training images")
    p.add_argument("--out", default="runs/dejpeg")
    p.add_argument("--iters", type=int, default=150_000)
    p.add_argument("--c0", type=int, default=40, help="base width; 40 -> shipped 2.63M config")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--gray-frac", type=float, default=0.25,
                   help="fraction of pairs converted to grayscale (chroma anchor)")
    p.add_argument("--identity-frac", type=float, default=0.10,
                   help="fraction of pairs left clean (near-identity anchor)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ckpt-every", type=int, default=10_000)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def cosine_lr(base: float, total: int, warmup: int):
    def at(step: int) -> float:
        if step < warmup:
            return base * (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return at


class EMA:
    """Exponential moving average of parameters. Eval/ship the shadow, not raw."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_into(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            p.copy_(self.shadow[n])


def save_checkpoint(path: Path, model: DeJPEGNet, ema: EMA, step: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.shadow,
        "config": {"c0": args.c0},
    }, path)


def save_weights_fp16(model: DeJPEGNet, ema: EMA, args, path: Path) -> None:
    raw = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema.copy_into(model)
    sd = {k: v.detach().clone() for k, v in model.fuse().state_dict().items()}
    model.load_state_dict(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.half() for k, v in sd.items()},
                "config": {"c0": args.c0}}, path)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = DeJPEGNet(c0=args.c0, grad_checkpoint=True).to(device, memory_format=torch.channels_last)
    trainable = model
    if not args.no_compile and device == "cuda":
        trainable = torch.compile(model)
    print(f"params={sum(p.numel() for p in model.parameters()):,} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.9))
    ema = EMA(model, args.ema_decay)
    percept = PerceptualLoss().to(device).eval()
    lr_at = cosine_lr(args.lr, args.iters, args.warmup)
    loader = make_loader(args.data, batch_size=args.batch, num_workers=args.workers,
                         patch=args.patch, gray_frac=args.gray_frac, identity_frac=args.identity_frac)
    batches = iter(loader)

    t0 = time.time()
    for step in range(args.iters):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            jpeg, target = next(batches)
            jpeg = jpeg.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                pred = trainable(jpeg)
                loss = (pred.float() - target).abs().mean()
                loss = loss + percept(pred.float(), target)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if step % 500 == 0 or step == args.iters - 1:
            rate = 1000 * (time.time() - t0) / max(1, step + 1)
            eta_h = (args.iters - step - 1) * rate / 3_600_000
            print(f"it={step:6d} lr={lr_at(step):.2e} loss={loss.item():.4f} "
                  f"{rate:.0f}ms/it eta {eta_h:.1f}h", flush=True)
        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(out / "checkpoint.pt", model, ema, step, args)

    save_checkpoint(out / "checkpoint.pt", model, ema, args.iters, args)
    save_weights_fp16(model, ema, args, out / "dejpeg-fp16.pt")
    try:
        export_onnx(model, out / "dejpeg-fp16.onnx", size=args.patch)
        print(f"exported {out/'dejpeg-fp16.onnx'}")
    except ImportError:
        print("onnx extras not installed; skipping export (pip install .[export])")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
