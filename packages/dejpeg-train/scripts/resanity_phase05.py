#!/usr/bin/env python3
"""Standalone re-sanity for Phase 0.5: load student.pt (EMA weights), measure
QF~20 deblocking on REAL degraded pairs only (controls excluded)."""
import glob, random, sys, torch
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO + "/src")
from dejpeg.paths import phase_dir, shards_dir

OUT = str(phase_dir("phase05"))
SHARDS = str(shards_dir() / "div2k-*.tar")
DEV = "cuda"

from dejpeg.model.student import DeJPEGNetS
from dejpeg.model.conditioning import record_to_condition
from dejpeg.data.sources import DegradedBatchSource
from dejpeg.train.schedule import EMA, bf16_autocast


def hwc_to_chw_t(a):
    return torch.from_numpy(a).permute(2, 0, 1).float()


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    if mse <= 0:
        return float("inf")
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    ckpt = torch.load(f"{OUT}/student.pt", map_location="cpu", weights_only=False)
    model = DeJPEGNetS().to(DEV)
    ema = EMA(model)
    ema.load_state_dict(ckpt["ema"])
    model.eval()

    source = DegradedBatchSource(SHARDS, seed=7)

    rng = random.Random(0)
    psnr_in = psnr_out = 0.0
    collected = 0
    target = 32
    attempts = 0
    with ema.swap(model), torch.no_grad():
        while collected < target and attempts < 600:
            attempts += 1
            s = source.draw(18, 22, rng)
            if s.get("is_control"):
                continue
            j = hwc_to_chw_t(s["jpeg"]).unsqueeze(0).to(DEV)
            t = hwc_to_chw_t(s["target"]).unsqueeze(0).to(DEV)
            cond = record_to_condition(s["record"], dropout_p=0.0).unsqueeze(0).to(DEV)
            ctx = torch.cat([cond, torch.zeros(1, 32).to(DEV)], dim=1)
            with bf16_autocast(True, device_type="cuda"):
                out = model(j, ctx).float()
            pi = psnr(j, t)
            po = psnr(out, t)
            if pi == float("inf") or po == float("inf"):
                continue
            psnr_in += pi
            psnr_out += po
            collected += 1

    psnr_in /= collected
    psnr_out /= collected
    margin = psnr_out - psnr_in
    verdict = "PASS" if psnr_out > psnr_in else "FAIL"
    print(f"[resanity] QF20 n={collected} input={psnr_in:.2f}dB restored={psnr_out:.2f}dB "
          f"margin={margin:+.2f}dB -> {verdict}")
    import json
    with open(f"{OUT}/qf20_sanity.json", "w") as f:
        json.dump({"psnr_input": psnr_in, "psnr_restored": psnr_out,
                   "margin_db": margin, "verdict": verdict, "n": collected}, f, indent=2)


if __name__ == "__main__":
    main()
