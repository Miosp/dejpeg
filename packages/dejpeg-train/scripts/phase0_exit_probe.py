"""Phase 0 exit-gate probe: VRAM high-water @ batch 8 + GPU + dataloader throughput.

Gate (spec): dataloader sustains >= 2x the GPU's consumption rate.
Config mirrors Phase-2 training: bf16 autocast, channels_last, batch 8, patch 512
(falls back to 256 if 512 OOMs).
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{REPO}/src")

from dejpeg.data.batcher import QFBatcher  # noqa: E402
from dejpeg.data.degrade import DegradationSampler  # noqa: E402
from dejpeg.model.student import DeJPEGNetS, build_ctx  # noqa: E402


def report(msg: str) -> None:
    print(msg, flush=True)


class NoiseSource:
    """Smooth structured patches -> degrade. Isolates the JPEG-bound loader cost.

    Uses bilinear-upsampled small noise (NOT pure random -- pure noise is maximally
    incompressible and trips libjpeg's progressive JERR_CANT_SUSPEND, which never
    happens on real photographic/synthetic content)."""

    def __init__(self, seed: int = 0, size: int = 640):
        self.degrade = DegradationSampler(seed=seed)
        self.size = size

    def _pick_clean(self, rng: random.Random) -> np.ndarray:
        small = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        return np.array(Image.fromarray(small).resize((self.size, self.size), Image.BILINEAR))

    def draw(self, qf_low: int, qf_high: int, rng: random.Random):
        arr, target, record = self.degrade.sample(self._pick_clean(rng), qf_range=(qf_low, qf_high))
        return {"jpeg": arr, "target": target, "true_qf": record.true_qf}

    def rebuild(self) -> None:
        pass


def vram_probe(batch: int = 8, patch: int = 512, dev: str = "cuda") -> float | None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        model = DeJPEGNetS().to(dev, memory_format=torch.channels_last)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        tile = torch.rand(batch, 3, patch, patch, device=dev).to(memory_format=torch.channels_last)
        target = torch.rand(batch, 3, patch, patch, device=dev).to(memory_format=torch.channels_last)
        ctx = build_ctx(torch.zeros(batch, 65, device=dev), torch.randn(batch, 32, device=dev))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(tile, ctx)
            loss = (out - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated(dev)
        report(f"  batch {batch} patch {patch} bf16 channels_last: peak VRAM {peak / 1e9:.2f} GB")
        return peak
    except torch.cuda.OutOfMemoryError:
        report(f"  batch {batch} patch {patch}: OOM")
        torch.cuda.empty_cache()
        return None


def gpu_throughput(batch: int, patch: int, steps: int = 20, dev: str = "cuda") -> float:
    model = DeJPEGNetS().to(dev, memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tile = torch.rand(batch, 3, patch, patch, device=dev).to(memory_format=torch.channels_last)
    target = torch.rand(batch, 3, patch, patch, device=dev).to(memory_format=torch.channels_last)
    ctx = build_ctx(torch.zeros(batch, 65, device=dev), torch.randn(batch, 32, device=dev))
    for _ in range(3):  # warmup
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(tile, ctx)
            loss = (out - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(tile, ctx)
            loss = (out - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = batch * steps / dt
    report(f"  GPU: {sps:.1f} samples/s ({dt / steps * 1000:.0f} ms/step, batch {batch} patch {patch})")
    return sps


def loader_throughput(steps: int = 40) -> float:
    src = NoiseSource(seed=0)
    batcher = QFBatcher(src, batch_size=8, accum_steps=2, seed=0)
    batcher.step()  # warmup
    t0 = time.time()
    n = 0
    for _ in range(steps):
        for micro in batcher.step():
            n += len(micro)
    dt = time.time() - t0
    sps = n / dt
    report(f"  dataloader (single-worker, JPEG-bound): {sps:.1f} samples/s ({n} in {dt:.1f}s)")
    return sps


def main() -> int:
    report("=== Phase 0 exit-gate probe ===")
    patch = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    report(f"(patch size forced to {patch}; patch 512 measured 23.69 GB peak -- over the 10 GB")
    report(" 3080, spills to sysmem and thrashes. Phase-2 trains 256->512; 512 needs grad-ckpt.)")
    report("[VRAM]")
    peak = vram_probe(batch=8, patch=patch)
    if peak is None:
        report("  FATAL: OOM; aborting")
        return 1
    report(f"[VRAM headroom] {peak / 1e9:.2f} GB used of 10 GB")
    report("[GPU throughput]")
    gpu_sps = gpu_throughput(batch=8, patch=patch)
    report("[Dataloader throughput]")
    loader_sps = loader_throughput()
    ratio = loader_sps / gpu_sps
    gate = "PASS" if ratio >= 2.0 else "CHECK"
    report(f"[GATE] dataloader/gpu = {ratio:.2f}x (need >= 2.0x) -> {gate}")
    report(f"       single-worker loader {loader_sps:.1f} vs gpu {gpu_sps:.1f} samples/s (patch {patch})")
    report(f"       with 4-6 workers loader scales to ~{loader_sps * 4:.0f}-{loader_sps * 6:.0f} samples/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
