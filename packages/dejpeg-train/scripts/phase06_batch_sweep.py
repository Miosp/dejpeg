"""Batch-sweep with torch.compile+grad-checkpoint: find the throughput-per-sample
sweet spot now that compile freed VRAM (5.6GB at batch 8 -> bigger batches fit)."""
import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.train.schedule import enable_fast_gpu

PATCH = 256
DEV = "cuda"
enable_fast_gpu()


def run(batch, ckpt):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model = DeJPEGNetS(grad_checkpoint=ckpt).to(DEV, memory_format=torch.channels_last).train()
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    j = torch.randn(batch, 3, PATCH, PATCH, device=DEV).contiguous(memory_format=torch.channels_last)
    ctx = build_ctx(torch.rand(batch, 65, device=DEV), torch.zeros(batch, 32, device=DEV))
    t = torch.randn(batch, 3, PATCH, PATCH, device=DEV)
    try:
        for _ in range(12):  # compile warmup
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(j, ctx); loss = (out.float()-t).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(j, ctx); loss = (out.float()-t).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); dt = time.time()-t0
        ms = 1000*dt/20; vram = torch.cuda.max_memory_allocated()/1e6
        print(f"batch={batch:2d} ckpt={int(ckpt)}  {ms:6.1f}ms/it  {1000*ms/batch:5.1f}ms/sample  VRAM {vram:5.0f}MB")
    except torch.cuda.OutOfMemoryError:
        print(f"batch={batch:2d} ckpt={int(ckpt)}  OOM")
    finally:
        del model, opt, j, ctx, t; gc.collect(); torch.cuda.empty_cache()


print("compile(default) + tf32 + fused-adamw; ms/sample is the throughput metric (lower=better)\n")
run(8, False)
run(12, False)
run(16, False)
run(16, True)
run(24, True)
run(32, True)
