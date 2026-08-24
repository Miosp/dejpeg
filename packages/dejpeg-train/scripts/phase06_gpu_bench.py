"""GPU-compute benchmark: squeeze maximum throughput from the student on a 10GB GPU.

Measures forward + backward + optimizer (model only, no CPU degrade) across:
  baseline | +tf32+cudnn | +fused AdamW | +torch.compile | +max-autotune(CUDA graphs)
  | +grad-checkpoint + batch16 (throughput-per-sample)
Reports ms/iter, ms/sample, peak VRAM. Identifies the best GPU-compute config.
"""
import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.train.schedule import enable_fast_gpu

BATCH = 8
PATCH = 256
DEV = "cuda"
N_TIME = 20
N_WARMUP = 12  # compile / cudnn-autotune warmup


def make_inputs(batch):
    j = torch.randn(batch, 3, PATCH, PATCH, device=DEV)
    q = torch.rand(batch, 65, device=DEV)
    deg = torch.zeros(batch, 32, device=DEV)
    return j.contiguous(memory_format=torch.channels_last), build_ctx(q, deg)


def time_config(label, *, tf32, compile_mode, ckpt, fused, batch):
    # set tf32 flags per config
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = tf32  # only meaningful with fast-gpu on
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = DeJPEGNetS(grad_checkpoint=ckpt).to(DEV, memory_format=torch.channels_last).train()
    n_params = sum(p.numel() for p in model.parameters())
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused)
    j, ctx = make_inputs(batch)
    target = torch.randn(batch, 3, PATCH, PATCH, device=DEV)

    try:
        for _ in range(N_WARMUP):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(j, ctx)
                loss = (out.float() - target).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_TIME):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(j, ctx)
                loss = (out.float() - target).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        ms_iter = 1000 * dt / N_TIME
        vram = torch.cuda.max_memory_allocated() / 1e6
        print(f"{label:26s} batch={batch:2d}  {ms_iter:6.1f}ms/it  "
              f"{1000*ms_iter/batch:5.1f}ms/sample  VRAM {vram:5.0f}MB")
        return ms_iter
    except Exception as e:
        print(f"{label:26s} ERROR: {type(e).__name__}: {str(e)[:90]}")
        return None
    finally:
        del model, opt, j, ctx, target, out, loss
        gc.collect()
        torch.cuda.empty_cache()


def main():
    print(f"student {sum(p.numel() for p in DeJPEGNetS().parameters()):,} params, "
          f"bf16, channels_last, fwd+bwd+opt only, time over {N_TIME} iters\n")
    time_config("baseline", tf32=False, compile_mode=None, ckpt=False, fused=False, batch=8)
    time_config("+tf32+cudnn", tf32=True, compile_mode=None, ckpt=False, fused=False, batch=8)
    time_config("+tf32+fused-adamw", tf32=True, compile_mode=None, ckpt=False, fused=True, batch=8)
    time_config("+compile(default)", tf32=True, compile_mode="default", ckpt=False, fused=True, batch=8)
    time_config("+max-autotune(cudagraph)", tf32=True, compile_mode="max-autotune", ckpt=False, fused=True, batch=8)
    time_config("+ckpt batch16", tf32=True, compile_mode="max-autotune", ckpt=True, fused=True, batch=16)
    time_config("+ckpt batch24", tf32=True, compile_mode="max-autotune", ckpt=True, fused=True, batch=24)


if __name__ == "__main__":
    main()
