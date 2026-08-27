"""Measure GPU utilization with the worker-parallel degrade loader.

Trains (compiled student) while sampling nvidia-smi GPU util, swept across
num_workers in {0, 4, 6}. Reports avg GPU util + iter time + ms/sample for each.
Goal: near-100% GPU util at the highest worker count (GPU never waits for data).
"""
import subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg_train.data.loader import make_dataloader
from dejpeg_train.model.student import DeJPEGNetS, build_ctx
from dejpeg_train.train.schedule import prepare_model_for_training
from dejpeg_train.paths import shards_dir

SHARDS = [str(shards_dir() / "div2k-*.tar"),
          str(shards_dir() / "flickr2k-*.tar")]
DEV = "cuda"
PATCH = 256
BATCH = 8
ITERS = 35


def gpu_util_loop(stop, sink):
    while not stop["v"]:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        try:
            sink.append(int(r.stdout.strip()))
        except ValueError:
            pass
        time.sleep(0.4)


def run(num_workers, batch=BATCH, ckpt=False):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model = DeJPEGNetS(grad_checkpoint=ckpt).to(DEV, memory_format=torch.channels_last).train()
    model_fwd = prepare_model_for_training(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    loader = make_dataloader(SHARDS, batch_size=batch, num_workers=num_workers, patch=PATCH, seed=42)
    it = iter(loader)

    # warmup (compile + fill prefetch queue)
    for j, t, cond in it:
        ctx = build_ctx(cond, torch.zeros(len(cond), 32)).to(DEV, non_blocking=True)
        j = j.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        t = t.to(DEV, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model_fwd(j, ctx); loss = (out.float() - t).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        break  # just one for warmup compile
    # real warmup of a few iters
    n_warm = 8
    for _ in range(n_warm):
        j, t, cond = next(it)
        ctx = build_ctx(cond, torch.zeros(len(cond), 32)).to(DEV, non_blocking=True)
        j = j.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        t = t.to(DEV, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model_fwd(j, ctx); loss = (out.float() - t).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()

    sink, stop = [], {"v": False}
    th = threading.Thread(target=gpu_util_loop, args=(stop, sink), daemon=True)
    th.start()
    t0 = time.time()
    for _ in range(ITERS):
        j, t, cond = next(it)
        ctx = build_ctx(cond, torch.zeros(len(cond), 32)).to(DEV, non_blocking=True)
        j = j.to(DEV, memory_format=torch.channels_last, non_blocking=True)
        t = t.to(DEV, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model_fwd(j, ctx); loss = (out.float() - t).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    stop["v"] = True; th.join()
    util = sum(sink) / max(1, len(sink))
    ms_it = 1000 * dt / ITERS
    print(f"workers={num_workers} batch={batch} ckpt={int(ckpt)}  GPU util {util:5.1f}%  "
          f"{ms_it:6.1f} ms/it  {ms_it/batch:5.1f} ms/sample")
    del model, model_fwd, opt, loader
    import gc; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"compiled student, patch {PATCH}, {ITERS} timed iters; does bigger batch push GPU util up?\n")
    run(6, batch=8)
    run(6, batch=16)
    run(6, batch=24, ckpt=True)
