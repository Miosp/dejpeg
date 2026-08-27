"""Phase 0.6 -- size x optimizer sweep.

User scope: <10MB FP16 is "best", <50MB OK. So sizes C0 in {30, 48, 64}
(~1.8M/5M/9M params -> 3.6/10/18 MB FP16) are all in budget. This sweeps those
sizes against AdamW (baseline), Lion (lr/10, wd*10), and Schedule-Free AdamW on
the N=8 overfit task, measuring params/MB, peak VRAM, ms/iter, final L1, and
iters-to-threshold. Prints one results table.

Run: uv run --with lion-pytorch --with schedule-free python -u scripts/phase06_optim_size_sweep.py
"""
import random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg_train.data.sources import DegradedBatchSource
from dejpeg_train.model.conditioning import record_to_condition
from dejpeg_train.model.student import DeJPEGNetS, build_ctx
from dejpeg_train.paths import shards_dir

SHARDS = str(shards_dir() / "div2k-*.tar")
N = 8
ITERS = 300
THRESH = 0.005
LR = {"adamw": 1e-3, "lion": 1e-4, "sf": 5e-4}
WD = {"adamw": 1e-3, "lion": 1e-2, "sf": 1e-3}


def draw_pairs():
    src = DegradedBatchSource(SHARDS, seed=42)
    rng = random.Random(0)
    pairs = []
    while len(pairs) < N:
        s = src.draw(15, 35, rng)
        if not s["is_control"]:
            pairs.append(s)
    chw = lambda a: torch.as_tensor(a).permute(2, 0, 1).contiguous()
    j = torch.stack([chw(p["jpeg"]) for p in pairs])[..., :256, :256].contiguous()
    t = torch.stack([chw(p["target"]) for p in pairs])[..., :256, :256].contiguous()
    q = torch.stack([record_to_condition(p["record"], dropout_p=0.0) for p in pairs])
    return j, t, build_ctx(q, torch.zeros(N, 32))


def make_opt(name, params):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=LR[name], weight_decay=WD[name], betas=(0.9, 0.9))
    if name == "lion":
        from lion_pytorch import Lion
        return Lion(params, lr=LR[name], weight_decay=WD[name], betas=(0.9, 0.95))
    if name == "sf":
        from schedulefree import AdamWScheduleFree
        opt = AdamWScheduleFree(params, lr=LR[name], weight_decay=WD[name])
        opt.train()  # schedule-free weight-averaging
        return opt


def run(c0, optim, j_cpu, t_cpu, c_cpu):
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    m = DeJPEGNetS(c0=c0).to("cuda", memory_format=torch.channels_last).train()
    n_params = sum(p.numel() for p in m.parameters())
    fp16_mb = n_params * 2 / 1e6
    opt = make_opt(optim, m.parameters())
    j = j_cpu.to("cuda", memory_format=torch.channels_last)
    t = t_cpu.to("cuda")
    c = c_cpu.to("cuda")
    # warmup 3
    for _ in range(3):
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = m(j, c); loss = (out - t).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    crossed = None
    t0 = time.time()
    last = None
    for it in range(ITERS):
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = m(j, c); loss = (out - t).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        last = loss.item()
        if last < THRESH and crossed is None:
            crossed = it
    torch.cuda.synchronize()
    ms_iter = 1000 * (time.time() - t0) / ITERS
    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    del m, opt, j, t, c, out, loss
    torch.cuda.empty_cache()
    return dict(c0=c0, optim=optim, params=n_params, fp16_mb=fp16_mb,
                vram_mb=vram_mb, ms_iter=ms_iter, final_l1=last, crossed=crossed)


def main():
    j, t, c = draw_pairs()
    print(f"N={N} overfit, {ITERS} iters, bf16, grad-clip 1.0, thresh={THRESH}")
    print(f"{'c0':>3} {'optim':>7} {'params':>9} {'FP16MB':>7} {'VRAMmb':>7} "
          f"{'ms/it':>6} {'finalL1':>8} {'cross@':>7}")
    rows = []
    for c0 in (30, 48, 64):
        for optim in ("adamw", "lion", "sf"):
            try:
                r = run(c0, optim, j, t, c)
                rows.append(r)
                cr = "-" if r["crossed"] is None else r["crossed"]
                print(f"{r['c0']:>3} {r['optim']:>7} {r['params']:>9,} {r['fp16_mb']:>7.2f} "
                      f"{r['vram_mb']:>7.0f} {r['ms_iter']:>6.1f} {r['final_l1']:>8.5f} {str(cr):>7}")
            except Exception as e:
                print(f"{c0:>3} {optim:>7} ERROR: {e!r}")
    print("\n=== ranked by final L1 ===")
    for r in sorted(rows, key=lambda x: x["final_l1"]):
        print(f"  c0={r['c0']:<3} {r['optim']:<6} finalL1={r['final_l1']:.5f} "
              f"crossed@{r['crossed']} ({r['fp16_mb']:.1f}MB FP16, {r['ms_iter']:.0f}ms/it)")


if __name__ == "__main__":
    main()
