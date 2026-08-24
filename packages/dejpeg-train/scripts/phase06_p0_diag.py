"""Phase 0.6 P0 diagnostic #4 -- N-scaling overfit.

If memorizing ONE image collapses fast but 8 is slow, the architecture is healthy
and the multi-image slowness is just the volume of natural-image HF detail a conv
net must fit (expected, not a defect). L1, lr1e-3, 600 iters, N in {1,2,4}.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg.data.sources import DegradedBatchSource
from dejpeg.model.conditioning import record_to_condition
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import shards_dir

SHARDS = str(shards_dir() / "div2k-*.tar")


def draw_n(n):
    src = DegradedBatchSource(SHARDS, seed=42)
    rng = random.Random(0)
    pairs = []
    while len(pairs) < n:
        s = src.draw(15, 35, rng)
        if not s["is_control"]:
            pairs.append(s)
    chw = lambda a: torch.as_tensor(a).permute(2, 0, 1).contiguous()
    j = torch.stack([chw(p["jpeg"]) for p in pairs])[..., :256, :256].contiguous()
    t = torch.stack([chw(p["target"]) for p in pairs])[..., :256, :256].contiguous()
    q = torch.stack([record_to_condition(p["record"], dropout_p=0.0) for p in pairs])
    return (j.to("cuda", memory_format=torch.channels_last),
            t.to("cuda"),
            build_ctx(q, torch.zeros(n, 32)).to("cuda"))


def run(n, iters=600):
    j, t, c = draw_n(n)
    torch.manual_seed(0)
    m = DeJPEGNetS().to("cuda", memory_format=torch.channels_last).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.9))
    first = None
    for it in range(iters):
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = m(j, c); loss = (out - t).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
        if it % 100 == 0 or it == iters - 1:
            print(f"  N={n} it={it:4d} loss={loss.item():.6f}")
    print(f"  N={n} DONE: {first:.6f} -> {loss.item():.6f} ({100*(1-loss.item()/first):.0f}% down)\n")


def main():
    print("FFN block, L1, AdamW lr1e-3, bf16, 600 iters\n")
    run(1)
    run(2)
    run(4)


if __name__ == "__main__":
    main()
