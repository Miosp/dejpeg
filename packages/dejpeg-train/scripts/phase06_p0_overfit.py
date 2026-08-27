"""Phase 0.6 P0 -- overfit sanity.

Train the student on just 8 fixed jpeg/target pairs (QF~20), L1 only, high LR,
many iterations. If the model + training loop are correct, L1 must collapse to
near zero (the model has ample capacity to memorize 8 patches). If it plateaus
high, something is broken (grad flow / normalization / loss) and we stop here.

Output: prints the loss curve and a clear PASS/FAIL verdict.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dejpeg_train.data.sources import DegradedBatchSource
from dejpeg_train.model.conditioning import record_to_condition
from dejpeg_train.model.student import DeJPEGNetS, build_ctx
from dejpeg_train.train.schedule import bf16_autocast, set_seed
from dejpeg_train.paths import shards_dir

SHARDS = str(shards_dir() / "div2k-*.tar")
N_PAIRS = 8
ITERS = 1000
LR = 5e-3
OVERFIT_THRESHOLD = 0.004  # L1 below this on 8 memorized patches = healthy
USE_FP32 = True  # diagnostic: remove bf16 precision as a factor


def main() -> None:
    set_seed(0)
    dev = "cuda"
    assert torch.cuda.is_available(), "cuda required for P0"

    src = DegradedBatchSource(SHARDS, seed=42)
    rng = random.Random(0)
    pairs = []
    while len(pairs) < N_PAIRS:
        s = src.draw(15, 35, rng)  # QF band around 20
        if not s["is_control"]:
            pairs.append(s)
    print(f"[p0] drew {len(pairs)} real pairs (true_qf="
          f"{[p['true_qf'] for p in pairs]})")

    def to_chw(a):
        return torch.as_tensor(a).permute(2, 0, 1).contiguous()  # H,W,C -> C,H,W

    jpeg = torch.stack([to_chw(p["jpeg"]) for p in pairs])[..., :256, :256].contiguous()
    target = torch.stack([to_chw(p["target"]) for p in pairs])[..., :256, :256].contiguous()
    q = torch.stack([record_to_condition(p["record"], dropout_p=0.0) for p in pairs])
    deg = torch.zeros(N_PAIRS, 32)
    ctx = build_ctx(q, deg)

    model = DeJPEGNetS().to(dev, memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    jpeg = jpeg.to(dev, memory_format=torch.channels_last)
    target = target.to(dev)
    ctx = ctx.to(dev)

    first = None
    amctx = (bf16_autocast(False, device_type="cuda") if not USE_FP32 else
             torch.cuda.amp.autocast(enabled=False))
    for it in range(ITERS):
        with amctx:
            out = model(jpeg, ctx)
            loss = (out - target).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        if it % 100 == 0 or it == ITERS - 1:
            print(f"[p0] iter {it:4d}  loss {loss.item():.6f}")
        if loss.item() < 1e-4 and it > 100:
            print(f"[p0] converged early at iter {it}")
            break

    final = loss.item()
    print(f"\n[p0] first={first:.6f}  final={final:.6f}  reduction={100*(1-final/first):.1f}%")
    if final < OVERFIT_THRESHOLD:
        print(f"[p0] VERDICT: PASS (L1 < {OVERFIT_THRESHOLD} on {N_PAIRS} patches)")
    else:
        print(f"[p0] VERDICT: FAIL (L1 plateaued at {final:.6f} > {OVERFIT_THRESHOLD}) -- investigate")


if __name__ == "__main__":
    main()
