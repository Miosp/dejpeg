"""Quick per-iter timing probe for the FFN student. 12 iters, report ms/iter."""
import random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg.data.sources import DegradedBatchSource
from dejpeg.model.conditioning import record_to_condition
from dejpeg.model.student import DeJPEGNetS, build_ctx
from dejpeg.paths import shards_dir

src = DegradedBatchSource(str(shards_dir() / "div2k-*.tar"), seed=42)
rng = random.Random(0)
pairs = []
while len(pairs) < 8:
    s = src.draw(15, 35, rng)
    if not s["is_control"]:
        pairs.append(s)
chw = lambda a: torch.as_tensor(a).permute(2,0,1).contiguous()
j = torch.stack([chw(p["jpeg"]) for p in pairs])[...,:256,:256].contiguous().to("cuda", memory_format=torch.channels_last)
t = torch.stack([chw(p["target"]) for p in pairs])[...,:256,:256].contiguous().to("cuda")
q = torch.stack([record_to_condition(p["record"], dropout_p=0.0) for p in pairs])
c = build_ctx(q, torch.zeros(8,32)).to("cuda")
model = DeJPEGNetS().to("cuda", memory_format=torch.channels_last).train()
print(f"params={sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9,0.9))
# warmup (compile/cache)
for _ in range(3):
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        out = model(j, c); loss = (out-t).abs().mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()
# timed
N = 10; t0 = time.time()
for it in range(N):
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        out = model(j, c); loss = (out-t).abs().mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"{N} iters in {dt:.2f}s -> {1000*dt/N:.1f} ms/iter (loss {loss.item():.5f})")
