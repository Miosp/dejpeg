"""Profile one training step -- breakdown of where the ~1.2s/iter goes.
Confirms the bottleneck before optimizing."""
import random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from dejpeg_train.data.batcher import QFBatcher
from dejpeg_train.data.sources import DegradedBatchSource
from dejpeg_train.loss.perceptual import PerceptualLoss
from dejpeg_train.model.conditioning import quant_table_to_condition
from dejpeg_train.model.student import DeJPEGNetS, build_ctx
from dejpeg_train.paths import shards_dir

SHARDS = [str(shards_dir() / "div2k-*.tar"),
          str(shards_dir() / "flickr2k-*.tar")]
PATCH = 256

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
src = DegradedBatchSource(SHARDS, seed=42)
batcher = QFBatcher(src, batch_size=8, accum_steps=1, seed=0)
model = DeJPEGNetS().to(dev, memory_format=torch.channels_last).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
percept = PerceptualLoss(net="vgg", crop=128).to(dev).eval()
rng = random.Random()


def sample_condition(s):
    qt = s["record"].quant_tables
    if s["is_control"] or not qt:
        return quant_table_to_condition([0] * 64, validity=0.0)
    return quant_table_to_condition(qt[min(qt)]["values"], validity=1.0)


def build_batch_cpu():
    samples = [s for micro in batcher.step() for s in micro]
    j_parts, t_parts, conds = [], [], []
    for s in samples:
        j = torch.as_tensor(s["jpeg"]).permute(2, 0, 1)
        t = torch.as_tensor(s["target"]).permute(2, 0, 1)
        h, w = j.shape[-2:]
        dy, dx = rng.randrange(0, h-PATCH+1), rng.randrange(0, w-PATCH+1)
        j_parts.append(j[:, dy:dy+PATCH, dx:dx+PATCH])
        t_parts.append(t[:, dy:dy+PATCH, dx:dx+PATCH])
        conds.append(sample_condition(s))
    return (torch.stack(j_parts), torch.stack(t_parts),
            build_ctx(torch.stack(conds), torch.zeros(len(conds), 32)))


# warmup (compile caches)
for _ in range(3):
    j, t, c = build_batch_cpu()
    j = j.to(dev, memory_format=torch.channels_last); t = t.to(dev); c = c.to(dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(j, c); loss = (out.float()-t).abs().mean() + percept(out.float(), t)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()

N = 15
t_degrade = t_h2d = t_fwd = t_lpips = t_back = t_opt = 0.0
for _ in range(N):
    torch.cuda.synchronize(); s0 = time.time()
    jc, tc, cc = build_batch_cpu()
    torch.cuda.synchronize(); s1 = time.time()
    j = jc.to(dev, memory_format=torch.channels_last); t = tc.to(dev); c = cc.to(dev)
    torch.cuda.synchronize(); s2 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(j, c)
    torch.cuda.synchronize(); s3 = time.time()
    l1 = (out.float()-t).abs().mean()
    lp = percept(out.float(), t)
    loss = l1 + lp
    torch.cuda.synchronize(); s4 = time.time()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.cuda.synchronize(); s5 = time.time()
    opt.step()
    torch.cuda.synchronize(); s6 = time.time()
    t_degrade += s1-s0; t_h2d += s2-s1; t_fwd += s3-s2; t_lpips += s4-s3; t_back += s5-s4; t_opt += s6-s5

tot = t_degrade+t_h2d+t_fwd+t_lpips+t_back+t_opt
print(f"avg over {N} iters, total {1000*tot/N:.0f}ms/iter:")
for name, val in [("degrade(cpu)", t_degrade), ("h2d", t_h2d), ("forward", t_fwd),
                  ("lpips", t_lpips), ("backward", t_back), ("opt", t_opt)]:
    print(f"  {name:14s} {1000*val/N:6.1f}ms  ({100*val/tot:4.1f}%)")
