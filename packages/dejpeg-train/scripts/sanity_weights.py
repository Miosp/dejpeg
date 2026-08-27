"""Sanity: weighted source sampling produces the configured mix."""
import collections, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dejpeg_train.data.sources import DegradedBatchSource
from dejpeg_train.paths import shards_dir

SHARDS = [str(shards_dir() / f"{s}-*.tar")
          for s in ("div2k", "flickr2k", "liu4k_v2", "user_raws")]
W = {"user_raws": 0.35, "div2k": 0.25, "flickr2k": 0.20, "liu4k_v2": 0.20}

src = DegradedBatchSource(SHARDS, seed=1)
print("members per source:", {k: len(v) for k, v in sorted(src._src_members.items())})
rng = random.Random(0)
tally = collections.Counter()
N = 300
for _ in range(N):
    s = src.draw_dist(rng)
    tally[s["source"]] += 1
print("uniform-member mix (the H4 problem):")
for k in sorted(tally):
    print(f"  {k:<10} {100*tally[k]/N:5.1f}%")

src2 = DegradedBatchSource(SHARDS, seed=1, source_weights=W)
rng = random.Random(0)
tally2 = collections.Counter()
for _ in range(N):
    s = src2.draw_dist(rng)
    tally2[s["source"]] += 1
print("weighted mix (target 35/25/20/20):")
ok = True
for k, w in W.items():
    got = 100 * tally2.get(k, 0) / N
    ok &= abs(got - 100 * w) < 8
    print(f"  {k:<10} {got:5.1f}%  (target {100*w:.0f}%)")
print("PASS" if ok else "FAIL")
