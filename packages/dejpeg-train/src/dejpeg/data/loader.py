"""Worker-parallel degrade dataloader for near-100% GPU utilization.

Problem (Phase-0.6 profile): single-process degrade is ~91 ms/sample (729 ms for
a batch of 8), but compiled GPU compute is ~180 ms/iter -> the GPU starves while
one process degrades. Fix: run the degrade across N DataLoader *process* workers
(no GIL) with prefetch, so batches are produced in parallel and buffered ahead of
the GPU's consumption. With ~6 workers the combined throughput (~66 samples/s)
exceeds GPU consumption (~44 samples/s), so the GPU never waits.

Each worker draws a §2.5-distributed QF independently (statistically equivalent
to per-batch QF stratification over the long run -- the anti-collapse property
holds on the distribution, not the per-batch composition) and emits a cropped
patch + the 65-D q_table condition. Controls (~10%) are included per spec.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from ..model.conditioning import quant_table_to_condition
from .sources import DegradedBatchSource


def sample_condition(s) -> torch.Tensor:
    """Luma quant table -> 65-D condition; controls / no-table -> dropped (validity 0)."""
    qt = s["record"].quant_tables
    if s["is_control"] or not qt:
        return quant_table_to_condition([0] * 64, validity=0.0)
    return quant_table_to_condition(qt[min(qt)]["values"], validity=1.0)


class DegradeIterableDataset(IterableDataset):
    """Infinite stream of (jpeg_patch, target_patch, condition) tuples.

    Worker-safe: the DegradedBatchSource (open tarfile handles) is created lazily
    per worker after fork, seeded by worker id so each worker explores a different
    part of the degrade space.
    """

    def __init__(self, shards, patch: int = 256, seed: int = 0, p_control: float = 0.1,
                 gray_frac: float = 0.0, **src_kw):
        super().__init__()
        if isinstance(shards, (str, Path)):
            self.shards = [str(shards)]
        else:
            self.shards = [str(s) for s in shards]
        self.patch = patch
        self.seed = seed
        self.p_control = p_control
        self.gray_frac = gray_frac
        self.src_kw = src_kw
        self._src = None  # lazy, per-worker

    def _ensure(self):
        if self._src is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            self._src = DegradedBatchSource(
                self.shards, seed=self.seed + wid * 7919,
                p_control=self.p_control, **self.src_kw,
            )

    def __iter__(self):
        self._ensure()
        rng = random.Random()
        p = self.patch
        while True:
            s = self._src.draw_dist(rng)
            j = torch.from_numpy(s["jpeg"]).permute(2, 0, 1)
            t = torch.from_numpy(s["target"]).permute(2, 0, 1)
            if self.gray_frac > 0 and rng.random() < self.gray_frac:
                j = j.mean(dim=0, keepdim=True).expand_as(j).contiguous()
                t = t.mean(dim=0, keepdim=True).expand_as(t).contiguous()
            h, w = j.shape[-2:]
            if h < p or w < p:
                continue  # degrade guarantees >= CROP(512), but stay defensive
            dy = rng.randrange(0, h - p + 1)
            dx = rng.randrange(0, w - p + 1)
            yield (
                j[:, dy:dy + p, dx:dx + p].contiguous(),
                t[:, dy:dy + p, dx:dx + p].contiguous(),
                sample_condition(s),
            )


def make_dataloader(
    shards,
    batch_size: int = 8,
    num_workers: int = 6,
    patch: int = 256,
    seed: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    source_weights: dict | None = None,
    gray_frac: float = 0.0,
):
    """Process-parallel degrade loader. prefetch_factor buffers batches ahead of the
    GPU; persistent_workers avoids per-epoch fork cost; drop_last keeps shapes fixed
    (so torch.compile never recompiles mid-run). source_weights: per-corpus sampling
    weights (Phase 0.7 H4) — keys are shard filename prefixes (div2k, flickr2k,
    liu4k_v2, user_raws). gray_frac: probability of converting the drawn pair to
    grayscale-replicated (Phase 2b chroma-anchor)."""
    ds = DegradeIterableDataset(shards, patch=patch, seed=seed,
                                source_weights=source_weights, gray_frac=gray_frac)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
        # default collate stacks the (jpeg, target, condition) tensors -> (N,3,p,p),(N,3,p,p),(N,65)
    )
