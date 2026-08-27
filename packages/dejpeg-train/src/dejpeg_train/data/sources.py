"""Batch sources: clean patches -> degrade -> QF-stratified draw (QBSource protocol).

  * DegradedBatchSource -- reads webdataset .tar shards from prepare.py, random
    access via kept-open tarfile member index.
  * SyntheticSource     -- generates non-photographic patches on the fly (probes
    and the pre-corpus Phase-0 exit gate).

Both return uniform 512x512 float32 [0,1] samples (jpeg + target) plus the
DegradationRecord. ``draw`` receives a python ``random.Random`` (the batcher's),
used for selection; degradation randomness is driven by the sampler's own RNG.
"""
from __future__ import annotations

import glob
import random
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .controls import ControlConfig, ControlSampler
from .degrade import DegradationConfig, DegradationSampler
from .synthetic import SyntheticGenerator

CROP = 512  # training patch size (== DegradationConfig.crop_size)


class _BaseSource:
    def __init__(
        self,
        degrade_config: DegradationConfig | None = None,
        control_config: ControlConfig | None = None,
        p_control: float = 0.1,
        seed: int = 0,
    ):
        self.degrade = DegradationSampler(degrade_config, seed=seed)
        self.controls = ControlSampler(control_config, seed=seed + 1)
        self.p_control = p_control

    def _pick_clean(self, rng: random.Random) -> np.ndarray:
        raise NotImplementedError

    def _draw_control(self, clean_hwc: np.ndarray, rng: random.Random):
        h, w = clean_hwc.shape[:2]
        size = min(CROP, h, w)
        dy = rng.randint(0, max(1, h - size + 1))
        dx = rng.randint(0, max(1, w - size + 1))
        patch = clean_hwc[dy : dy + size, dx : dx + size]
        return self.controls.sample(patch)

    def draw(self, qf_low: int, qf_high: int, rng: random.Random):
        clean_hwc = self._pick_clean(rng)
        if rng.random() < self.p_control:
            arr, target, record = self._draw_control(clean_hwc, rng)
        else:
            arr, target, record = self.degrade.sample(clean_hwc, qf_range=(qf_low, qf_high))
        return {
            "jpeg": arr,
            "target": target,
            "record": record,
            "true_qf": record.true_qf,
            "is_control": record.is_control,
            "source": getattr(self, "_last_source", "?"),
        }

    def draw_dist(self, rng: random.Random):
        """§2.5-distribution draw (sampler uses its built-in QF distribution; no range).
        Used by worker-based dataloaders where per-sample QF is sampled independently --
        statistically equivalent to the QFBatcher's stratification over enough samples."""
        clean_hwc = self._pick_clean(rng)
        if rng.random() < self.p_control:
            arr, target, record = self._draw_control(clean_hwc, rng)
        else:
            arr, target, record = self.degrade.sample(clean_hwc, qf_range=None)
        return {
            "jpeg": arr,
            "target": target,
            "record": record,
            "true_qf": record.true_qf,
            "is_control": record.is_control,
            "source": getattr(self, "_last_source", "?"),
        }

    def rebuild(self) -> None:
        pass


class SyntheticSource(_BaseSource):
    def __init__(self, size: int = 640, allow_html: bool = False, **kw):
        super().__init__(**kw)
        self.gen = SyntheticGenerator(size=size, allow_html=allow_html)


def _pick_clean(self, rng: random.Random) -> np.ndarray:
    seed = rng.randint(0, 2**31 - 1)
    return self.gen.generate(seed)


SyntheticSource._pick_clean = _pick_clean


class DegradedBatchSource(_BaseSource):
    def __init__(self, shard_glob, source_weights: dict | None = None, **kw):
        super().__init__(**kw)
        # accept a single glob string or a list of globs (combined into one pool)
        self.shard_globs = [shard_glob] if isinstance(shard_glob, str) else list(shard_glob)
        # per-source sampling weights keyed by shard filename prefix (text before
        # first '-'); e.g. {"user_raws": 0.35, "div2k": 0.25, ...}. Default (None)
        # is uniform over SOURCES (balanced corpora), NOT over members.
        self.source_weights = source_weights
        self._tars: list[tarfile.TarFile] = []
        self._members: list[tuple[tarfile.TarFile, str]] = []
        self._src_members: dict[str, list] = {}
        self._last_source: str = "?"
        self._index()

    @property
    def shard_glob(self) -> str:
        return self.shard_globs[0] if len(self.shard_globs) == 1 else str(self.shard_globs)

    def _index(self) -> None:
        for t in self._tars:
            try:
                t.close()
            except Exception:
                pass
        self._tars = []
        self._members = []
        self._src_members = {}
        paths = []
        for g in self.shard_globs:
            paths.extend(glob.glob(g))
        for p in sorted(set(paths)):
            src = Path(p).name.split("-")[0]
            t = tarfile.open(p, "r")
            self._tars.append(t)
            bucket = self._src_members.setdefault(src, [])
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".webp"):
                    entry = (t, m.name)
                    self._members.append(entry)
                    bucket.append(entry)
        # resolve weights over the sources actually present
        if self.source_weights:
            tot = sum(w for s, w in self.source_weights.items() if self._src_members.get(s))
            if tot <= 0:
                raise RuntimeError("source_weights selects no present sources")
            self._cum = []
            acc = 0.0
            for s, w in self.source_weights.items():
                if self._src_members.get(s) and w > 0:
                    acc += w / tot
                    self._cum.append((acc, s))
        else:
            n = len(self._src_members)
            self._cum = [((i + 1) / n, s) for i, s in enumerate(sorted(self._src_members))]

    def _pick_clean(self, rng: random.Random) -> np.ndarray:
        if not self._members:
            raise RuntimeError(f"no shards match {self.shard_globs}")
        if len(self._cum) == 1:
            src = self._cum[0][1]
        else:
            r = rng.random()
            src = self._cum[-1][1]
            for cut, s in self._cum:
                if r < cut:
                    src = s
                    break
        bucket = self._src_members[src]
        self._last_source = src
        tar, name = bucket[rng.randrange(len(bucket))]
        f = tar.extractfile(name)
        return np.array(Image.open(BytesIO(f.read())).convert("RGB"))

    def rebuild(self) -> None:
        self._index()

    def __len__(self) -> int:
        return len(self._members)
