"""QF-stratified batcher (spec §2.6).

10 per-QF-bin queues. Each effective batch (batch_size x accum_steps) gets a
guaranteed bin mix = 0.5 uniform-across-bins + 0.5 target-weighted (mid-emphasis,
spec §2.5). The mix is allocated over the ACCUMULATED batch as a unit, so a single
gradient step always contains low-QF (hard) samples -- defeating the collapse
mode where the model drifts to near-identity.

A naive per-micro-batch stratifier would pass the simple frequency test yet still
let an accumulated batch shed its low-QF slots; the accumulated-allocation test
catches exactly that.
"""
from __future__ import annotations

import collections
import heapq
import random
from typing import Any, Callable, Protocol

NUM_BINS = 10  # true_qf in [1,100]; bin = (qf-1)//10 clamped to [0,9]

# Spec §2.5 weighting: low(1-30)=0.2, mid(30-75)=0.6, high(75-100)=0.2.
# Spread across the 10 bins.
DEFAULT_WEIGHTED = [
    0.0667, 0.0667, 0.0667,   # bins 0-2 (1-30): low
    0.12, 0.12, 0.12, 0.12, 0.12,  # bins 3-7 (31-80): mid
    0.10, 0.10,               # bins 8-9 (81-100): high
]


def qf_bin(qf: int) -> int:
    return min(NUM_BINS - 1, max(0, (int(qf) - 1) // NUM_BINS))


class QBSource(Protocol):
    """A source that yields a degraded sample whose true_qf lands in a bin."""

    def draw(self, qf_low: int, qf_high: int, rng: random.Random) -> Any: ...

    def rebuild(self) -> None: ...


class QFBatcher:
    def __init__(
        self,
        source: QBSource,
        batch_size: int = 8,
        accum_steps: int = 2,
        weighted_dist: list[float] | None = None,
        num_bins: int = NUM_BINS,
        seed: int = 0,
    ):
        self.source = source
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.effective = batch_size * accum_steps
        self.num_bins = num_bins
        self.rng = random.Random(seed)
        self.target = self._make_target(weighted_dist)
        self.queues: list[collections.deque] = [collections.deque() for _ in range(num_bins)]
        # running fractional deficit per bin -> long-run average converges to target
        # exactly (avoids per-step quantization locking a bin to one slot count).
        self._deficit: list[float] = [0.0] * num_bins

    def _make_target(self, weighted_dist: list[float] | None) -> list[float]:
        w = weighted_dist or DEFAULT_WEIGHTED
        assert len(w) == self.num_bins
        u = [1.0 / self.num_bins] * self.num_bins
        t = [0.5 * u[b] + 0.5 * w[b] for b in range(self.num_bins)]
        s = sum(t)
        return [x / s for x in t]

    def _slot_counts(self) -> list[int]:
        """Allocate `effective` slots across bins.

        Every bin is guaranteed >= 1 slot per accumulated batch (no starvation ->
        every gradient step sees a low-QF hard sample). The remaining
        ``effective - num_bins`` "extra" slots are distributed by a per-bin
        fractional-deficit max-heap so the LONG-RUN average converges to `target`
        exactly (bins with larger target collect extras more often).
        """
        alloc = [1] * self.num_bins
        r = self.effective - self.num_bins
        if r < 0:
            # effective < num_bins: guarantee impossible; pure largest-deficit.
            r = self.effective
            alloc = [0] * self.num_bins
        extra = [self.target[b] * self.effective - 1.0 for b in range(self.num_bins)]
        for b in range(self.num_bins):
            self._deficit[b] += extra[b]
        heap = [(-self._deficit[b], b) for b in range(self.num_bins)]
        heapq.heapify(heap)
        for _ in range(max(0, r)):
            _, b = heapq.heappop(heap)
            alloc[b] += 1
            self._deficit[b] -= 1.0
            heapq.heappush(heap, (-self._deficit[b], b))
        return alloc

    def allocate_effective(self) -> list[int]:
        """One accumulated batch's worth of bin ids, shuffled."""
        counts = self._slot_counts()
        bins: list[int] = []
        for b, c in enumerate(counts):
            bins.extend([b] * c)
        self.rng.shuffle(bins)
        return bins

    def step(self) -> list[list[Any]]:
        """Return `accum_steps` micro-batches (list of samples each).

        The concatenation of all micro-batches is the stratified accumulated batch.
        """
        bins = self.allocate_effective()
        micros: list[list[Any]] = []
        for m in range(self.accum_steps):
            slot_bins = bins[m * self.batch_size : (m + 1) * self.batch_size]
            micros.append([self._draw_from_bin(b) for b in slot_bins])
        return micros

    def _draw_from_bin(self, b: int) -> Any:
        low = 1 + NUM_BINS * b
        high = 1 + NUM_BINS * (b + 1)
        return self.source.draw(low, high, self.rng)

    def rebuild(self) -> None:
        """Drop buffered queues and re-read source weights (e.g. new shards added).

        Ensures freshly-added shards are not oversampled from stale high-water marks.
        """
        self.queues = [collections.deque() for _ in range(self.num_bins)]
        self._deficit = [0.0] * self.num_bins
        if hasattr(self.source, "rebuild"):
            self.source.rebuild()


class _MicroOnlyBatcher:
    """Deliberately broken: stratifies each micro-batch independently.

    Used only to prove the accumulated-batch test catches the failure mode.
    """

    def __init__(self, source: QBSource, batch_size: int = 8, accum_steps: int = 2, seed: int = 0):
        self.source = source
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.rng = random.Random(seed)

    def step(self) -> list[list[Any]]:
        micros = []
        for _ in range(self.accum_steps):
            # each micro picks bins uniformly on its own -> accumulated can shed bins
            bins = [self.rng.randrange(NUM_BINS) for _ in range(self.batch_size)]
            micros.append([self._draw(b) for b in bins])
        return micros

    def _draw(self, b: int) -> Any:
        low = 1 + NUM_BINS * b
        high = 1 + NUM_BINS * (b + 1)
        return self.source.draw(low, high, self.rng)
