"""Phase 0 Task 0.7 -- QF-stratified batcher tests.

Load-bearing: the accumulated-batch stratification test must (a) pass for the
correct allocator and (b) FAIL for a naive per-micro-batch allocator. If it passed
both ways the entire anti-collapse design would be untested.
"""
from __future__ import annotations

import collections

import pytest

from dejpeg.data.batcher import (
    DEFAULT_WEIGHTED,
    NUM_BINS,
    QFBatcher,
    _MicroOnlyBatcher,
    qf_bin,
)


class MockSample:
    def __init__(self, qf: int):
        self.true_qf = qf


class MockSource:
    def __init__(self):
        self.draws: list[int] = []

    def draw(self, qf_low, qf_high, rng):
        qf = (qf_low + qf_high) // 2
        self.draws.append(qf_bin(qf))
        return MockSample(qf)

    def rebuild(self):
        self.draws.clear()


class RebuildSource:
    def __init__(self):
        self.version = 0
        self.by_version: collections.Counter = collections.Counter()

    def draw(self, qf_low, qf_high, rng):
        self.by_version[self.version] += 1
        return MockSample((qf_low + qf_high) // 2)

    def rebuild(self):
        self.version += 1


# ----------------------------------------------------------------- allocation


def test_qf_bin_boundaries():
    assert qf_bin(1) == 0
    assert qf_bin(10) == 0
    assert qf_bin(11) == 1
    assert qf_bin(100) == 9
    assert qf_bin(0) == 0   # clamp
    assert qf_bin(150) == 9  # clamp


def test_effective_batch_size_matches_batch_x_accum():
    b = QFBatcher(MockSource(), batch_size=8, accum_steps=2)
    assert b.effective == 16
    counts = b._slot_counts()
    assert sum(counts) == 16


# ----------------------------------------------------- accumulated stratification


def _run(batcher, steps: int = 200):
    all_qf: list[int] = []
    per_accum_missing_bin0 = 0
    for _ in range(steps):
        micros = batcher.step()
        flat = [s.true_qf for micro in micros for s in micro]
        all_qf.extend(flat)
        if not any(qf_bin(q) == 0 for q in flat):
            per_accum_missing_bin0 += 1
    return all_qf, per_accum_missing_bin0


def test_correct_batcher_stratifies_accumulated_within_tolerance():
    src = MockSource()
    b = QFBatcher(src, batch_size=8, accum_steps=2, seed=0)
    all_qf, missing = _run(b, steps=200)
    n = len(all_qf)
    freq = [0] * NUM_BINS
    for q in all_qf:
        freq[qf_bin(q)] += 1
    for bin_i in range(NUM_BINS):
        f = freq[bin_i] / n
        assert abs(f - b.target[bin_i]) < 0.03, f"bin {bin_i} freq {f:.3f} vs target {b.target[bin_i]:.3f}"
    assert missing == 0, "correct allocator must never shed bin0 from an accumulated batch"


def test_micro_only_allocator_sheds_bins_from_accumulated_batch():
    """Negative control: a per-micro stratifier must fail the accumulated check.

    If this stops failing, the accumulated test no longer guards the collapse mode.
    """
    src = MockSource()
    broken = _MicroOnlyBatcher(src, batch_size=8, accum_steps=2, seed=0)
    _, missing = _run(broken, steps=200)
    # P(bin0 absent in 16 draws) = 0.9**16 ~= 0.185 -> ~37/200 expected
    assert missing > 10, f"micro-only allocator unexpectedly kept bin0 (missing={missing})"


# ------------------------------------------------------------- mid-emphasis


def test_target_distribution_is_mid_emphasized():
    b = QFBatcher(MockSource(), seed=0)
    mid_mass = sum(b.target[3:8])
    high_mass = sum(b.target[8:10])
    low_mass = sum(b.target[0:3])
    assert mid_mass > high_mass and mid_mass > low_mass


# --------------------------------------------------------------- rebuild path


def test_rebuild_clears_queues_and_calls_source():
    src = RebuildSource()
    b = QFBatcher(src, batch_size=8, accum_steps=2, seed=0)
    # pre-rebuild draws land in version 0
    for _ in range(3):
        b.step()
    assert src.by_version[0] > 0
    assert all(len(q) == 0 for q in b.queues)
    b.rebuild()
    assert src.version == 1
    assert all(len(q) == 0 for q in b.queues)
    # post-rebuild draws land in version 1 (no stale oversample from version 0)
    b.step()
    assert src.by_version[1] > 0
    # version 0 count unchanged after rebuild -> no stale attribution
    pre = src.by_version[0]
    b.step()
    assert src.by_version[0] == pre


def test_default_weighted_sums_to_one():
    assert abs(sum(DEFAULT_WEIGHTED) - 1.0) < 1e-3
