"""Phase 0 Task 0.10 -- experiment infra tests.

Load-bearing: checkpoint save/reload resumes to a bit-identical state (model +
optimizer + EMA + RNG reproduces the same draws afterwards).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from dejpeg_train.bench.latency import count_params, profile
from dejpeg_train.eval.panel import contact_sheet
from dejpeg_train.train.schedule import (
    EMA,
    bf16_autocast,
    capture_rng,
    config_hash,
    cosine_lr,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    set_seed,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Linear(8, 8))

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------- seeding


def test_set_seed_is_deterministic():
    set_seed(0)
    a = torch.randn(3, 3)
    set_seed(0)
    b = torch.randn(3, 3)
    assert torch.equal(a, b)
    set_seed(1)
    c = torch.randn(3, 3)
    assert not torch.equal(a, c)


# ---------------------------------------------------------------- cosine LR


def test_cosine_lr_endpoints():
    f = cosine_lr(base_lr=1e-3, total_steps=1000, warmup=100)
    assert abs(f(100) - 1e-3) < 1e-9        # end of warmup == base
    assert f(999) < 1e-5                      # near end ~ 0
    assert f(0) < 1e-3                        # warmup ramp starts low


def test_cosine_lr_no_warmup():
    f = cosine_lr(base_lr=2e-4, total_steps=500, warmup=0)
    assert abs(f(0) - 2e-4) < 1e-9            # no warmup: starts at base


# -------------------------------------------------------------------- EMA


def test_ema_update_and_swap():
    set_seed(0)
    m = Tiny()
    ema = EMA(m, decay=0.5)
    # perturb parameters
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)  # shadow = 0.5*orig + 0.5*(orig+1) = orig + 0.5
    orig = ema.state_dict()
    set_seed(0)
    ref = Tiny().state_dict()
    for n in orig:
        assert torch.allclose(orig[n], ref[n] + 0.5, atol=1e-6)
    # swap loads shadow into model
    set_seed(0)
    m2 = Tiny()
    with ema.swap(m2):
        swapped = {n: p.detach().clone() for n, p in m2.named_parameters()}
    for n in orig:
        assert torch.allclose(swapped[n], orig[n], atol=1e-6)


# --------------------------------------------------------------- autocast


def test_bf16_autocast_runs_on_cpu():
    m = Tiny().eval()
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        with bf16_autocast(enabled=True, device_type="cpu"):
            out = m(x)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------- config hash


def test_config_hash_stable_and_distinct():
    a = {"lr": 1e-3, "layers": [2, 2, 4], "name": "x"}
    b = {"name": "x", "layers": [2, 2, 4], "lr": 1e-3}  # reordered
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"lr": 1e-4, "layers": [2, 2, 4], "name": "x"})


# ----------------------------------------------- checkpoint bit-identical resume


def test_checkpoint_roundtrip_is_bit_identical(tmp_path):
    set_seed(42)
    model = Tiny()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema = EMA(model)
    # one training-ish step to populate optimizer state (forward under grad)
    loss = model(torch.randn(1, 3, 8, 8)).sum()
    loss.backward()
    opt.step()
    ema.update(model)

    # capture RNG *before* a reference draw, then save
    rng_state = capture_rng()
    ref_draw = torch.rand(5)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        ema=ema,
        optimizer=opt,
        step=7,
        manifest_hash="deadbeef",
        config={"lr": 1e-3, "phase": 0},
        rng_states=rng_state,
    )

    # clobber everything
    set_seed(0)
    model2 = Tiny()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ema2 = EMA(model2)

    ckpt = load_checkpoint(ckpt_path)
    model2.load_state_dict(ckpt["model"])
    opt2.load_state_dict(ckpt["optimizer"])
    ema2.load_state_dict(ckpt["ema"])
    restore_rng(ckpt["rng"])

    # RNG reproduces the same draws after resume
    got_draw = torch.rand(5)
    assert torch.equal(got_draw, ref_draw), "RNG did not resume bit-identically"

    # model + optimizer + EMA states equal
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"param {n1} differs after resume"
    # optimizer state equal (compare via serialized state_dict, int-keyed)
    osd, osd2 = opt.state_dict(), opt2.state_dict()
    assert osd["param_groups"] == osd2["param_groups"]
    for k in osd["state"]:
        for kk in osd["state"][k]:
            a, b = osd["state"][k][kk], osd2["state"][k][kk]
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), f"opt state {k}.{kk} differs"
            else:
                assert a == b, f"opt state {k}.{kk} differs"
    for n in ema.shadow:
        assert torch.equal(ema.shadow[n], ema2.shadow[n]), f"ema {n} differs"

    assert ckpt["step"] == 7
    assert ckpt["manifest_hash"] == "deadbeef"
    assert ckpt["config_hash"] == config_hash({"lr": 1e-3, "phase": 0})


# --------------------------------------------------------------- panel + bench


def test_contact_sheet_writes_png(tmp_path):
    imgs = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(16)]
    out_path = tmp_path / "panel.png"
    arr = contact_sheet(imgs, cols=4, thumb=160, path=out_path)
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert out_path.exists() and out_path.stat().st_size > 0


def test_latency_profile_reports_all_four():
    from dejpeg_train.model.student import DeJPEGNetS

    model = DeJPEGNetS()
    tile = torch.randn(1, 3, 64, 64)
    ctx = torch.randn(1, 97)
    res = profile(model, args=(tile, ctx), device="cpu", warmup=1, reps=2)
    assert res["params"] == count_params(model) > 0
    assert res["flops"] > 0
    assert res["activations"] > 0
    assert np.isfinite(res["runtime_ms"])
