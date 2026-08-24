"""Latency / complexity profiling (spec §5, bench/ protocol).

Report all four together per the NTIRE protocol: params, FLOPs, activations,
runtime. Activations correlate with GPU runtime better than params do, so they
are reported separately. FLOPs/activations counted via forward hooks (Conv2d +
Linear MACs); runtime is a warm-up + timed mean over device.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class _Probe:
    def __init__(self):
        self.macs = 0
        self.activations = 0
        self._handles = []

    def attach(self, model: nn.Module) -> None:
        for m in model.modules():
            self._handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module: nn.Module, inp, out):
        # MACs
        if isinstance(module, nn.Conv2d):
            macs = _conv_macs(module, inp, out)
            self.macs += macs
        elif isinstance(module, nn.Linear):
            macs = _linear_macs(module, inp, out)
            self.macs += macs
        # activations (output element count)
        self.activations += _numel(out)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _conv_macs(mod: nn.Conv2d, inp, out) -> int:
    if not isinstance(out, torch.Tensor):
        return 0
    n, _, h, w = out.shape
    groups = mod.groups
    return n * mod.out_channels * h * w * (mod.in_channels // groups) * mod.kernel_size[0] * mod.kernel_size[1]


def _linear_macs(mod: nn.Linear, inp, out) -> int:
    if not isinstance(out, torch.Tensor):
        return 0
    return int(out.numel()) * mod.in_features


def _numel(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.numel())
    if isinstance(x, (tuple, list)):
        return sum(_numel(t) for t in x)
    return 0


@torch.no_grad()
def profile(
    model: nn.Module,
    args: tuple | None = None,
    kwargs: dict | None = None,
    device: str = "cpu",
    warmup: int = 3,
    reps: int = 10,
) -> dict:
    """Run one profiled forward. args/kwargs are positional/keyword model inputs."""
    args = tuple(args) if args else ()
    kwargs = kwargs or {}
    model = model.to(device).eval()
    probe = _Probe()
    probe.attach(model)
    _ = model(*args, **kwargs)
    probe.detach()

    runtime_ms = float("nan")
    try:
        for _ in range(warmup):
            model(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - t0) / reps * 1000.0
    except Exception:
        pass

    return {
        "params": count_params(model),
        "flops": 2 * probe.macs,
        "activations": probe.activations,
        "runtime_ms": runtime_ms,
    }
