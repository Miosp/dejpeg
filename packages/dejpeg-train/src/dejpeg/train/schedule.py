"""Experiment infrastructure (spec §7).

Seeds, bf16 autocast, cosine LR with warmup, EMA shadow weights, and resumable
checkpoints. Checkpoints capture model + EMA + optimizer + disc + dataloader RNG
state + manifest hash + config + config hash so a run resumes to a bit-identical
state. EMA is what we eval and ship, never the raw weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# -------------------------------------------------------------------- seeding


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ----------------------------------------------------------------- autocast


@contextmanager
def bf16_autocast(enabled: bool = True, device_type: str = "cuda"):
    """bf16 autocast. Use bf16, NOT fp16 -- fp16 autocast + LPIPS is a NaN source
    (spec). FP16 is the deploy format only (Phase 3b export), never training."""
    if not enabled:
        yield
        return
    dt = torch.bfloat16
    with torch.autocast(device_type=device_type, dtype=dt, enabled=True):
        yield


# ------------------------------------------------------------- gradient clipping


def enable_fast_gpu() -> None:
    """Maximum conv/matmul throughput on Ampere+ (RTX 3080). Enable tf32 for matmul
    and cudnn, autotune conv algorithms (input shapes are fixed during training),
    and drop cudnn determinism. tf32's precision cost is well within training noise.
    Call once at the start of any GPU training run."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def prepare_model_for_training(model, compile_mode="default"):
    """Enable fast-GPU flags and torch.compile the model.

    Measured (Phase 0.6): torch.compile(default) gives ~2.7x throughput
    (495 -> 179 ms/iter on the C0=30 student) AND ~34% lower activation memory
    (8.5 -> 5.6 GB) via kernel fusion -> fewer launches + smaller live sets.
    Returns the compiled forward module; keep the original ``model`` for
    state_dict / EMA / export (params are shared)."""
    enable_fast_gpu()
    return torch.compile(model, mode=compile_mode) if compile_mode else model


def clip_grad_norm(parameters, max_norm: float) -> float:
    """Gradient-norm clip, shared across all trainers.

    Phase-0.6-P0 sweep found higher LR diverges without clipping (L1@2e-2 -> 111);
    clipping at ~1.0 lets the trainer run the canonical 1e-3 stably and leaves
    headroom to push LR if a phase needs it. Pass max_norm<=0 (or None) to skip.
    Returns the pre-clip total norm.
    """
    if max_norm is None or max_norm <= 0:
        return float("nan")
    return float(torch.nn.utils.clip_grad_norm_(list(parameters), max_norm))


# ----------------------------------------------------------------- LR schedule


def cosine_lr(base_lr: float, total_steps: int, warmup: int = 0):
    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return base_lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return f


# ------------------------------------------------------------------------ EMA


class EMA:
    """Exponential moving average of model parameters. Eval/ship the shadow, not raw."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    def state_dict(self) -> dict:
        return {n: t.clone() for n, t in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {n: t.clone() for n, t in sd.items()}

    @contextmanager
    def swap(self, model: torch.nn.Module):
        """Temporarily load shadow into model (for eval), restore on exit."""
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self.shadow:
                        p.copy_(backup[n])


# ----------------------------------------------------------------- checkpoints


def config_hash(config: dict | None) -> str:
    if config is None:
        return ""
    s = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class CheckpointMeta:
    step: int
    manifest_hash: str | None
    config: dict | None
    config_hash: str


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer | None = None,
    disc: torch.nn.Module | None = None,
    step: int = 0,
    manifest_hash: str | None = None,
    config: dict | None = None,
    rng_states: dict | None = None,
) -> None:
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "disc": disc.state_dict() if disc is not None else None,
        "rng": rng_states if rng_states is not None else capture_rng(),
        "manifest_hash": manifest_hash,
        "config": config,
        "config_hash": config_hash(config),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)
