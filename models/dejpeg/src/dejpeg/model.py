"""DeJPEGNet: a compact activation-free U-Net for JPEG artifact removal.

The network restores an RGB image by predicting a residual: ``out = x + head(features)``
with the head zero-initialized at birth, so an untrained model is exactly the identity.
Optimization starts from "do nothing" and only ever learns corrections.

Design properties that matter:

* **Activation-free** -- nonlinearity comes from SimpleGate (split + multiply) and
  multiplicative attention, not ReLU/GELU. Fewer kernel launches, friendlier to
  ONNX/WebGPU export.
* **Tile-invariant by construction** -- there is no global operation anywhere.
  Channel attention pools over a fixed 32x32 window, so restoring two overlapping
  crops of the same image yields identical pixels on the overlap. Inference can
  therefore tile arbitrarily large images with no seams (verified in tests).
* **Reparameterizable depthwise convs** -- training uses parallel {3x3, 1x1,
  identity} branches; :meth:`DeJPEGNet.fuse` folds them into a single depthwise
  3x3 for deployment, at zero quality cost.

Reference config (shipped weights): ``c0=40`` -> 2.63M parameters (~5 MB in FP16).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

ATTENTIONS = ("lsca", "none", "global")


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (NAFNet convention)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        d = (x - u).pow(2).mean(dim=1, keepdim=True)
        norm = (x - u) / torch.sqrt(d + self.eps)
        return norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split channels in half, multiply elementwise. The activation-free nonlinearity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class RepDWConv3x3(nn.Module):
    """Depthwise conv trained as parallel {3x3, 1x1, identity}; fused to one 3x3 for deploy."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw3(x) + self.dw1(x) + x

    def fuse(self) -> nn.Conv2d:
        """Fold both branches plus the identity into a single bias-free depthwise 3x3."""
        dev = self.dw3.weight.device
        fused = nn.Conv2d(self.dw3.in_channels, self.dw3.out_channels, 3, padding=1,
                          groups=self.dw3.groups, bias=False).to(dev)
        kernel = self.dw3.weight.data.clone()
        kernel[:, :, 1:2, 1:2] += self.dw1.weight.data
        kernel[:, :, 1:2, 1:2] += 1.0
        fused.weight.data.copy_(kernel)
        return fused


class LSCA(nn.Module):
    """Local Simple Channel Attention: fixed-window average pool -> 1x1 conv ->
    nearest-upsample -> multiplicative gate.

    The pool window is local (never image-sized), which is what preserves
    tile-invariance. See :class:`GlobalSCA` for the deliberate anti-pattern.
    """

    def __init__(self, channels: int, window: int = 32):
        super().__init__()
        # Floor-division pooling only: ceil_mode=True with dynamic H/W exports
        # a ceil() into the ONNX shape math, which onnxruntime-web cannot
        # evaluate. forward() reproduces ceil_mode=True + count_include_pad
        # =False exactly via a static pad and a valid-count denominator.
        self.pool = nn.AvgPool2d(window, stride=window)
        self.window = window
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        pad = self.window - 1
        xp = F.pad(x, (0, pad, 0, pad))
        mask = F.pad(torch.ones_like(x), (0, pad, 0, pad))
        pooled = self.pool(xp) / self.pool(mask)
        p = F.interpolate(self.conv(pooled), size=(h, w), mode="nearest")
        return x * p


class GlobalSCA(nn.Module):
    """Global-pool channel attention. Breaks tile-invariance; kept only as the
    negative control proving the tile-invariance test can actually fail."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x.mean(dim=[2, 3], keepdim=True))


def build_attention(name: str, channels: int, window: int) -> nn.Module:
    if name == "lsca":
        return LSCA(channels, window)
    if name == "global":
        return GlobalSCA(channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"unknown attention {name!r}; expected one of {ATTENTIONS}")


class NAFBlock(nn.Module):
    """NAFNet-style block, canonical two-branch form.

    Attention branch: LN -> 1x1(C->2C) -> reparam depthwise 3x3 -> SimpleGate ->
    channel attention -> 1x1(C->C), scaled into the residual by zero-init ``beta``.
    FFN branch: LN -> 1x1(C->2C) -> SimpleGate -> 1x1(C->C), scaled by zero-init
    ``gamma``. Zero-initialized scales make every block start as identity --
    deep stacks train stably from step zero.
    """

    def __init__(self, channels: int, attention: str = "lsca", window: int = 32, ffn_expand: int = 2):
        super().__init__()
        ffn_c = ffn_expand * channels
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, 2 * channels, 1)
        self.dwconv = RepDWConv3x3(2 * channels)
        self.sg1 = SimpleGate()
        self.attn = build_attention(attention, channels, window)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_c, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_c // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.conv2(self.attn(self.sg1(self.dwconv(self.conv1(self.norm1(x)))))) * self.beta
        return y + self.conv5(self.sg2(self.conv4(self.norm2(y)))) * self.gamma


class Upsample(nn.Module):
    """1x1 conv to 4x channels, then PixelShuffle(2). No transpose convs anywhere."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout * 4, 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class DeJPEGNet(nn.Module):
    """U-Net with three encoder levels (widths c0, 2c0, 4c0), a 4c0 bottleneck and
    a mirrored decoder. Strided-conv down / PixelShuffle up; skip connections at
    full resolution. Input/output are RGB in [0, 1]; output = input + residual.

    Args mirror the shipped checkpoint; changing them changes the parameter count.
    """

    def __init__(
        self,
        c0: int = 40,
        enc_blocks: tuple[int, ...] = (2, 2, 4),
        mid_blocks: int = 4,
        dec_blocks: tuple[int, ...] = (2, 2, 2),
        attention: str = "lsca",
        window: int = 32,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        widths = [c0, c0 * 2, c0 * 4]

        def block(c: int) -> NAFBlock:
            return NAFBlock(c, attention, window)

        self.shallow = nn.Conv2d(3, widths[0], 3, padding=1)

        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for lvl, (w, nb) in enumerate(zip(widths, enc_blocks, strict=False)):
            self.enc.append(nn.ModuleList([block(w) for _ in range(nb)]))
            if lvl < len(widths) - 1:
                self.downs.append(nn.Conv2d(widths[lvl], widths[lvl + 1], 3, stride=2, padding=1))
        self.downs.append(nn.Conv2d(widths[-1], widths[-1], 3, stride=2, padding=1))  # bottleneck entry

        self.mid = nn.ModuleList([block(widths[-1]) for _ in range(mid_blocks)])

        dec_widths = list(reversed(widths))
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i, (w, nb) in enumerate(zip(dec_widths, dec_blocks, strict=False)):
            cin = widths[-1] if i == 0 else dec_widths[i - 1]
            self.ups.append(Upsample(cin, w))
            self.dec.append(nn.ModuleList([block(w) for _ in range(nb)]))

        self.head = nn.Conv2d(widths[0], 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _run_blocks(self, blocks, x: torch.Tensor) -> torch.Tensor:
        for blk in blocks:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = ckpt.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.shallow(x)
        skips = []
        for lvl, blocks in enumerate(self.enc):
            x = self._run_blocks(blocks, x)
            skips.append(x)
            if lvl < len(self.enc) - 1:
                x = self.downs[lvl](x)
        x = self.downs[-1](x)
        x = self._run_blocks(self.mid, x)
        for i, (up, blocks) in enumerate(zip(self.ups, self.dec, strict=False)):
            x = up(x) + skips[len(skips) - 1 - i]
            x = self._run_blocks(blocks, x)
        return inp + self.head(x)

    @torch.no_grad()
    def fuse(self) -> DeJPEGNet:
        """Return an eval copy with all RepDWConv3x3 folded into plain depthwise convs."""
        model = copy.deepcopy(self).eval()

        def walk(m: nn.Module) -> None:
            for name, child in list(m.named_children()):
                if isinstance(child, RepDWConv3x3):
                    setattr(m, name, child.fuse())
                else:
                    walk(child)

        walk(model)
        return model
