"""Phase 0 Task 0.9 -- loss tests.

Key assertions (plan):
  * blockiness: high on a genuinely blocky image; near zero on a clean image with
    strong edges falling on the 8x8 grid (the relative-ratio property).
  * LDL: artifact map high on synthetic HF noise, low on clean real texture.
"""
from __future__ import annotations

import pytest
import torch

from dejpeg.loss.blockiness import blockiness_loss
from dejpeg.loss.contrastive import DegEncoderLoss, enqueue_dequeue, info_nce
from dejpeg.loss.distill import pseudo_label_loss, spatial_affinity_loss
from dejpeg.loss.gan import PatchDiscriminator, discriminator_hinge_loss, generator_hinge_loss
from dejpeg.loss.ldl import ldl_loss


# ----------------------------------------------------------- image fixtures


def blocky_image(size=64, block=8, channels=3):
    img = torch.zeros(1, channels, size, size)
    for br in range(size // block):
        for bc in range(size // block):
            val = (br + bc) / (2 * (size // block - 1) + 1)
            img[:, :, br * block : (br + 1) * block, bc * block : (bc + 1) * block] = val
    return img


def smooth_gradient(size=64, channels=3):
    idx = torch.arange(size).float()
    img = (idx.view(size, 1) + idx.view(1, size)) / (2 * size)
    return img.unsqueeze(0).unsqueeze(0).expand(1, channels, size, size).contiguous()


def grid_edge_image(size=64, block=8, channels=3):
    idx = torch.arange(size)
    stripe = ((idx // block) % 2).float()
    vert = stripe.view(1, 1, 1, size).expand(1, channels, size, size)
    horiz = stripe.view(1, 1, size, 1).expand(1, channels, size, size)
    img = (vert + horiz) / 2
    return img.contiguous()


# ----------------------------------------------------------------- blockiness


def test_blockiness_high_on_blocky_image():
    pred = blocky_image()
    target = smooth_gradient()
    loss = blockiness_loss(pred, target, offset=(0, 0))
    assert loss.item() > 0.5, f"blockiness {loss.item():.3f} not high on blocky image"


def test_blockiness_near_zero_on_clean_image_with_grid_edges():
    # Same image as pred and target: strong edges ON the 8x8 grid, but relative
    # ratio difference is zero -> no penalty (absolute metric would fire here).
    img = grid_edge_image()
    loss = blockiness_loss(img, img, offset=(0, 0))
    assert loss.item() < 1e-4, f"blockiness {loss.item():.4f} should be ~0 when pred==target"


def test_blockiness_zero_when_pred_cleaner_than_target():
    pred = smooth_gradient()
    target = blocky_image()
    loss = blockiness_loss(pred, target, offset=(0, 0))
    assert loss.item() < 1e-4, "relu should clip negative ratio difference to zero"


# ----------------------------------------------------------------------- LDL


def test_ldl_high_on_synthetic_hf_noise():
    torch.manual_seed(0)
    target = torch.zeros(1, 3, 64, 64)              # smooth target
    pred = target + 0.1 * torch.randn(1, 3, 64, 64)  # residual = HF noise
    loss = ldl_loss(pred, target)
    assert loss.item() > 1e-3, f"LDL {loss.item():.4f} not high on HF noise"


def test_ldl_low_on_clean_texture():
    torch.manual_seed(0)
    target = torch.randn(1, 3, 64, 64)  # high local variance everywhere (texture)
    pred = target.clone()               # zero residual
    loss = ldl_loss(pred, target)
    assert loss.item() < 1e-6, f"LDL {loss.item():.6f} not ~0 with zero residual"


def test_ldl_downweights_residual_on_texture_vs_smooth():
    # Same noise residual: penalized on smooth target, attenuated on textured one.
    torch.manual_seed(0)
    noise = 0.1 * torch.randn(1, 3, 64, 64)
    smooth = torch.zeros(1, 3, 64, 64)
    textured = torch.randn(1, 3, 64, 64)
    loss_smooth = ldl_loss(smooth + noise, smooth).item()
    loss_texture = ldl_loss(textured + noise, textured).item()
    assert loss_smooth > loss_texture * 5


# ----------------------------------------------------------- smoke (other losses)


def test_gan_discriminator_and_hinge():
    disc = PatchDiscriminator(in_ch=6)
    x = torch.randn(2, 6, 64, 64)
    out = disc(x)
    assert out.dim() == 4 and out.shape[0] == 2
    rl = torch.randn(2, 1, 8, 8)
    fl = torch.randn(2, 1, 8, 8)
    assert torch.isfinite(discriminator_hinge_loss(rl, fl))
    assert torch.isfinite(generator_hinge_loss(fl))


def test_info_nce_finite_and_enqueue():
    q = torch.randn(4, 32)
    k = torch.randn(4, 32)
    queue = torch.randn(16, 32)
    loss = info_nce(q, k, queue)
    assert torch.isfinite(loss)
    new_q = enqueue_dequeue(queue, k, max_size=16)
    assert new_q.shape == (16, 32)


def test_deg_encoder_loss_finite():
    loss_fn = DegEncoderLoss(emb_dim=32)
    query = torch.randn(4, 32)
    key = torch.randn(4, 32)
    queue = torch.randn(16, 32)
    lq_target = torch.rand(4, 3, 64, 64)
    true_qf = torch.randint(1, 101, (4,)).float()
    loss, parts = loss_fn(query, key, queue, lq_target, true_qf)
    assert torch.isfinite(loss)
    assert set(parts) == {"contrastive", "recon", "qf"}


def test_distill_losses_finite():
    sfeats = [torch.randn(1, 32, 16, 16), torch.randn(1, 64, 8, 8)]
    tfeats = [torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    sa = spatial_affinity_loss(sfeats, tfeats)
    assert torch.isfinite(sa)
    pl = pseudo_label_loss(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    assert torch.isfinite(pl)


def test_perceptual_forward_smoke():
    try:
        from dejpeg.loss.perceptual import PerceptualLoss
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"lpips unavailable: {e}")
    try:
        loss = PerceptualLoss(net="vgg", crop=128)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"lpips weights unavailable offline: {e}")
    loss.eval()
    pred = torch.rand(1, 3, 192, 192)
    target = torch.rand(1, 3, 192, 192)
    with torch.no_grad():
        v = loss(pred, target)
    assert torch.isfinite(v)
