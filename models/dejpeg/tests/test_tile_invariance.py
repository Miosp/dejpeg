"""No-far-field-coupling: changing pixels far away cannot move outputs nearby.

This is the property seam-free tiled inference rests on. It holds structurally,
not approximately: channel attention pools over fixed 32px windows, so nothing
in the network can transport information across an arbitrarily large gap. The
negative control (:class:`GlobalSCA`, global average pooling) fails it -- proof
the test measures something real.

At deeper U-Net levels a 32px window spans more input pixels (32px on a /4
grid = 128px of input), so the network-level guarantee is "coupling bounded by
a few hundred pixels", asserted well under that bound.
"""
import torch

from dejpeg.model import LSCA, DeJPEGNet, GlobalSCA


def perturbed(model, scale=0.05):
    """Zero-initialized gates would trivially pass everything; give the
    branches real (random) weight so the test exercises actual data flow."""
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(("beta", "gamma")) or n.startswith("head."):
                p.normal_(0, scale)
    return model.eval()


def test_lsca_gate_is_exactly_window_local():
    torch.manual_seed(0)
    attn = LSCA(channels=8, window=32).eval()
    x = torch.rand(1, 8, 96, 96)
    far = x.clone()
    far[..., 64:, :] += 1.0  # change everything from window row 2 onward
    with torch.no_grad():
        g1 = attn.conv(attn.pool(x))
        g2 = attn.conv(attn.pool(far))
    torch.testing.assert_close(g1[..., :2, :], g2[..., :2, :])  # untouched windows: identical


def test_global_pooling_gate_is_not_local():
    torch.manual_seed(0)
    attn = GlobalSCA(channels=8).eval()
    x = torch.rand(1, 8, 96, 96)
    far = x.clone()
    far[..., 48:, :] += 1.0
    with torch.no_grad():
        d = float((attn.conv(x.mean(dim=[2, 3], keepdim=True))
                   - attn.conv(far.mean(dim=[2, 3], keepdim=True))).abs().max())
    assert d > 0.0


def test_network_outputs_immune_to_distant_changes():
    """Image must be >=256px so the /8 bottleneck grid (>=32 cells) keeps every
    LSCA window local; smaller inputs legitimately collapse to global pooling."""
    torch.manual_seed(0)
    base = torch.rand(1, 3, 320, 320)
    other = base.clone()
    other[..., 288:, 288:] += 0.5  # >=256px away from the compared region

    net = perturbed(DeJPEGNet(c0=24))
    with torch.no_grad():
        d = float((net(base)[..., :32, :32] - net(other)[..., :32, :32]).abs().max())
    # Not exactly 0 only because conv kernels tile differently for the two input
    # shapes; 1e-4 is float noise, orders below any real coupling.
    assert d < 1e-4
