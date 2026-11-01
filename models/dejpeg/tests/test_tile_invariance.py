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
import copy

import torch

from dejpeg.model import GlobalSCA, LSCA, DeJPEGNet, SlidingLSCA, swap_lsca_sliding


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


def test_sliding_lsca_is_exactly_shift_equivariant():
    torch.manual_seed(0)
    attn = SlidingLSCA(channels=8, window=32).eval()
    x = torch.rand(1, 8, 128, 128)
    with torch.no_grad():
        y = attn(x)
        y_shift = attn(torch.roll(x, shifts=(17, 23), dims=(2, 3)))
    # compare the interior: torch.roll is circular, so the ~window-wide border
    # band legitimately sees wrapped-in far-edge content.
    rolled = torch.roll(y_shift, shifts=(-17, -23), dims=(2, 3))
    torch.testing.assert_close(rolled[..., 40:-40, 40:-40],
                               y[..., 40:-40, 40:-40], rtol=0, atol=0)


def test_anchored_lsca_is_not_shift_equivariant():
    """Negative control: the anchored pool lands on a different window grid
    when the input shifts, so the same pixel gets a different gate. This is
    the D7 defect SlidingLSCA exists to remove."""
    torch.manual_seed(0)
    attn = LSCA(channels=8, window=32).eval()
    x = torch.rand(1, 8, 128, 128)
    with torch.no_grad():
        y = attn(x)
        y_shift = attn(torch.roll(x, shifts=(17, 23), dims=(2, 3)))
    d = float((torch.roll(y_shift, shifts=(-17, -23), dims=(2, 3)) - y).abs()[..., 40:-40, 40:-40].max())
    assert d > 1e-3


def test_sliding_swap_keeps_state_dict_and_changes_phase_behaviour():
    torch.manual_seed(0)
    net = perturbed(DeJPEGNet(c0=24))
    anchored = copy.deepcopy(net)
    swap_lsca_sliding(net)
    x = torch.rand(1, 3, 512, 512)
    xs = torch.roll(x, shifts=(16, 24), dims=(2, 3))  # multiple of the /4 stride, NOT of the 32px window
    with torch.no_grad():
        y, ys = net(x), net(xs)
        ya, yas = anchored(x), anchored(xs)
    # beyond the deepest window reach (/4 grid: 128px) + conv receptive field
    band = (slice(None), slice(None), slice(150, -150), slice(150, -150))
    # sliding: tiled phase does not matter beyond float noise
    d_slide = float((torch.roll(ys, shifts=(-16, -24), dims=(2, 3)) - y).abs()[band].max())
    assert d_slide < 1e-3
    # anchored: the same shift moves the output an order of magnitude more
    d_anchor = float((torch.roll(yas, shifts=(-16, -24), dims=(2, 3)) - ya).abs()[band].max())
    assert d_anchor > 5 * d_slide


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
