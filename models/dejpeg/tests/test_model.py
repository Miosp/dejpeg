"""Architecture invariants: identity-at-init, parameter count, fusion parity."""
import pytest
import torch

from dejpeg.model import DeJPEGNet

SHIPPED_PARAMS = 2_629_163  # c0=40 -- locks the architecture to the shipped checkpoint


def test_shipped_config_parameter_count():
    model = DeJPEGNet(c0=40)
    assert sum(p.numel() for p in model.parameters()) == SHIPPED_PARAMS


def test_zero_init_head_makes_untrained_model_the_identity():
    model = DeJPEGNet(c0=8).eval()
    x = torch.rand(1, 3, 64, 64)
    torch.testing.assert_close(model(x), x)


def test_fused_model_matches_training_graph():
    torch.manual_seed(0)
    model = DeJPEGNet(c0=16).eval()
    x = torch.rand(1, 3, 64, 96)
    fused = model.fuse()
    assert not any(hasattr(m, "dw3") for m in fused.modules())
    torch.testing.assert_close(fused(x), model(x), rtol=1e-4, atol=1e-5)


def test_sizes_multiple_of_the_32px_contract():
    """Raw forward requires dims divisible by 32 (attention window); infer.py pads
    arbitrary sizes up to this contract."""
    model = DeJPEGNet(c0=8).eval()
    for shape in [(1, 3, 96, 160), (2, 3, 32, 224), (1, 3, 128, 128)]:
        out = model(torch.rand(*shape))
        assert out.shape == shape


@pytest.mark.parametrize("c0", [16, 40])
def test_grad_checkpoint_matches_plain_path(c0):
    torch.manual_seed(0)
    plain = DeJPEGNet(c0=c0).train()
    ckpted = DeJPEGNet(c0=c0, grad_checkpoint=True).train()
    ckpted.load_state_dict(plain.state_dict())
    x = torch.rand(1, 3, 64, 64)
    plain(x).sum().backward()
    ckpted(x).sum().backward()
    for n, pa, pb in zip(plain.state_dict(), plain.parameters(), ckpted.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, msg=lambda m, name=n: f"{name}: {m}")
