"""Gate G1: transform dung va inverse co gradient (spec muc 17)."""

from __future__ import annotations

import torch

from qjepa.transforms import build_image_transform, build_imu_transform


def _relative(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_qwt_roundtrip_cases():
    q = build_image_transform("qwt")
    torch.manual_seed(0)
    cases = {
        "random": torch.rand(2, 3, 64, 64),
        "constant": torch.full((1, 3, 32, 32), 0.5),
        "impulse": torch.zeros(1, 3, 32, 32),
        "ramp": torch.linspace(0, 1, 64).view(1, 1, 1, 64).expand(1, 3, 64, 64).contiguous(),
    }
    cases["impulse"][0, 1, 7, 11] = 1.0
    for name, x in cases.items():
        c, layout = q.analysis(x)
        assert c.shape[1] == 48, f"{name}: QWT phai tra 48 kenh, nhan {c.shape[1]}"
        r = q.synthesis(c, layout)
        assert _relative(r, x) <= 1e-5, f"{name}: relative error {_relative(r, x)}"
        assert (r - x).abs().max().item() <= 1e-5


def test_qwt_zero_signal_uses_absolute_error():
    q = build_image_transform("qwt")
    x = torch.zeros(1, 3, 32, 32)
    c, layout = q.analysis(x)
    assert q.synthesis(c, layout).abs().max().item() <= 1e-6


def test_qwt_rgb_channels_stay_separate():
    """Nhieu loan kenh R khong duoc doi he so cua G va B (spec muc 6.2)."""
    q = build_image_transform("qwt")
    x = torch.rand(1, 3, 32, 32)
    c0, _ = q.analysis(x)
    x2 = x.clone()
    x2[:, 0] += 0.1
    c1, _ = q.analysis(x2)
    delta = (c1 - c0).reshape(1, 3, 4, 4, 16, 16).abs().amax(dim=(2, 3, 4, 5))[0]
    assert delta[0] > 1e-3
    assert delta[1].item() == 0.0 and delta[2].item() == 0.0


def test_qwt_trees_form_hilbert_pair():
    """Hai cay phai triet tieu phan lon nua pho am - bang chung dual-tree.

    Neu ai do thay QWT bang hai cay GIONG NHAU (vi du Haar doi ten), ty le nay
    tut ve khoang 0 va test fail.
    """
    import numpy as np

    q = build_image_transform("qwt")
    n = 64

    def wavelet(component: int) -> np.ndarray:
        c, layout = q.analysis(torch.zeros(1, 3, n, n))
        c = c.reshape(1, 3, 4, 4, n // 2, n // 2).clone()
        c[0, 0, 2, component, n // 4, n // 4] = 1.0
        return q.synthesis(c.reshape(1, 48, n // 2, n // 2), layout)[0, 0, n // 2].numpy()

    spectrum = np.fft.fft(wavelet(0) + 1j * wavelet(1))
    half = len(spectrum) // 2
    pos = np.sum(np.abs(spectrum[1:half]) ** 2)
    neg = np.sum(np.abs(spectrum[half + 1 :]) ** 2)
    assert pos / (pos + neg) > 0.75, f"khong phai cap Hilbert: {pos / (pos + neg):.3f}"


def test_synthesis_gradient_flows():
    q = build_image_transform("qwt")
    x = torch.rand(1, 3, 32, 32)
    c, layout = q.analysis(x)
    c = c.detach().clone().requires_grad_(True)
    (q.synthesis(c, layout) * torch.randn(1, 3, 32, 32)).sum().backward()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert c.grad.abs().max() > 0


def test_synthesis_gradcheck_double():
    q = build_image_transform("qwt").double()
    x = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    c, layout = q.analysis(x)
    c = c.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: q.synthesis(t, layout), (c,), eps=1e-6, atol=1e-8)


def test_haar_imu_roundtrip_and_packing():
    h = build_imu_transform("dwt_haar_1d")
    u = torch.randn(2, 6, 128)
    c, layout = h.analysis(u)
    assert c.shape == (2, 12, 64)
    assert _relative(h.synthesis(c, layout), u) <= 1e-5
    # Pack order: sau kenh approx truoc, roi sau kenh detail.
    assert layout.extra["pack_order"] == "approx_all_channels_then_detail_all_channels"


def test_transform_rejects_odd_size():
    import pytest

    q = build_image_transform("qwt")
    with pytest.raises(ValueError):
        q.analysis(torch.rand(1, 3, 33, 32))
    h = build_imu_transform("dwt_haar_1d")
    with pytest.raises(ValueError):
        h.analysis(torch.randn(1, 6, 127))


def test_no_silent_fallback_for_unknown_backend():
    import pytest

    with pytest.raises(ValueError, match="qwt"):
        build_image_transform("dtcwt")


def test_layout_revision_mismatch_is_rejected():
    import pytest

    from qjepa.transforms.layout import Layout

    q = build_image_transform("qwt")
    c, layout = q.analysis(torch.rand(1, 3, 16, 16))
    bad = Layout(**{**layout.to_dict(), "revision": "0.0.0"})
    with pytest.raises(ValueError, match="revision"):
        q.synthesis(c, bad)
