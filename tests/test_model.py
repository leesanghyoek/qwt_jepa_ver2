"""Gate G2: forward/backward, EMA, teacher va ablation cross-modal."""

from __future__ import annotations

import pytest
import torch

from qjepa.models.joint_jepa import (
    JointRestorationJEPA,
    image_tokens,
    imu_tokens,
    teacher_momentum,
)


def make_batch(b=2, window=128):
    image = torch.rand(b, 3, 256, 256)
    imu = torch.randn(b, 6, window)
    image_time = torch.full((b,), 1.0)
    imu_times = torch.stack([torch.linspace(0.365, 1.635, window) for _ in range(b)])
    return image, imu, image_time, imu_times


@pytest.mark.parametrize("batch", [1, 2])
def test_forward_shapes(batch):
    model = JointRestorationJEPA()
    out = model(*make_batch(batch))
    assert out["image_hat"].shape == (batch, 3, 256, 256)
    assert out["imu_hat_phys"].shape == (batch, 6, 128)
    assert out["fi"].shape == (batch, 128, 16, 16)
    assert out["fu"].shape == (batch, 128, 8)
    assert image_tokens(out["zi"]).shape == (batch, 256, 128)
    assert imu_tokens(out["zu"]).shape == (batch, 8, 128)
    assert torch.isfinite(out["image_hat"]).all() and torch.isfinite(out["imu_hat_phys"]).all()


def test_zero_init_heads_give_identity_at_start():
    """Head zero-init -> output dau tien la inverse cua he so nhieu."""
    model = JointRestorationJEPA()
    image, imu, t, ut = make_batch(1)
    out = model(image, imu, t, ut)
    assert (out["image_hat"] - image).abs().max() < 1e-5
    assert (out["imu_hat_phys"] - imu).abs().max() < 1e-4


def test_rejects_wrong_imu_window():
    """L=256 phai bi tu choi: 128 hang va anh 256x256 khong lien quan nhau."""
    model = JointRestorationJEPA()
    image, _, t, _ = make_batch(1)
    with pytest.raises(ValueError, match="128"):
        model(image, torch.randn(1, 6, 256), t, torch.stack([torch.linspace(0, 2.55, 256)]))


def test_rejects_missing_gyro_channels():
    model = JointRestorationJEPA()
    image, _, t, ut = make_batch(1)
    with pytest.raises(ValueError):
        model(image, torch.randn(1, 3, 128), t, ut)


def test_teacher_is_separate_and_not_trained():
    model = JointRestorationJEPA()
    model.initialize_teacher()
    online_ids = {id(p) for p in model.online_parameters()}
    teacher_params = list(model.teacher_image_encoder.parameters())
    assert all(not p.requires_grad for p in teacher_params)
    assert all(id(p) not in online_ids for p in teacher_params)
    # Khong chia se storage voi online.
    online_storage = {p.data_ptr() for p in model.image_encoder.parameters()}
    assert all(p.data_ptr() not in online_storage for p in teacher_params)


def test_teacher_stays_eval_when_parent_trains():
    model = JointRestorationJEPA()
    model.initialize_teacher()
    model.train()
    assert not model.teacher_image_encoder.training
    assert not model.teacher_imu_encoder.training


def test_ema_matches_hand_computation():
    model = JointRestorationJEPA()
    model.initialize_teacher()
    teacher_p = next(model.teacher_image_encoder.parameters())
    online_p = next(model.image_encoder.parameters())
    with torch.no_grad():
        online_p.add_(0.5)
    before = teacher_p.detach().clone()
    expected = 0.99 * before + 0.01 * online_p.detach()
    model.update_teacher(0.99)
    assert torch.allclose(teacher_p, expected, atol=1e-7)


def test_teacher_momentum_schedule():
    assert teacher_momentum(0, 8000) == pytest.approx(0.99)
    assert teacher_momentum(7999, 8000) == pytest.approx(0.999)
    assert 0.99 < teacher_momentum(4000, 8000) < 0.999


def test_predictors_receive_gradient():
    model = JointRestorationJEPA()
    model.initialize_teacher()
    image, imu, t, ut = make_batch(2)
    out = model(image, imu, t, ut)
    pred = model.image_predictor(image_tokens(out["zi"]))
    pred.square().mean().backward()
    grads = [p.grad for p in model.image_predictor.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in model.teacher_image_encoder.parameters())


def test_cross_modal_off_gives_true_independence():
    """Muc 9.3: doi IMU khong duoc doi output anh khi cross_modal=false."""
    model = JointRestorationJEPA(cross_modal=False).eval()
    image, imu, t, ut = make_batch(2)
    with torch.no_grad():
        a = model(image, imu, t, ut)
        b = model(image, imu + 5.0, t, ut)
    assert torch.allclose(a["image_hat"], b["image_hat"], atol=1e-6)
    assert not torch.allclose(a["imu_hat_phys"], b["imu_hat_phys"], atol=1e-6)

    with torch.no_grad():
        c = model(torch.rand_like(image), imu, t, ut)
    assert torch.allclose(a["imu_hat_phys"], c["imu_hat_phys"], atol=1e-6)


def test_cross_modal_on_creates_dependence():
    """Bat cross_modal -> IMU phai anh huong toi FUSED FEATURE cua nhanh anh.

    Kiem tra tren `zi` chu khong tren `image_hat`: head he so duoc zero-init nen
    luc khoi tao output luon bang inverse cua he so nhieu du fusion tra gi. Do la
    hanh vi dung (muc 10.3), khong phai fusion bi dut.
    """
    model = JointRestorationJEPA(cross_modal=True).eval()
    image, imu, t, ut = make_batch(2)
    with torch.no_grad():
        a = model(image, imu, t, ut)
        b = model(image, imu + 5.0, t, ut)
    assert not torch.allclose(a["zi"], b["zi"], atol=1e-7)
    # Sau khi head khong con zero, anh huong do phai den duoc output.
    with torch.no_grad():
        model.image_decoder.head.weight.normal_(0, 1e-3)
        a2 = model(image, imu, t, ut)
        b2 = model(image, imu + 5.0, t, ut)
    assert not torch.allclose(a2["image_hat"], b2["image_hat"], atol=1e-7)


def test_export_state_drops_teacher_and_predictor():
    model = JointRestorationJEPA()
    model.initialize_teacher()
    keys = model.export_state()
    assert not any(k.startswith(("teacher_image_encoder", "teacher_imu_encoder")) for k in keys)
    assert not any(k.startswith(("image_predictor", "imu_predictor")) for k in keys)
    assert any(k.startswith("image_encoder") for k in keys)
    assert any(k.startswith("normalizer") for k in keys)


def test_backbone_moves_after_a_few_updates():
    """Head zero-init -> gradient dau co the bang 0; phai kiem tra sau vai step."""
    torch.manual_seed(0)
    model = JointRestorationJEPA()
    opt = torch.optim.AdamW(model.online_parameters(), lr=1e-3)
    image, imu, t, ut = make_batch(2)
    target = torch.rand_like(image)
    before = next(model.image_encoder.parameters()).detach().clone()
    for _ in range(5):
        opt.zero_grad()
        out = model(image, imu, t, ut)
        (out["image_hat"] - target).abs().mean().backward()
        opt.step()
    assert not torch.equal(before, next(model.image_encoder.parameters()))
