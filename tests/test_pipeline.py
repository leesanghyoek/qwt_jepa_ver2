"""Gate G5: save/load/export, cong them kiem tra config strict va loss."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from qjepa.config import Config, load_config
from qjepa.data.normalize import ImuNormalizer
from qjepa.models.joint_jepa import JointRestorationJEPA
from qjepa.training.losses import jepa_weight, layer_norm_no_affine
from qjepa.training.trainer import build_model, lr_lambda, parameter_groups


def make_batch(b=2):
    return (
        torch.rand(b, 3, 256, 256),
        torch.randn(b, 6, 128),
        torch.full((b,), 1.0),
        torch.stack([torch.linspace(0.365, 1.635, 128) for _ in range(b)]),
    )


def test_save_load_reproduces_output(tmp_path):
    torch.manual_seed(0)
    model = JointRestorationJEPA().eval()
    with torch.no_grad():
        model.image_decoder.head.weight.normal_(0, 1e-3)
    batch = make_batch(1)
    with torch.no_grad():
        before = model(*batch)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)

    loaded = JointRestorationJEPA().eval()
    loaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        after = loaded(*batch)
    assert torch.allclose(before["image_hat"], after["image_hat"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(before["imu_hat_phys"], after["imu_hat_phys"], atol=1e-6, rtol=1e-5)


def test_export_matches_full_model_and_drops_teacher(tmp_path):
    torch.manual_seed(0)
    model = JointRestorationJEPA().eval()
    with torch.no_grad():
        model.image_decoder.head.weight.normal_(0, 1e-3)
    model.initialize_teacher()
    batch = make_batch(1)
    with torch.no_grad():
        full = model(*batch)

    exported = JointRestorationJEPA().eval()
    missing, unexpected = exported.load_state_dict(model.export_state(), strict=False)
    assert not unexpected
    # Chi predictor duoc phep thieu; encoder/fusion/decoder phai day du.
    assert all(m.startswith(("image_predictor", "imu_predictor")) for m in missing)
    with torch.no_grad():
        out = exported(*batch)
    assert torch.allclose(full["image_hat"], out["image_hat"], atol=1e-6, rtol=1e-5)


def test_strict_load_rejects_architecture_mismatch():
    a = JointRestorationJEPA(cross_modal=True)
    b = JointRestorationJEPA(image_transform="dwt_haar_baseline")
    with pytest.raises(RuntimeError):
        b.load_state_dict(a.state_dict(), strict=True)


def test_normalizer_travels_with_checkpoint(tmp_path):
    n = ImuNormalizer(mean=torch.arange(6).float(), std=torch.full((6,), 2.0))
    model = JointRestorationJEPA(normalizer=n)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    fresh = JointRestorationJEPA()
    fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.allclose(fresh.normalizer.mu, torch.arange(6).float())
    assert torch.allclose(fresh.normalizer.scale, torch.full((6,), 2.0))


def test_config_rejects_unknown_and_bad_values(tmp_path):
    from qjepa.config import _build

    with pytest.raises(ValueError, match="khong duoc ho tro"):
        _build(Config, {"data": {"imu_window_sampels": 128}})
    with pytest.raises(ValueError, match="128"):
        replace(Config(), data=replace(Config().data, imu_window_samples=256)).validate()
    with pytest.raises(ValueError, match="allow_silent_fallback"):
        replace(Config(), transform=replace(Config().transform, allow_silent_fallback=True)).validate()
    with pytest.raises(ValueError, match="max_optimizer_steps"):
        replace(Config(), train=replace(Config().train, stage_a_optimizer_steps=1)).validate()


def test_shipped_configs_are_valid():
    from pathlib import Path

    for path in sorted(Path("configs").glob("*.yaml")):
        load_config(path)


def test_parameter_groups_exclude_bias_and_norm_from_decay():
    model = JointRestorationJEPA()
    groups = parameter_groups(model, 1e-4, True)
    assert groups[0]["weight_decay"] == 1e-4 and groups[1]["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in groups[0]["params"])
    assert all(p.ndim <= 1 for p in groups[1]["params"])


def test_parameter_groups_exclude_teacher():
    model = JointRestorationJEPA()
    model.initialize_teacher()
    ids = {id(p) for g in parameter_groups(model, 1e-4, True) for p in g["params"]}
    assert all(id(p) not in ids for p in model.teacher_image_encoder.parameters())


def test_lr_schedule_warmup_then_cosine():
    total, warm = 1000, 0.05
    assert lr_lambda(0, total, warm, 2e-4, 1e-6) == pytest.approx(1 / 50)
    assert lr_lambda(49, total, warm, 2e-4, 1e-6) == pytest.approx(1.0)
    assert lr_lambda(total - 1, total, warm, 2e-4, 1e-6) == pytest.approx(1e-6 / 2e-4, abs=1e-6)


def test_jepa_ramp_bounds():
    assert jepa_weight(0, 1000) == pytest.approx(0.01)
    assert jepa_weight(999, 1000) == pytest.approx(0.10)
    assert jepa_weight(5000, 1000) == pytest.approx(0.10)


def test_layer_norm_no_affine_has_no_parameters():
    x = torch.randn(2, 8, 128)
    y = layer_norm_no_affine(x)
    assert torch.allclose(y.mean(-1), torch.zeros(2, 8), atol=1e-5)
    assert torch.allclose(y.std(-1, unbiased=False), torch.ones(2, 8), atol=1e-3)
