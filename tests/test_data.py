"""Gate G0: pairing, split, shuffle, corruption doc lap va normalization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qjepa.corruptions.image import ImageCorruptionConfig, ImageCorruptor
from qjepa.corruptions.imu import ImuCorruptionConfig, ImuCorruptor
from qjepa.corruptions.rng import derive_seed
from qjepa.data.manifest import _window_centers
from qjepa.data.normalize import ImuNormalizer


def test_seed_is_stable_across_processes():
    """Khong duoc dung Python hash(): no randomize theo process."""
    a = derive_seed(42, "train", 0, "Env/Data_easy/P000", "image", 7)
    assert a == derive_seed(42, "train", 0, "Env/Data_easy/P000", "image", 7)
    assert a != derive_seed(42, "train", 1, "Env/Data_easy/P000", "image", 7)
    assert a != derive_seed(42, "valid", 0, "Env/Data_easy/P000", "image", 7)
    assert a != derive_seed(42, "train", 0, "Env/Data_easy/P001", "image", 7)
    assert a != derive_seed(42, "train", 0, "Env/Data_easy/P000", "imu", 7)


def test_window_centers_and_selection():
    t = np.arange(1000) * 0.01
    centers = _window_centers(t, 128)
    assert len(centers) == 1000 - 128 + 1
    # Center cua window bat dau o s la trung binh hai dau.
    assert centers[0] == pytest.approx((t[0] + t[127]) / 2)
    # Window 128 mau o 100 Hz keo dai 1.27 s, khong phai 1.28 s.
    assert (t[127] - t[0]) == pytest.approx(1.27)


def test_imu_noise_is_shared_across_overlapping_windows():
    """Hai cua so chong lan phai thay CUNG gia tri nhieu tai cung timestamp."""
    t = np.arange(2000) * 0.01
    u = np.zeros((2000, 6))
    c = ImuCorruptor(ImuCorruptionConfig(clean_probability=0.0))
    a = c(u[100:228], split="train", realization=0, trajectory="T", times=t, start=100, end=228)[0]
    b = c(u[110:238], split="train", realization=0, trajectory="T", times=t, start=110, end=238)[0]
    assert np.allclose(a[10:], b[:118])


def test_imu_random_walk_not_reset_per_window():
    t = np.arange(3000) * 0.01
    u = np.zeros((3000, 6))
    c = ImuCorruptor(ImuCorruptionConfig(clean_probability=0.0, bias_enabled=True, drift_enabled=True))
    trace, _ = c.trace("train", 0, "T", t)
    # Drift tich luy: do lech cuoi phai lon hon dau.
    assert np.abs(trace[-1, :3]).mean() != pytest.approx(np.abs(trace[0, :3]).mean(), rel=0.05)


def test_image_and_imu_severity_are_independent():
    """Hai nguon khong duoc gan bang mot severity scalar chung (muc 5.1)."""
    img_c = ImageCorruptor(ImageCorruptionConfig(), master_seed=42)
    imu_c = ImuCorruptor(ImuCorruptionConfig(), master_seed=42)
    t = np.arange(500) * 0.01
    sigmas, blurs = [], []
    for i in range(40):
        traj = f"Env/Data_easy/P{i:03d}"
        blurs.append(img_c.parameters("train", 0, traj, 0.5)["blur_sigma"])
        sigmas.append(imu_c.trace("train", 0, traj, t)[1]["sigma_sample"][0])
    corr = np.corrcoef(blurs, sigmas)[0, 1]
    assert abs(corr) < 0.5, f"tham so hai nguon tuong quan bat thuong: {corr:.3f}"


def test_clean_flag_returns_exact_input():
    """clean=true phai tra dung input, khong con blur nho (muc 17 G0)."""
    img_c = ImageCorruptor(ImageCorruptionConfig())
    x = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    out, params = img_c(x, split="valid", realization=0, trajectory="T",
                        image_time=0.5, image_index=0, force_clean=True)
    assert params["clean"] and np.array_equal(out, x)


def test_corruption_does_not_mutate_clean():
    img_c = ImageCorruptor(ImageCorruptionConfig(clean_probability=0.0))
    x = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    original = x.copy()
    img_c(x, split="train", realization=0, trajectory="T", image_time=0.5, image_index=0)
    assert np.array_equal(x, original)

    imu_c = ImuCorruptor(ImuCorruptionConfig(clean_probability=0.0))
    t = np.arange(200) * 0.01
    u = np.random.default_rng(1).normal(size=(128, 6))
    u_orig = u.copy()
    imu_c(u, split="train", realization=0, trajectory="T", times=t, start=0, end=128)
    assert np.array_equal(u, u_orig)


def test_image_optical_params_fixed_per_second_segment():
    c = ImageCorruptor(ImageCorruptionConfig())
    same = [c.parameters("train", 0, "T", tt)["blur_sigma"] for tt in (1.0, 1.4, 1.99)]
    other = c.parameters("train", 0, "T", 2.01)["blur_sigma"]
    assert len(set(same)) == 1 and other != same[0]


def test_validation_realization_is_epoch_independent():
    """Seed validation khong duoc chua epoch (muc 5.1)."""
    c = ImageCorruptor(ImageCorruptionConfig())
    a = c.parameters("valid", 0, "T", 1.0)
    b = c.parameters("valid", 0, "T", 1.0)
    assert a == b


def test_normalizer_roundtrip_and_floor():
    mean = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    std = torch.tensor([2.0, 2.0, 2.0, 1e-9, 0.5, 0.5])  # kenh 3 duoi floor
    n = ImuNormalizer(mean=mean, std=std)
    assert n.scale[3].item() == pytest.approx(1e-4)   # floor gyro
    u = torch.randn(2, 6, 128) * 3 + 1
    assert torch.allclose(n.denormalize(n.normalize(u)), u, atol=1e-4)


def test_normalizer_buffers_are_not_trainable():
    n = ImuNormalizer(mean=torch.zeros(6), std=torch.ones(6))
    assert list(n.parameters()) == []
    assert "mu" in dict(n.named_buffers()) and "scale" in dict(n.named_buffers())


def test_collate_transposes_imu_to_channels_first():
    from qjepa.data.dataset import collate

    batch = [
        {
            "image_clean": torch.rand(3, 8, 8), "image_bad": torch.rand(3, 8, 8),
            "imu_clean_phys": torch.arange(128 * 6, dtype=torch.float32).reshape(128, 6),
            "imu_bad_phys": torch.zeros(128, 6),
            "image_time": torch.tensor(1.0), "imu_times": torch.linspace(0, 1.27, 128),
            "imu_start": torch.tensor(0), "sample_id": "a", "trajectory_key": "T",
            "image_clean_flag": torch.tensor(False), "imu_clean_flag": torch.tensor(False),
        }
    ]
    out = collate(batch)
    assert out["imu_clean_phys"].shape == (1, 6, 128)
    # Hang 0 cua storage [128,6] phai thanh cot 0 cua [6,128], khong bi flatten sai.
    assert torch.allclose(out["imu_clean_phys"][0, :, 0], torch.arange(6, dtype=torch.float32))
    assert out["imu_times"].shape == (1, 128)
