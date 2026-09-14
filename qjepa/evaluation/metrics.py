"""Metric anh va IMU (spec muc 18).

Anh: MAE, PSNR (data_range=1), SSIM theo protocol Gaussian 11x11 sigma=1.5.
IMU: MAE/RMSE tung truc va tung triplet, trong don vi vat ly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype or torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    window: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """SSIM tung kenh RGB, chi vung valid, trung binh kenh/vi tri (muc 18.2)."""
    if a.shape != b.shape or a.dim() != 4:
        raise ValueError(f"can hai tensor [B,C,H,W] cung shape, nhan {a.shape}/{b.shape}")
    c = a.shape[1]
    win = _gaussian_window(window, sigma, a.device, a.dtype).expand(c, 1, window, window)
    mu_a = F.conv2d(a, win, groups=c)
    mu_b = F.conv2d(b, win, groups=c)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    # Population moments, dung cung cua so Gaussian (weighted).
    sigma_aa = F.conv2d(a * a, win, groups=c) - mu_aa
    sigma_bb = F.conv2d(b * b, win, groups=c) - mu_bb
    sigma_ab = F.conv2d(a * b, win, groups=c) - mu_ab
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    return (num / den).mean()


def image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    clamp: bool = True,
    border_crop: int = 0,
) -> dict[str, float]:
    """MAE/MSE/PSNR/SSIM. PSNR tinh moi frame roi trung binh (muc 18.2)."""
    raw_mae = (pred - target).abs().mean().item()
    out_of_range = ((pred < 0) | (pred > data_range)).float().mean().item()
    if clamp:
        pred = pred.clamp(0.0, data_range)
    if border_crop:
        b = border_crop
        pred, target = pred[..., b:-b, b:-b], target[..., b:-b, b:-b]

    mse_per_frame = ((pred - target) ** 2).flatten(1).mean(dim=1)
    finite = mse_per_frame > 0
    psnr_per_frame = torch.where(
        finite,
        10 * torch.log10(data_range**2 / mse_per_frame.clamp_min(1e-20)),
        torch.full_like(mse_per_frame, float("inf")),
    )
    return {
        "mae": (pred - target).abs().mean().item(),
        "raw_mae": raw_mae,
        "mse": mse_per_frame.mean().item(),
        "psnr": psnr_per_frame[finite].mean().item() if finite.any() else float("inf"),
        "psnr_infinite_frames": int((~finite).sum().item()),
        "ssim": ssim(pred, target, data_range=data_range).item(),
        "out_of_range_fraction": out_of_range,
    }


def imu_metrics(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> dict[str, float]:
    """MAE/RMSE theo truc va theo triplet, trong m/s^2 va rad/s (muc 18.4)."""
    if pred_phys.shape != target_phys.shape or pred_phys.dim() != 3:
        raise ValueError("can [B,6,L] cung shape")
    err = pred_phys - target_phys
    axes = ("ax", "ay", "az", "gx", "gy", "gz")
    out: dict[str, float] = {}
    for i, name in enumerate(axes):
        e = err[:, i]
        out[f"mae_{name}"] = e.abs().mean().item()
        out[f"rmse_{name}"] = e.pow(2).mean().sqrt().item()
    out["mse_accel"] = err[:, 0:3].pow(2).mean().item()
    out["mse_gyro"] = err[:, 3:6].pow(2).mean().item()
    out["rmse_accel"] = math.sqrt(out["mse_accel"])
    out["rmse_gyro"] = math.sqrt(out["mse_gyro"])
    out["mae_accel"] = err[:, 0:3].abs().mean().item()
    out["mae_gyro"] = err[:, 3:6].abs().mean().item()
    return out


def error_reduction(mse_restored: float, mse_bad: float, floor: float = 1e-12) -> float | None:
    """1 - MSE_restored/MSE_bad. Tra None khi baseline qua nho (muc 18.4)."""
    if mse_bad <= floor:
        return None
    return 1.0 - mse_restored / mse_bad
