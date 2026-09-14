"""Loss phuc hoi va loss du doan dac trung (spec muc 12.3, 13).

Anh o mien [0,1]; IMU o mien normalized. Moi modality duoc lay mean RIENG roi
moi cong, de 256 token anh khong tu chiem trong so gap 32 lan 8 token IMU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    image: float = 1.0
    imu: float = 1.0
    edge: float = 0.0
    imu_delta: float = 0.0
    variance: float = 0.0
    variance_gamma: float = 0.5
    smooth_l1_beta: float = 1.0


def layer_norm_no_affine(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm khong co affine parameters tren chieu cuoi (muc 12.3)."""
    return F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps)


def image_loss(image_hat: torch.Tensor, image_clean: torch.Tensor) -> torch.Tensor:
    """L1. Khong clamp output trong train loss (muc 10.3)."""
    return (image_hat - image_clean).abs().mean()


def imu_loss(
    imu_hat_norm: torch.Tensor, imu_clean_norm: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (L_imu, L_acc, L_gyro); can bang accel/gyro thay vi mean chung."""
    acc = F.smooth_l1_loss(imu_hat_norm[:, 0:3], imu_clean_norm[:, 0:3], beta=beta)
    gyro = F.smooth_l1_loss(imu_hat_norm[:, 3:6], imu_clean_norm[:, 3:6], beta=beta)
    return 0.5 * (acc + gyro), acc, gyro


def edge_loss(image_hat: torch.Tensor, image_clean: torch.Tensor) -> torch.Tensor:
    """L1 giua sai phan huu han ngang/doc (muc 13, mac dinh weight 0)."""
    dx = (image_hat[..., :, 1:] - image_hat[..., :, :-1]) - (
        image_clean[..., :, 1:] - image_clean[..., :, :-1]
    )
    dy = (image_hat[..., 1:, :] - image_hat[..., :-1, :]) - (
        image_clean[..., 1:, :] - image_clean[..., :-1, :]
    )
    return dx.abs().mean() + dy.abs().mean()


def imu_delta_loss(
    imu_hat_norm: torch.Tensor, imu_clean_norm: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """SmoothL1 giua HIEU mau lien nhau cua output va clean - khong phat do lon."""
    return F.smooth_l1_loss(
        torch.diff(imu_hat_norm, dim=-1), torch.diff(imu_clean_norm, dim=-1), beta=beta
    )


def variance_loss(tokens: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    """L_var = mean(relu(gamma - std qua B sample tai cung (n,d)))  (muc 12.4).

    Chi co y nghia khi batch THAT du da dang; gradient accumulation khong lam
    batch statistics lon hon.
    """
    if tokens.shape[0] < 4:
        return tokens.new_zeros(())
    std = tokens.std(dim=0)
    return F.relu(gamma - std).mean()


def jepa_loss(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    pred_imu: torch.Tensor,
    target_imu: torch.Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """0.5 * (JI + JU) voi target da stop-gradient (muc 12.3)."""
    ji = F.smooth_l1_loss(
        layer_norm_no_affine(pred_image, eps),
        layer_norm_no_affine(target_image.detach(), eps),
        beta=beta,
    )
    ju = F.smooth_l1_loss(
        layer_norm_no_affine(pred_imu, eps),
        layer_norm_no_affine(target_imu.detach(), eps),
        beta=beta,
    )
    return 0.5 * (ji + ju), ji, ju


def jepa_weight(
    local_step: int, ramp_updates: int, start: float = 0.01, maximum: float = 0.10
) -> float:
    """Ramp tuyen tinh trong Stage B roi giu maximum (muc 14.1)."""
    p = min(local_step / max(ramp_updates - 1, 1), 1.0)
    return start + (maximum - start) * p
