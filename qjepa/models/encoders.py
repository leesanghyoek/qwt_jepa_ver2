"""Hai encoder CNN rieng: CNN2D cho anh mot frame, CNN1D cho IMU.

Spec muc 8.2 va 8.3. Khong temporal mixer, khong reshape batch x time.
Encoder tra ve (dense_features, skips); teacher chi lay dense features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage

DEFAULT_CHANNELS: tuple[int, ...] = (32, 64, 96, 128)


class CoefficientEncoder(nn.Module):
    """Encoder chung cho 1D/2D: mot stage stride 1 roi ba stage stride 2.

    Args:
        in_channels: so kenh he so wavelet (QWT 48, Haar anh 12, Haar IMU 12).
        channels: bon muc channels, mac dinh (32, 64, 96, 128).
        dim: 2 cho anh, 1 cho IMU.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        *,
        dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"can dung 4 muc channels, nhan {channels}")
        self.dim = dim
        self.channels = tuple(channels)
        c0, c1, c2, c3 = channels
        self.stage0 = Stage(in_channels, c0, 1, dim=dim, groups=groups)
        self.stage1 = Stage(c0, c1, 2, dim=dim, groups=groups)
        self.stage2 = Stage(c1, c2, 2, dim=dim, groups=groups)
        self.stage3 = Stage(c2, c3, 2, dim=dim, groups=groups)

    @property
    def out_channels(self) -> int:
        return self.channels[-1]

    def forward(self, coeffs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """coefficients -> (dense features, [s0, s1, s2])."""
        s0 = self.stage0(coeffs)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s3, [s0, s1, s2]


def image_encoder(in_channels: int, channels=DEFAULT_CHANNELS, groups: int = 8):
    """[B,Ccoeff,Hc,Wc] -> [B,128,Hc/8,Wc/8] va ba skips."""
    return CoefficientEncoder(in_channels, channels, dim=2, groups=groups)


def imu_encoder(in_channels: int = 12, channels=DEFAULT_CHANNELS, groups: int = 8):
    """[B,12,L/2] -> [B,128,L/16] va ba skips."""
    return CoefficientEncoder(in_channels, channels, dim=1, groups=groups)
