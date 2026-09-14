"""Decoder du doan DELTA COEFFICIENTS cho tung nhanh (spec muc 10).

Duong phuc hoi la residual tren he so nhieu:

    C_hat = C_bad + delta_C
    output = transform.synthesis(C_hat, layout)

Head cuoi duoc zero-init nen model luc khoi tao tra gan dung inverse cua he so
nhieu (gan identity). Khong ReLU/sigmoid/clamp tren he so hoac delta.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage, resize_to


class CoefficientDecoder(nn.Module):
    """Upsample ve size THAT cua skip, concat, roi du doan delta coefficients.

    Args:
        out_channels: so kenh he so wavelet phai tra ve.
        channels: cung bon muc channels voi encoder.
        dim: 2 cho anh, 1 cho IMU.
    """

    def __init__(
        self,
        out_channels: int,
        channels: tuple[int, ...] = (32, 64, 96, 128),
        *,
        dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        c0, c1, c2, c3 = channels
        self.up2 = Stage(c3 + c2, c2, 1, dim=dim, groups=groups)
        self.up1 = Stage(c2 + c1, c1, 1, dim=dim, groups=groups)
        self.up0 = Stage(c1 + c0, c0, 1, dim=dim, groups=groups)
        conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.head = conv(c0, out_channels, 3, 1, 1)
        # Muc 10.3: zero-init hai coefficient head cuoi.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, fused: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        s0, s1, s2 = skips
        x = resize_to(fused, s2.shape[-self.dim :], dim=self.dim)
        x = self.up2(torch.cat([x, s2], dim=1))
        x = resize_to(x, s1.shape[-self.dim :], dim=self.dim)
        x = self.up1(torch.cat([x, s1], dim=1))
        x = resize_to(x, s0.shape[-self.dim :], dim=self.dim)
        x = self.up0(torch.cat([x, s0], dim=1))
        return self.head(x)


def image_decoder(out_channels: int, channels=(32, 64, 96, 128), groups: int = 8):
    return CoefficientDecoder(out_channels, channels, dim=2, groups=groups)


def imu_decoder(out_channels: int = 12, channels=(32, 64, 96, 128), groups: int = 8):
    return CoefficientDecoder(out_channels, channels, dim=1, groups=groups)


class LatentPredictor(nn.Module):
    """Predictor JEPA tren chieu cuoi (spec muc 12.1).

        P(z) = Linear(H, D)(GELU(Linear(D, H)(LayerNorm(z))))

    Hoat dong theo token; so token/vi tri khong doi. Du doan LATENT, khong
    phai pixel hay gia tri gyro.
    """

    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
