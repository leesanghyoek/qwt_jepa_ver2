"""DWT Haar 1D mot level cho IMU (spec muc 6.4).

    A[n] = (u[2n] + u[2n+1]) / sqrt(2)
    D[n] = (u[2n] - u[2n+1]) / sqrt(2)

    u[2n]   = (A[n] + D[n]) / sqrt(2)
    u[2n+1] = (A[n] - D[n]) / sqrt(2)

Packing: [A_ax..A_gz, D_ax..D_gz], tuc [B,6,L] -> [B,12,L/2].

Day la DWT 1D, KHONG phai QWT. Truc cam bien khong phai mot truc khong gian
lien tuc nen khong ap QWT 2D len ma tran thoi gian x sau truc (spec muc 6.4).
"""

from __future__ import annotations

import math

import torch

from .layout import Layout, fp32_transform

BACKEND = "haar1d"
REVISION = "1.0.0"

_SQRT2 = math.sqrt(2.0)

BAND_ORDER = ("approx", "detail")
CHANNEL_ORDER = ("ax", "ay", "az", "gx", "gy", "gz")


class HaarTransform1D(torch.nn.Module):
    """Haar 1D theo truc thoi gian, tung kenh doc lap."""

    def __init__(self, levels: int = 1, channels: int = 6) -> None:
        super().__init__()
        if levels != 1:
            raise ValueError(f"levels={levels} chua ho tro; cau hinh chinh la mot level")
        self.levels = levels
        self.channels = channels

    @property
    def coeff_channels(self) -> int:
        return 2 * self.channels

    @fp32_transform
    def analysis(self, u: torch.Tensor) -> tuple[torch.Tensor, Layout]:
        """[B,6,L] -> ([B,12,L/2], Layout). L phai chan."""
        if u.dim() != 3 or u.shape[1] != self.channels:
            raise ValueError(f"can [B,{self.channels},L], nhan {tuple(u.shape)}")
        b, c, length = u.shape
        if length % 2:
            raise ValueError(
                f"Haar mot level can L chan, nhan L={length}. Cau hinh chinh L=128; "
                "khong pad hang gia de du L (spec muc 3.3)."
            )
        even, odd = u[..., 0::2], u[..., 1::2]
        approx = (even + odd) / _SQRT2
        detail = (even - odd) / _SQRT2
        packed = torch.cat([approx, detail], dim=1)
        layout = Layout(
            backend=BACKEND,
            revision=REVISION,
            original_shape=(b, c, length),
            coeff_shape=tuple(packed.shape),
            levels=self.levels,
            boundary_mode="none_even_length",
            scale_convention="orthonormal_sqrt2",
            band_order=BAND_ORDER,
            channel_order=CHANNEL_ORDER,
            extra={"pack_order": "approx_all_channels_then_detail_all_channels"},
        )
        return packed, layout

    @fp32_transform
    def synthesis(self, coeffs: torch.Tensor, layout: Layout) -> torch.Tensor:
        """([B,12,L/2], Layout) -> [B,6,L]. Differentiable theo coeffs."""
        layout.require(BACKEND, REVISION)
        b, c, length = layout.original_shape
        if coeffs.shape[1] != self.coeff_channels:
            raise ValueError(
                f"can {self.coeff_channels} kenh he so, nhan {coeffs.shape[1]}"
            )
        approx, detail = coeffs[:, :c], coeffs[:, c:]
        even = (approx + detail) / _SQRT2
        odd = (approx - detail) / _SQRT2
        out = torch.stack([even, odd], dim=-1).reshape(b, c, length)
        return out
