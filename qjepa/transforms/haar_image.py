"""Haar 2D mot level cho anh - BASELINE debug, khong phai QWT (spec muc 6.3).

Tra ve [B,12,H/2,W/2]: bon subband x ba kenh RGB. Dung de kiem tra loader,
CNN, fusion, JEPA, decoder va training khi QWT chua qua gate. Run dung
transform nay PHAI ghi ro trong run name/checkpoint/report va KHONG duoc coi
la hoan thanh yeu cau QWT.
"""

from __future__ import annotations

import math

import torch

from .layout import Layout, fp32_transform

BACKEND = "dwt_haar_baseline"
REVISION = "1.0.0"

_SQRT2 = math.sqrt(2.0)
BAND_ORDER = ("approx", "detail_1", "detail_2", "detail_3")
CHANNEL_ORDER = ("R", "G", "B")


class HaarImageBaseline(torch.nn.Module):
    """Haar 2D tung kenh RGB. Baseline ky thuat co ten ro rang."""

    def __init__(self, levels: int = 1) -> None:
        super().__init__()
        if levels != 1:
            raise ValueError(f"levels={levels} chua ho tro")
        self.levels = levels

    @property
    def coeff_channels(self) -> int:
        return 3 * len(BAND_ORDER)

    @fp32_transform
    def analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, Layout]:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"can [B,3,H,W], nhan {tuple(x.shape)}")
        b, _, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(f"can H,W chan, nhan {h}x{w}")
        lo_x = (x[..., 0::2] + x[..., 1::2]) / _SQRT2
        hi_x = (x[..., 0::2] - x[..., 1::2]) / _SQRT2
        ll = (lo_x[..., 0::2, :] + lo_x[..., 1::2, :]) / _SQRT2
        lh = (lo_x[..., 0::2, :] - lo_x[..., 1::2, :]) / _SQRT2
        hl = (hi_x[..., 0::2, :] + hi_x[..., 1::2, :]) / _SQRT2
        hh = (hi_x[..., 0::2, :] - hi_x[..., 1::2, :]) / _SQRT2
        packed = torch.stack([ll, lh, hl, hh], dim=2).reshape(b, 12, h // 2, w // 2)
        layout = Layout(
            backend=BACKEND,
            revision=REVISION,
            original_shape=(b, 3, h, w),
            coeff_shape=tuple(packed.shape),
            levels=self.levels,
            boundary_mode="none_even_size",
            scale_convention="orthonormal_sqrt2",
            band_order=BAND_ORDER,
            channel_order=CHANNEL_ORDER,
            extra={"note": "baseline debug, khong phai QWT"},
        )
        return packed, layout

    @fp32_transform
    def synthesis(self, coeffs: torch.Tensor, layout: Layout) -> torch.Tensor:
        layout.require(BACKEND, REVISION)
        b, _, h, w = layout.original_shape
        c = coeffs.reshape(b, 3, 4, coeffs.shape[-2], coeffs.shape[-1])
        ll, lh, hl, hh = c[:, :, 0], c[:, :, 1], c[:, :, 2], c[:, :, 3]
        lo_e, lo_o = (ll + lh) / _SQRT2, (ll - lh) / _SQRT2
        hi_e, hi_o = (hl + hh) / _SQRT2, (hl - hh) / _SQRT2
        lo_x = torch.stack([lo_e, lo_o], dim=-2).reshape(b, 3, h, w // 2)
        hi_x = torch.stack([hi_e, hi_o], dim=-2).reshape(b, 3, h, w // 2)
        x_e, x_o = (lo_x + hi_x) / _SQRT2, (lo_x - hi_x) / _SQRT2
        return torch.stack([x_e, x_o], dim=-1).reshape(b, 3, h, w)
