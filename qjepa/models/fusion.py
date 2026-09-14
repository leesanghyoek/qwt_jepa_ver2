"""Shared gated MLP fusion giua nhanh anh va nhanh IMU (spec muc 9).

Trao doi thong tin qua mot summary toan cuc `s`, roi dieu tiet dense feature
cua TUNG nhanh bang cong co hoc. Dense features va skips cua moi nhanh van
duoc giu nguyen duong rieng, nen moi nhanh con phuc hoi duoc khi nguon kia
it huu ich.

Gate KHONG phai xac suat "cam bien dang tin cay" da calibration: no la trong
so dac trung hoc tu loss, khong nhan severity ground truth (spec muc 9.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedGatedFusion(nn.Module):
    """MLP dung chung + hai duong dieu tiet rieng.

    Args:
        dim: embedding dimension D (mac dinh 128).
        hidden: chieu an cua shared MLP (mac dinh 256).
        imu_summary_bins: so bin pooling tren FU (mac dinh 4).
        time_metadata_dim: so chieu metadata thoi gian (mac dinh 3).
        cross_modal: True dung ca hai nguon; False mask dung cach theo muc 9.3.
        gate_bias_init: bias khoi tao cua hai gate head (mac dinh -2.0).
    """

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 256,
        *,
        imu_summary_bins: int = 4,
        time_metadata_dim: int = 3,
        cross_modal: bool = True,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.bins = imu_summary_bins
        self.time_dim = time_metadata_dim
        self.cross_modal = cross_modal

        self.norm_image_summary = nn.LayerNorm(dim)
        self.imu_pool = nn.AdaptiveAvgPool1d(imu_summary_bins)
        self.imu_summary_proj = nn.Linear(dim * imu_summary_bins, dim)
        self.norm_imu_summary = nn.LayerNorm(dim)

        self.in_dim = 2 * dim + time_metadata_dim  # 259 voi D=128
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.norm_summary = nn.LayerNorm(dim)

        self.gate_image = nn.Linear(self.in_dim, dim)
        self.gate_imu = nn.Linear(self.in_dim, dim)
        for gate in (self.gate_image, self.gate_imu):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_bias_init)

        self.delta_image = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.delta_imu = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        )

    def summaries(self, fi: torch.Tensor, fu: torch.Tensor):
        """Tao gI va gU rieng cho tung nhanh (spec muc 9.1)."""
        gi = self.norm_image_summary(fi.mean(dim=(-2, -1)))
        pooled = self.imu_pool(fu).flatten(1)
        gu = self.norm_imu_summary(self.imu_summary_proj(pooled))
        return gi, gu

    def forward(
        self,
        fi: torch.Tensor,
        fu: torch.Tensor,
        meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(FI, FU, m) -> (FI_fused, FU_fused)."""
        if meta.shape[-1] != self.time_dim:
            raise ValueError(f"metadata can {self.time_dim} chieu, nhan {meta.shape[-1]}")
        gi, gu = self.summaries(fi, fu)

        if self.cross_modal:
            v_image = v_imu = torch.cat([gi, gu, meta], dim=-1)
        else:
            # Muc 9.3: mask CA summary lan gate; metadata thoi gian duoc phep giu.
            zero_i, zero_u = torch.zeros_like(gi), torch.zeros_like(gu)
            v_image = torch.cat([gi, zero_u, meta], dim=-1)
            v_imu = torch.cat([zero_i, gu, meta], dim=-1)

        s_image = self.norm_summary(self.shared_mlp(v_image))
        s_imu = s_image if self.cross_modal else self.norm_summary(self.shared_mlp(v_imu))

        gate_i = torch.sigmoid(self.gate_image(v_image))
        gate_u = torch.sigmoid(self.gate_imu(v_imu))

        s_i = s_image[..., None, None].expand(-1, -1, *fi.shape[-2:])
        di = self.delta_image(torch.cat([fi, s_i], dim=1))
        fi_fused = fi + gate_i[..., None, None] * di

        s_u = s_imu[..., None].expand(-1, -1, fu.shape[-1])
        du = self.delta_imu(torch.cat([fu, s_u], dim=1))
        fu_fused = fu + gate_u[..., None] * du

        return fi_fused, fu_fused


def build_time_metadata(image_time: torch.Tensor, imu_times: torch.Tensor) -> torch.Tensor:
    """Metadata thoi gian tuong doi [B,3] (spec muc 4.4).

        m = [ (t_image - giua cua so) / T,  log(T / 1s),  log(dt_median / 0.01s) ]

    Reject timestamp loi thay vi giau bang epsilon.
    """
    if imu_times.dim() != 2:
        raise ValueError(f"imu_times can [B,L], nhan {tuple(imu_times.shape)}")
    t0, t1 = imu_times[:, 0], imu_times[:, -1]
    span = t1 - t0
    if not torch.all(span > 0):
        raise ValueError("imu_times khong tang: span <= 0 o it nhat mot sample")
    dt_median = torch.median(torch.diff(imu_times, dim=-1), dim=-1).values
    if not torch.all(dt_median > 0):
        raise ValueError("dt median <= 0 o it nhat mot sample")
    centre = (t0 + t1) / 2
    return torch.stack(
        [
            (image_time - centre) / span,
            torch.log(span / 1.0),
            torch.log(dt_median / 0.01),
        ],
        dim=-1,
    ).to(torch.float32)
