"""Decoder DANH GIA bieu dien tren backbone da dong bang (themjacobian muc 12).

Khac voi decoder phuc hoi chinh: decoder nay **chi nhan ZI/ZU** sau fusion va du
doan **he so TUYET DOI**, khong cong vao he so nhieu dau vao.

Bi cam nhan:
  - anh/IMU goc, he so wavelet nhieu (lam input hoac residual)
  - skip S0/S1/S2, V0/V1/V2 cua encoder
  - feature tu teacher/predictor, clean target, severity that
  - layout dong theo sample — chi dung cau hinh TINH de inverse dung shape

"Khong skip" o day nghia la khong co duong tat tu du lieu hay cac tang encoder
truoc; ConvBlock/ResBlock NOI BO cua chinh decoder van duoc phep.

Ten "probe decoder" o day la decoder DE DANH GIA — khac voi "input perturbation
probe" cua module encoder_sensitivity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage, resize_to


class _ProbeBranch(nn.Module):
    """Ba tang upsample + head he so tuyet doi, cho mot modality."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (96, 64, 32),
        *,
        dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        c2, c1, c0 = channels
        self.p2 = Stage(in_channels, c2, 1, dim=dim, groups=groups)
        self.p1 = Stage(c2, c1, 1, dim=dim, groups=groups)
        self.p0 = Stage(c1, c0, 1, dim=dim, groups=groups)
        conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.head = conv(c0, out_channels, 3, 1, 1)
        # He so TUYET DOI: khong zero-init (khong phai correction head).
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor, sizes: list) -> torch.Tensor:
        x = self.p2(resize_to(z, sizes[0], dim=self.dim))
        x = self.p1(resize_to(x, sizes[1], dim=self.dim))
        x = self.p0(resize_to(x, sizes[2], dim=self.dim))
        return self.head(x)


class RepresentationProbeDecoder(nn.Module):
    """Hai nhanh decoder danh gia, doc ZI/ZU roi inverse wavelet.

    Args:
        image_transform / imu_transform: transform CO DINH cua backbone (dung de
            synthesis; khong co tham so hoc).
        image_size: kich thuoc anh goc, de dung layout tinh.
        imu_window: so hang IMU L.
    """

    def __init__(
        self,
        image_transform: nn.Module,
        imu_transform: nn.Module,
        *,
        image_size: tuple[int, int] = (256, 256),
        imu_window: int = 128,
        feature_channels: int = 128,
        decoder_channels: tuple[int, ...] = (96, 64, 32),
        imu_channels: int = 6,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.image_transform = image_transform
        self.imu_transform = imu_transform
        self.image_size = tuple(image_size)
        self.imu_window = imu_window
        self.imu_channels = imu_channels

        h, w = self.image_size
        hc, wc = h // 2, w // 2                 # lich size suy tu kien truc TINH
        self.image_sizes = [(hc // 4, wc // 4), (hc // 2, wc // 2), (hc, wc)]
        lc = imu_window // 2
        self.imu_sizes = [(lc // 4,), (lc // 2,), (lc,)]

        self.image_branch = _ProbeBranch(
            feature_channels, image_transform.coeff_channels, decoder_channels,
            dim=2, groups=groups,
        )
        self.imu_branch = _ProbeBranch(
            feature_channels, imu_transform.coeff_channels, decoder_channels,
            dim=1, groups=groups,
        )

    def forward(self, zi: torch.Tensor, zu: torch.Tensor) -> dict[str, torch.Tensor]:
        """(ZI, ZU) -> anh phuc hoi va IMU normalized. Khong nhan gi khac."""
        if zi.ndim != 4 or zu.ndim != 3:
            raise ValueError(f"can ZI [B,D,H,W] va ZU [B,D,T], nhan {zi.shape}/{zu.shape}")
        if zi.shape[0] != zu.shape[0]:
            raise ValueError("batch cua ZI va ZU khac nhau")
        b = zi.shape[0]

        ci_hat = self.image_branch(zi, self.image_sizes)
        cu_hat = self.imu_branch(zu, self.imu_sizes)

        # Layout tinh: chi mo ta shape/filter, khong mang du lieu theo sample.
        _, image_layout = self.image_transform.analysis(
            torch.zeros(b, 3, *self.image_size, device=zi.device, dtype=zi.dtype)
        )
        _, imu_layout = self.imu_transform.analysis(
            torch.zeros(b, self.imu_channels, self.imu_window, device=zu.device, dtype=zu.dtype)
        )
        # He so TUYET DOI -> synthesis truc tiep, KHONG cong CI_bad/CU_bad.
        return {
            "image_hat": self.image_transform.synthesis(ci_hat, image_layout),
            "imu_hat_norm": self.imu_transform.synthesis(cu_hat, imu_layout),
            "image_coefficients": ci_hat,
            "imu_coefficients": cu_hat,
        }

    def state_hash(self) -> str:
        """Hash cua initial state — control va treatment phai dung chung."""
        import hashlib

        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
