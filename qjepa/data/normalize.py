"""Chuan hoa IMU theo tung kenh bang thong ke train clean (spec muc 4.3).

    U_norm = (U_phys - mu) / scale
    U_phys = U_norm * scale + mu

`scale = max(train_std, floor)`; floor chi de on dinh so hoc, KHONG phai muc
nhieu. mu/scale la buffer khong train, di theo checkpoint va export.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_STD_FLOOR: tuple[float, ...] = (1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4)


class ImuNormalizer(nn.Module):
    """Buffers mu/scale cho sau kenh [ax, ay, az, gx, gy, gz]."""

    def __init__(
        self,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        std_floor: tuple[float, ...] = DEFAULT_STD_FLOOR,
        channels: int = 6,
    ) -> None:
        super().__init__()
        self.channels = channels
        floor = torch.tensor(std_floor, dtype=torch.float32)
        mu = torch.zeros(channels) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        sd = torch.ones(channels) if std is None else torch.as_tensor(std, dtype=torch.float32)
        if mu.numel() != channels or sd.numel() != channels:
            raise ValueError(f"mean/std phai co {channels} phan tu")
        self.register_buffer("mu", mu)
        self.register_buffer("scale", torch.maximum(sd, floor))
        self.register_buffer("std_floor", floor)
        self.register_buffer("fitted", torch.tensor(mean is not None))

    def normalize(self, u_phys: torch.Tensor) -> torch.Tensor:
        """[B,6,L] vat ly -> normalized."""
        return (u_phys - self.mu[None, :, None]) / self.scale[None, :, None]

    def denormalize(self, u_norm: torch.Tensor) -> torch.Tensor:
        """[B,6,L] normalized -> vat ly."""
        return u_norm * self.scale[None, :, None] + self.mu[None, :, None]

    def extra_repr(self) -> str:
        return f"mu={self.mu.tolist()}, scale={self.scale.tolist()}"
