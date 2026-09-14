"""Convolution blocks dung chung cho nhanh anh (2D) va nhanh IMU (1D).

Spec muc 8.1. Khong BatchNorm, dropout=0, GroupNorm(groups=8).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv(dim: int):
    return nn.Conv2d if dim == 2 else nn.Conv1d


def group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    if channels % groups:
        raise ValueError(
            f"channels={channels} khong chia het groups={groups}; "
            "spec muc 8.1 chon channels 32/64/96/128 de chia het 8"
        )
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    """Conv(k=3) -> GroupNorm -> SiLU."""

    def __init__(self, cin: int, cout: int, stride: int = 1, *, dim: int, groups: int = 8):
        super().__init__()
        self.conv = _conv(dim)(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = group_norm(cout, groups)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """y = SiLU(x + GN(Conv(SiLU(GN(Conv(x))))))."""

    def __init__(self, channels: int, *, dim: int, groups: int = 8):
        super().__init__()
        conv = _conv(dim)
        self.conv1 = conv(channels, channels, 3, 1, 1, bias=False)
        self.norm1 = group_norm(channels, groups)
        self.conv2 = conv(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = group_norm(channels, groups)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


class Stage(nn.Module):
    """ConvBlock + ResBlock, mot tang cua encoder/decoder."""

    def __init__(self, cin: int, cout: int, stride: int = 1, *, dim: int, groups: int = 8):
        super().__init__()
        self.block = ConvBlock(cin, cout, stride, dim=dim, groups=groups)
        self.res = ResBlock(cout, dim=dim, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.block(x))


def resize_to(x: torch.Tensor, size, *, dim: int) -> torch.Tensor:
    """Resize ve dung size cua skip that (spec muc 10): bilinear 2D, linear 1D."""
    mode = "bilinear" if dim == 2 else "linear"
    if tuple(x.shape[-dim:]) == tuple(size):
        return x
    return torch.nn.functional.interpolate(x, size=size, mode=mode, align_corners=False)
