"""Suy giam anh (spec muc 5.2).

Thu tu: clean 256x256 -> Gaussian blur -> downsample/upsample tuy chon
-> Gaussian noise -> clip [0,1].

Tham so quang hoc va clean flag co dinh theo doan camera mot giay; noise
realization rieng tung frame. Khong tao blur phu thuoc gyro - hai nguon phai
doc lap (spec muc 1, 5.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rng import generator


@dataclass(frozen=True)
class ImageCorruptionConfig:
    blur_sigma_px: tuple[float, float] = (0.3, 1.0)
    gaussian_std: tuple[float, float] = (2 / 255, 8 / 255)
    downsample_probability: float = 0.0
    downsample_scale: tuple[float, float] = (0.5, 1.0)
    jpeg_probability: float = 0.0
    clean_probability: float = 0.1
    segment_seconds: float = 1.0


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(math.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur tach duoc, reflect padding. sigma=0 la identity."""
    if sigma <= 0:
        return img
    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    out = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="reflect")
    out = np.tensordot(k, np.stack([out[i : i + img.shape[0]] for i in range(len(k))]), axes=(0, 0))
    out = np.pad(out, ((0, 0), (r, r), (0, 0)), mode="reflect")
    out = np.tensordot(k, np.stack([out[:, i : i + img.shape[1]] for i in range(len(k))]), axes=(0, 0))
    return out


def _resample(img: np.ndarray, scale: float) -> np.ndarray:
    """Downsample roi upsample lai 256 bang bilinear (khong dung thu vien anh)."""
    h, w = img.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))

    def _interp(src: np.ndarray, out_len: int, axis: int) -> np.ndarray:
        in_len = src.shape[axis]
        pos = (np.arange(out_len) + 0.5) * in_len / out_len - 0.5
        pos = np.clip(pos, 0, in_len - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, in_len - 1)
        frac = (pos - lo).reshape([-1 if i == axis else 1 for i in range(src.ndim)])
        return np.take(src, lo, axis=axis) * (1 - frac) + np.take(src, hi, axis=axis) * frac

    small = _interp(_interp(img, nh, 0), nw, 1)
    return _interp(_interp(small, h, 0), w, 1)


class ImageCorruptor:
    """Sinh anh bad tu anh clean, tai lap duoc theo (trajectory, frame)."""

    def __init__(self, config: ImageCorruptionConfig, master_seed: int = 42) -> None:
        self.cfg = config
        self.master_seed = master_seed

    def segment_index(self, image_time: float) -> int:
        return int(math.floor(image_time / self.cfg.segment_seconds))

    def parameters(self, split: str, realization: int, trajectory: str, image_time: float) -> dict:
        """Tham so quang hoc + clean flag, co dinh theo doan mot giay."""
        seg = self.segment_index(image_time)
        rng = generator(self.master_seed, split, realization, trajectory, "image_segment", seg)
        clean = bool(rng.random() < self.cfg.clean_probability)
        return {
            "clean": clean,
            "segment": seg,
            "blur_sigma": float(rng.uniform(*self.cfg.blur_sigma_px)),
            "noise_std": float(rng.uniform(*self.cfg.gaussian_std)),
            "downsample": bool(rng.random() < self.cfg.downsample_probability),
            "downsample_scale": float(rng.uniform(*self.cfg.downsample_scale)),
        }

    def __call__(
        self,
        image_clean: np.ndarray,
        *,
        split: str,
        realization: int,
        trajectory: str,
        image_time: float,
        image_index: int,
        force_clean: bool | None = None,
    ) -> tuple[np.ndarray, dict]:
        """image_clean [H,W,3] float32 trong [0,1] -> (image_bad, params)."""
        params = self.parameters(split, realization, trajectory, image_time)
        if force_clean is not None:
            params = {**params, "clean": force_clean}
        if params["clean"]:
            return image_clean.copy(), params

        img = image_clean.astype(np.float64)
        img = _blur(img, params["blur_sigma"])
        if params["downsample"] and params["downsample_scale"] < 1.0:
            img = _resample(img, params["downsample_scale"])
        rng = generator(self.master_seed, split, realization, trajectory, "image_noise", image_index)
        img = img + rng.normal(0.0, params["noise_std"], size=img.shape)
        return np.clip(img, 0.0, 1.0).astype(np.float32), params
