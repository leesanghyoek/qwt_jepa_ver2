"""Bien doi co dinh: QWT 2D cho anh, Haar 1D cho IMU."""

from .layout import Layout
from .qwt import QuaternionWaveletTransform
from .haar import HaarTransform1D
from .haar_image import HaarImageBaseline

__all__ = [
    "Layout",
    "QuaternionWaveletTransform",
    "HaarTransform1D",
    "HaarImageBaseline",
    "build_image_transform",
    "build_imu_transform",
]


def build_image_transform(name: str, levels: int = 1):
    """Factory theo `transform.image`. Khong co fallback am tham (spec muc 6.3)."""
    if name == "qwt":
        return QuaternionWaveletTransform(levels=levels)
    if name == "dwt_haar_baseline":
        return HaarImageBaseline(levels=levels)
    raise ValueError(
        f"transform.image={name!r} khong hop le; chon 'qwt' (cau hinh muc tieu) "
        "hoac 'dwt_haar_baseline' (run baseline rieng, KHONG hoan thanh yeu cau QWT)"
    )


def build_imu_transform(name: str, levels: int = 1, channels: int = 6):
    if name == "dwt_haar_1d":
        return HaarTransform1D(levels=levels, channels=channels)
    raise ValueError(f"transform.imu={name!r} khong hop le; chi ho tro 'dwt_haar_1d'")
