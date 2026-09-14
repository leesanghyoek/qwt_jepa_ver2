"""Layout mo ta mot lan analysis: du de synthesis cat lai dung shape goc.

Spec muc 6.1: layout phai chua original shape, padding, boundary mode, level,
band/component order, filter/backend identifier va scale convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Layout:
    """Metadata bat buoc di kem coefficients."""

    backend: str
    """Dinh danh backend, vi du 'qwt_dualtree_db4' hoac 'haar1d'."""

    revision: str
    """Revision cua implementation, ghi vao checkpoint de phat hien mismatch."""

    original_shape: tuple[int, ...]
    """Shape cua tensor dau vao truoc padding."""

    coeff_shape: tuple[int, ...]
    """Shape cua packed coefficients tra ve boi analysis()."""

    levels: int
    boundary_mode: str
    scale_convention: str

    band_order: tuple[str, ...] = ()
    """Thu tu band_group, vi du ('approx', 'detail_1', 'detail_2', 'detail_3')."""

    component_order: tuple[str, ...] = ()
    """Thu tu quaternion component, vi du ('real', 'i', 'j', 'k')."""

    channel_order: tuple[str, ...] = ()
    """Thu tu kenh nguon, vi du ('R', 'G', 'B')."""

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require(self, backend: str, revision: str | None = None) -> None:
        """Raise neu layout khong den tu backend mong doi."""
        if self.backend != backend:
            raise ValueError(f"layout backend {self.backend!r}, can {backend!r}")
        if revision is not None and self.revision != revision:
            raise ValueError(
                f"layout revision {self.revision!r}, can {revision!r}; "
                "checkpoint va code khong khop"
            )


def fp32_transform(method):
    """Chay analysis/synthesis o FP32 ngay ca khi AMP dang bat (spec muc 10.3).

    Wavelet la phep bien doi co dinh; chay no o fp16/bf16 lam hong do chinh xac
    round-trip ma khong tiet kiem dang ke. Cast la mot phep doi dtype thuong,
    KHONG detach, nen gradient van di qua binh thuong.
    """
    import functools

    import torch

    @functools.wraps(method)
    def wrapper(self, x, *args, **kwargs):
        # Chi nang fp16/bf16 len fp32; giu nguyen fp32 va fp64 (gradcheck dung
        # fp64 va phai giu duoc do chinh xac do).
        promoted = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        with torch.autocast(x.device.type, enabled=False):
            return method(self, promoted, *args, **kwargs)

    return wrapper
