"""Quaternion Wavelet Transform 2D dual-tree cho anh RGB (spec muc 6.2).

Cau truc
--------
QWT dual-tree duoc dung theo nghia cua Chan, Choi, Baraniuk, "Coherent
Multiscale Image Processing Using Dual-Tree Quaternion Wavelets", IEEE TIP
17(7), 2008: hai cay filter bank lech nhau nua mau tao thanh mot cap xap xi
Hilbert; tich separable cua hai cay theo hai truc cho BON thanh phan
quaternion.

    component (w, i, j, k)  <->  (tree_x, tree_y) = (a,a), (b,a), (a,b), (b,b)

    w = real           : khong Hilbert
    i                  : Hilbert theo truc x (chieu rong)
    j                  : Hilbert theo truc y (chieu cao)
    k                  : Hilbert theo ca hai truc

Moi to hop cay la mot DWT separable 2D day du, cho bon subband:

    band_group = (approx, detail_1, detail_2, detail_3) = (LL, LH, HL, HH)

trong do LH = thap theo x / cao theo y, HL = cao theo x / thap theo y.

Cap Hilbert o MOT level duoc tao bang cach dich dau vao mot mau giua hai cay
(Selesnick, Baraniuk, Kingsbury, "The Dual-Tree Complex Wavelet Transform",
IEEE Signal Processing Magazine 22(6), 2005, muc "first stage"). Sau khi
downsample 2, mot mau dau vao tuong ung nua mau tren luoi he so - dung do tre
nua mau ma cap Hilbert yeu cau.

Tinh nguoc
----------
Bo loc la Daubechies db4 truc chuan, extension tuan hoan, nen MOI to hop cay
la mot phep bien doi truc giao: synthesis cua no dung bang adjoint cua
analysis va tai tao chinh xac. Bien doi tong hop du thua 4 lan; synthesis lay
trung binh bon tai tao, nen round-trip chinh xac tren MOI dau vao.

Vi du thua, `analysis(synthesis(c)) == c` KHONG dung cho c tuy y - dung nhu
spec muc 6.2 da ghi. Gate bat buoc la round-trip tren du lieu dau vao va
gradient cua synthesis.

Packing
-------
    [B, 3, H, W] -> [B, 3, 4, 4, Hc, Wc] -> [B, 48, Hc, Wc]
    axes:           batch, RGB, band_group, component, height, width
    channel = ((rgb * 4) + band) * 4 + component

RGB duoc bien doi RIENG tung kenh. Khong dong goi RGB thanh mot quaternion
mau - do la mot ho bien doi khac, xem spec muc 6.2.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .layout import Layout, fp32_transform

BACKEND = "qwt_dualtree_db4"
REVISION = "1.0.0"

BAND_ORDER = ("approx", "detail_1", "detail_2", "detail_3")
COMPONENT_ORDER = ("real", "i", "j", "k")
CHANNEL_ORDER = ("R", "G", "B")

# Daubechies db4 (4 vanishing moments, 8 taps), chuan hoa sum(h0) = sqrt(2).
# Nguon: I. Daubechies, "Ten Lectures on Wavelets", SIAM 1992, Table 6.1;
# trung voi pywt.Wavelet('db4').dec_lo dao nguoc. Test kiem tra truc chuan
# bang so thay vi tin vao hang so chep tay.
DB4_H0: tuple[float, ...] = (
    0.230377813308855230,
    0.714846570552541500,
    0.630880767929590400,
    -0.027983769416983850,
    -0.187034811718881140,
    0.030841381835986965,
    0.032883011666982945,
    -0.010597401784997278,
)


def _qmf(h0: torch.Tensor) -> torch.Tensor:
    """Bo loc thong cao truc giao: h1[n] = (-1)^n * h0[F-1-n]."""
    f = h0.numel()
    sign = torch.tensor([(-1.0) ** n for n in range(f)], dtype=h0.dtype)
    return sign * h0.flip(0)


def _analysis_1d(x: torch.Tensor, filt: torch.Tensor, offset: int) -> torch.Tensor:
    """Loc + downsample 2 doc theo chieu cuoi, extension tuan hoan.

    a[n] = sum_k filt[k] * x[(2n + offset + k) mod N]

    Args:
        x: [..., N], N chan.
        filt: [F].
        offset: 0 cho cay a, 1 cho cay b.

    Returns:
        [..., N // 2]
    """
    n = x.shape[-1]
    f = filt.numel()
    shape = x.shape
    flat = x.reshape(-1, 1, n)

    # Chi so lon nhat can doc la 2*(N/2-1) + offset + F-1 = N - 2 + offset + F - 1.
    need = offset + f - 1
    reps = math.ceil(need / n) + 1
    padded = flat.repeat(1, 1, reps + 1)[..., : n + need]
    padded = padded[..., offset:] if offset else padded

    out = F.conv1d(padded, filt.view(1, 1, f), stride=2)
    return out.reshape(*shape[:-1], n // 2)


def _synthesis_1d(a: torch.Tensor, filt: torch.Tensor, offset: int, n: int) -> torch.Tensor:
    """Adjoint chinh xac cua `_analysis_1d`, cong don theo modulo N.

    x[m] += sum_n a[n] * filt[(m - 2n - offset) mod N]
    """
    f = filt.numel()
    shape = a.shape
    flat = a.reshape(-1, 1, shape[-1])

    # conv_transpose1d la adjoint cua conv1d cung weight/stride.
    wide = F.conv_transpose1d(flat, filt.view(1, 1, f), stride=2)
    wide = wide[..., : shape[-1] * 2 + f - 1]

    # Gap phan tran ve theo modulo N, bu lai offset da bo khi analysis.
    total = wide.shape[-1]
    out = wide.new_zeros(wide.shape[0], 1, n)
    for start in range(0, total, n):
        chunk = wide[..., start : start + n]
        idx = (torch.arange(chunk.shape[-1], device=a.device) + start + offset) % n
        out.index_add_(-1, idx, chunk)
    return out.reshape(*shape[:-1], n)


def _analysis_axis(x: torch.Tensor, h0, h1, offset: int, axis: int):
    """Tach thap/cao doc theo `axis` (-1 hoac -2)."""
    if axis == -2:
        x = x.transpose(-1, -2)
    lo = _analysis_1d(x, h0, offset)
    hi = _analysis_1d(x, h1, offset)
    if axis == -2:
        lo, hi = lo.transpose(-1, -2), hi.transpose(-1, -2)
    return lo, hi


def _synthesis_axis(lo, hi, h0, h1, offset: int, n: int, axis: int):
    if axis == -2:
        lo, hi = lo.transpose(-1, -2), hi.transpose(-1, -2)
    out = _synthesis_1d(lo, h0, offset, n) + _synthesis_1d(hi, h1, offset, n)
    if axis == -2:
        out = out.transpose(-1, -2)
    return out


class QuaternionWaveletTransform(torch.nn.Module):
    """QWT 2D dual-tree mot level, ap dung rieng tung kenh RGB.

    Bo loc duoc giu lam buffer co dinh: khong nam trong optimizer, di theo
    checkpoint, va o cung device voi tensor dau vao.
    """

    def __init__(self, levels: int = 1) -> None:
        super().__init__()
        if levels != 1:
            raise ValueError(
                f"levels={levels} chua duoc ho tro; cau hinh chinh la mot level "
                "(spec muc 16: transform.image_levels = 1)"
            )
        self.levels = levels
        h0 = torch.tensor(DB4_H0, dtype=torch.float32)
        self.register_buffer("h0", h0, persistent=True)
        self.register_buffer("h1", _qmf(h0), persistent=True)

    # Muc 6.2: RGB 3 kenh x 4 band x 4 component.
    @property
    def coeff_channels(self) -> int:
        return 3 * len(BAND_ORDER) * len(COMPONENT_ORDER)

    @fp32_transform
    def analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, Layout]:
        """[B,3,H,W] -> ([B,48,H/2,W/2], Layout)."""
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"can [B,3,H,W] RGB, nhan {tuple(x.shape)}")
        b, _, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(
                f"QWT mot level can H va W chan, nhan {h}x{w}. Cau hinh chinh la "
                "256x256; khong tu pad de giau shape sai (spec muc 6.2)."
            )
        h0, h1 = self.h0.to(x.dtype), self.h1.to(x.dtype)

        # [B,3,H,W] -> [B,3,band,component,H/2,W/2]
        out = x.new_empty(b, 3, 4, 4, h // 2, w // 2)
        for ci, off_x in enumerate((0, 1)):          # cay theo truc x
            lo_x, hi_x = _analysis_axis(x, h0, h1, off_x, axis=-1)
            for cj, off_y in enumerate((0, 1)):      # cay theo truc y
                ll, lh = _analysis_axis(lo_x, h0, h1, off_y, axis=-2)
                hl, hh = _analysis_axis(hi_x, h0, h1, off_y, axis=-2)
                # component: (a,a)=real, (b,a)=i, (a,b)=j, (b,b)=k
                comp = ci + 2 * cj
                out[:, :, 0, comp] = ll
                out[:, :, 1, comp] = lh
                out[:, :, 2, comp] = hl
                out[:, :, 3, comp] = hh

        packed = out.reshape(b, self.coeff_channels, h // 2, w // 2)
        layout = Layout(
            backend=BACKEND,
            revision=REVISION,
            original_shape=(b, 3, h, w),
            coeff_shape=tuple(packed.shape),
            levels=self.levels,
            boundary_mode="periodic",
            scale_convention="orthonormal_db4_mean_of_four_trees",
            band_order=BAND_ORDER,
            component_order=COMPONENT_ORDER,
            channel_order=CHANNEL_ORDER,
            extra={"filter": "db4", "taps": len(DB4_H0), "tree_shift_samples": 1},
        )
        return packed, layout

    @fp32_transform
    def synthesis(self, coeffs: torch.Tensor, layout: Layout) -> torch.Tensor:
        """([B,48,Hc,Wc], Layout) -> [B,3,H,W]. Differentiable theo coeffs."""
        layout.require(BACKEND, REVISION)
        b, _, h, w = layout.original_shape
        if coeffs.shape[0] != b or coeffs.shape[1] != self.coeff_channels:
            raise ValueError(
                f"coefficients {tuple(coeffs.shape)} khong khop layout "
                f"{layout.coeff_shape}"
            )
        h0, h1 = self.h0.to(coeffs.dtype), self.h1.to(coeffs.dtype)
        c = coeffs.reshape(b, 3, 4, 4, coeffs.shape[-2], coeffs.shape[-1])

        acc = None
        for ci, off_x in enumerate((0, 1)):
            for cj, off_y in enumerate((0, 1)):
                comp = ci + 2 * cj
                ll, lh = c[:, :, 0, comp], c[:, :, 1, comp]
                hl, hh = c[:, :, 2, comp], c[:, :, 3, comp]
                lo_x = _synthesis_axis(ll, lh, h0, h1, off_y, h, axis=-2)
                hi_x = _synthesis_axis(hl, hh, h0, h1, off_y, h, axis=-2)
                rec = _synthesis_axis(lo_x, hi_x, h0, h1, off_x, w, axis=-1)
                acc = rec if acc is None else acc + rec
        return acc / 4.0
