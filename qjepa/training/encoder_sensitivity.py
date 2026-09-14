"""Finite-difference Jacobian regularization tai encoder (themjacobian muc 6, 9).

Do do nhay cua **FI/FU dense cuoi hai encoder online, TRUOC fusion** doi voi mot
nhieu loan nho them vao du lieu **truoc wavelet**.

    e_a[n] = mean_nonbatch( (x'_a[n] - x_a[n])^2 )        # nang luong delta THUC
    d_a[n] = mean_nonbatch( (h'_a[n] - h_a[n])^2 )        # h = LN khong affine cua f
    g_a[n] = d_a[n] / e_a[n]
    L_enc_a = mean_batch( g_a )

Day la **normalized-latent directional finite-difference sensitivity**, KHONG phai
Frobenius norm, spectral norm hay Lipschitz bound cua Jacobian that. LayerNorm chi
dung DE DO; FI/FU goc van di vao fusion khong doi.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SOURCES = ("image", "imu")


@dataclass(frozen=True)
class EncoderSensitivityConfig:
    enabled: bool = False
    image_epsilon: float = 1.0 / 255.0
    imu_normalized_epsilon: float = 0.01
    alpha: float = 1.0
    layer_norm_eps: float = 1e-5
    minimum_input_energy: float = 1e-12
    weight_max: float = 1e-4
    ramp_updates: int = 200
    probe_seed: int = 73129
    modality_multipliers: tuple[float, float] = (1.0, 1.0)   # (image, imu)


def _at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """Nang fp16/bf16 len fp32; GIU NGUYEN fp32 va fp64.

    Khong dung `.float()` truc tiep: no ha fp64 xuong fp32 va lam hong cac phep
    kiem chung finite-difference chay o double.
    """
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def mean_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Trung binh moi chieu tru batch -> [B]."""
    return _at_least_fp32(x).flatten(1).mean(dim=1)


def normalized_dense_for_measurement(f: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm KHONG affine tren truc channel, tai TUNG vi tri (muc 6.2).

    f: `[B,D,H,W]` (anh) hoac `[B,D,T]` (IMU). Khong normalize theo batch, khong
    gop spatial/time thanh mot vector, khong dung pooled summary.
    """
    if f.ndim not in (3, 4):
        raise ValueError(f"can dense feature channel-first [B,D,...], nhan {tuple(f.shape)}")
    q = _at_least_fp32(f).movedim(1, -1)
    h = F.layer_norm(q, (q.shape[-1],), weight=None, bias=None, eps=eps)
    return h.movedim(-1, 1)


def seeded_rademacher(
    x: torch.Tensor, *, probe_seed: int, context: tuple, draw_index: int = 0
) -> torch.Tensor:
    """Huong Rademacher (-1/+1) tu seed SHA-256 stateless (muc 10).

    Khong dung global RNG: perturbation khong duoc lam lech corruption/sampler
    cua control va treatment.
    """
    raw = "|".join(str(c) for c in (probe_seed, *context, draw_index)).encode()
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 1
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, x.shape, generator=gen, dtype=torch.int8)
    return (bits.to(x.device, torch.float32) * 2.0 - 1.0)


def make_encoder_probe(
    x: torch.Tensor,
    source: str,
    direction: torch.Tensor,
    *,
    image_epsilon: float = 1.0 / 255.0,
    imu_epsilon: float = 0.01,
    alpha: float = 1.0,
    minimum_energy: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """-> (x_perturbed, actual_energy [B], clipped_fraction).

    Anh bi clamp ve `[0,1]` roi **do delta THUC sau clamp** — khong dung delta
    danh nghia o vung bien. IMU khong clamp.
    """
    if source not in SOURCES:
        raise ValueError(f"source={source!r}, phai la mot trong {SOURCES}")
    if x.dtype != torch.float32:
        raise ValueError(f"can FP32, nhan {x.dtype}")
    if not torch.isfinite(x).all():
        raise ValueError("input chua NaN/Inf")
    if direction.shape != x.shape:
        raise ValueError(f"direction {tuple(direction.shape)} khac input {tuple(x.shape)}")
    if not torch.all(direction.abs() == 1):
        raise ValueError("direction phai la Rademacher (+-1)")
    epsilon = image_epsilon if source == "image" else imu_epsilon
    if epsilon <= 0 or alpha <= 0:
        raise ValueError(f"bien do probe khong hop le: epsilon={epsilon}, alpha={alpha}")
    if source == "image" and (x.min() < 0 or x.max() > 1):
        raise ValueError("anh nam ngoai [0,1]")

    with torch.no_grad():
        proposed = x + alpha * epsilon * direction
        xp = proposed.clamp(0.0, 1.0) if source == "image" else proposed
        energy = mean_per_sample((xp - x).square())
        if not torch.isfinite(energy).all():
            raise FloatingPointError("nang luong perturbation khong huu han")
        if (energy <= minimum_energy).any():
            raise ValueError(
                f"perturbation bi mat (energy <= {minimum_energy}); "
                "khong duoc am tham bo sample roi bao gain = 0"
            )
        clipped = float((proposed != xp).float().mean())
    return xp, energy, clipped


def encoder_fd_loss(
    f_base: torch.Tensor,
    f_probe: torch.Tensor,
    actual_input_energy: torch.Tensor,
    *,
    ln_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (loss scalar, gain per-sample da detach de log).

    KHONG detach f_base: gradient phai di qua ca hai nhanh toi cung encoder.
    """
    if f_base.shape != f_probe.shape:
        raise ValueError(f"shape lech: {tuple(f_base.shape)} vs {tuple(f_probe.shape)}")
    h_base = normalized_dense_for_measurement(f_base, ln_eps)
    h_probe = normalized_dense_for_measurement(f_probe, ln_eps)
    numerator = mean_per_sample((h_probe - h_base).square())
    gain = numerator / actual_input_energy
    if not torch.isfinite(gain).all():
        raise FloatingPointError("do nhay encoder khong huu han")
    return gain.mean(), gain.detach()


def raw_feature_fd_gain(
    f_base: torch.Tensor, f_probe: torch.Tensor, actual_input_energy: torch.Tensor
) -> torch.Tensor:
    """Gain tren feature RAW (chua LN) — CHI diagnostic (muc 8.6).

    Khong cong vao loss va khong so truc tiep voi normalized gain: raw gain giam
    do co scale khong tu chung minh robustness.
    """
    with torch.no_grad():
        return mean_per_sample((f_probe - f_base).square()) / actual_input_energy


def encoder_weight(step: int, cfg: EncoderSensitivityConfig) -> float:
    """Ramp tuyen tinh theo so optimizer update THANH CONG (muc 11).

        lambda(s) = weight_max * min((s+1)/ramp_updates, 1)
    """
    return cfg.weight_max * min((step + 1) / max(cfg.ramp_updates, 1), 1.0)


def source_for_update(step: int) -> str:
    """Luan phien mot source moi update: chan = anh, le = IMU (muc 7)."""
    return "image" if step % 2 == 0 else "imu"
