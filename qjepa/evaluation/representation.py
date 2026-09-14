"""Diagnostic do nhay va theo doi latent collapse (themjacobian muc 8, 15).

**Khong** cong vao loss train. Output gain o day dung don vi/cong thuc KHAC voi
normalized latent gain cua `encoder_sensitivity` — hai metric khong so truc tiep.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import collate
from ..training.encoder_sensitivity import (
    make_encoder_probe,
    mean_per_sample,
    normalized_dense_for_measurement,
    seeded_rademacher,
)

R_IMAGE = 1.0 / 255.0
R_IMU = 0.01


@torch.no_grad()
def output_gain(
    model, batch: dict, source: str, target: str, *, direction_seed: int, alpha: float = 1.0, device="cpu"
) -> torch.Tensor:
    """Gain input→output qua TOAN BO model phuc hoi chinh (muc 15).

        E_in  = mean((delta_input / r_input)^2)
        E_out = mean(((out_perturbed - out_base) / r_output)^2)
        G     = E_out / E_in

    Identity co gain = 1. Khong clamp output anh truoc khi do.
    """
    image = batch["image_bad"].to(device)
    imu_phys = batch["imu_bad_phys"].to(device)
    imu_norm = model.normalizer.normalize(imu_phys)
    t_img, t_imu = batch["image_time"].to(device), batch["imu_times"].to(device)

    base = model(image, imu_phys, t_img, t_imu)
    x = image if source == "image" else imu_norm
    v = seeded_rademacher(x, probe_seed=direction_seed, context=(source, target))
    xp, energy, _ = make_encoder_probe(
        x, source, v, image_epsilon=R_IMAGE, imu_epsilon=R_IMU, alpha=alpha
    )
    if source == "image":
        pert = model(xp, imu_phys, t_img, t_imu)
    else:
        pert = model(image, model.normalizer.denormalize(xp), t_img, t_imu)

    r_in = R_IMAGE if source == "image" else R_IMU
    key = "image_hat" if target == "image" else "imu_hat_norm"
    r_out = R_IMAGE if target == "image" else R_IMU
    e_in = energy / (r_in**2)
    e_out = mean_per_sample(((pert[key] - base[key]) / r_out).square())
    return e_out / e_in


@torch.no_grad()
def latent_statistics(features: torch.Tensor) -> dict:
    """Thong ke phat hien collapse tren mot bank (muc 8).

    `features`: `[N, D, ...]` gom N sample cua bank.
    """
    n = features.shape[0]
    flat = features.float().flatten(2) if features.ndim > 2 else features.float()[..., None]
    # (1) std qua CAC SAMPLE KHAC NHAU tai cung vi tri/channel — khong phai std
    # giua cac pixel cua mot sample (do se tao variance gia).
    across = flat.std(dim=0, unbiased=False)
    pooled = flat.mean(dim=2)                                   # [N, D] cho rank
    centered = pooled - pooled.mean(dim=0, keepdim=True)
    stats = {
        "num_samples": int(n),
        "raw_rms": float(flat.pow(2).mean().sqrt()),
        "std_across_samples_mean": float(across.mean()),
        "std_across_samples_median": float(across.median()),
        "channel_variance_mean": float(flat.var(dim=1, unbiased=False).mean()),
    }
    if n >= 2:
        s = torch.linalg.svdvals(centered.double())
        p = s / s.sum().clamp_min(1e-12)
        entropy = -(p * p.clamp_min(1e-12).log()).sum()
        stats["effective_rank"] = float(entropy.exp())
        stats["max_rank_possible"] = int(min(pooled.shape))
        norm = torch.nn.functional.normalize(pooled, dim=1)
        sim = norm @ norm.T
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        stats["cosine_similarity_mean"] = float(off.mean())
    return stats


@torch.no_grad()
def sensitivity_report(model, cfg, samples, *, device="cpu", bank_samples: int = 64,
                       seed: int = 73130, alphas=(0.5, 1.0, 2.0)) -> dict:
    """Bao cao day du: latent stats + 4 output gain + alpha sweep."""
    from ..training.trainer import make_dataset

    model.eval()
    bank = samples[:bank_samples]
    dataset = make_dataset(bank, cfg, realization=cfg.corruption.validation_realization)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate,
                        num_workers=cfg.train.num_workers)

    fi_all, fu_all, hi_all, hu_all = [], [], [], []
    gains: dict[str, list[float]] = {}
    enc_gain: dict[str, list[float]] = {}
    n_seen = 0
    for batch in loader:
        image = batch["image_bad"].to(device)
        imu_phys = batch["imu_bad_phys"].to(device)
        out = model(image, imu_phys, batch["image_time"].to(device), batch["imu_times"].to(device))
        fi_all.append(out["fi"].cpu()); fu_all.append(out["fu"].cpu())
        hi_all.append(normalized_dense_for_measurement(out["fi"]).cpu())
        hu_all.append(normalized_dense_for_measurement(out["fu"]).cpu())
        n_seen += image.shape[0]

        # Bon huong output gain (muc 15).
        for src in ("image", "imu"):
            for tgt in ("image", "imu"):
                g = output_gain(model, batch, src, tgt, direction_seed=seed, device=device)
                gains.setdefault(f"{src}_to_{tgt}", []).extend(g.cpu().tolist())
        # Encoder gain normalized, cung alpha sweep.
        imu_norm = model.normalizer.normalize(imu_phys)
        for alpha in alphas:
            for src, x, f_base in (("image", image, out["fi"]), ("imu", imu_norm, out["fu"])):
                v = seeded_rademacher(x, probe_seed=seed, context=("enc", src))
                xp, energy, _ = make_encoder_probe(
                    x, src, v, image_epsilon=R_IMAGE, imu_epsilon=R_IMU, alpha=alpha
                )
                f_probe = (model.encode_image_dense(xp) if src == "image"
                           else model.encode_imu_dense_normalized(xp))
                num = mean_per_sample(
                    (normalized_dense_for_measurement(f_probe)
                     - normalized_dense_for_measurement(f_base)).square()
                )
                enc_gain.setdefault(f"{src}_alpha{alpha}", []).extend((num / energy).cpu().tolist())

    return {
        "bank_samples_requested": bank_samples,
        "bank_samples_actual": n_seen,
        "latent_FI": latent_statistics(torch.cat(fi_all)),
        "latent_FU": latent_statistics(torch.cat(fu_all)),
        "latent_FI_normalized": latent_statistics(torch.cat(hi_all)),
        "latent_FU_normalized": latent_statistics(torch.cat(hu_all)),
        "output_gain": {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                            "p95": float(np.percentile(v, 95))} for k, v in gains.items()},
        "encoder_gain_normalized": {k: float(np.mean(v)) for k, v in enc_gain.items()},
        "note": (
            "output_gain va encoder_gain_normalized dung don vi/cong thuc KHAC nhau "
            "(muc 15) — khong so truc tiep hai cot nay."
        ),
    }
