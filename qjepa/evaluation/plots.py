"""Ve do thi chan doan tu log train va bang kiem "model co dang hoc dung khong".

Cau hoi chinh KHONG phai "loss co giam khong" — loss giam van co the di kem model
lam hong du lieu. Cau hoi dung la **co tot hon viec khong lam gi khong**:

    r = MSE(output) / MSE(input nhieu)      -> r < 1 moi la co ich

Vi vay `r_image`, `r_acc`, `r_gyro` va duong `r = 1` la trung tam cua moi do thi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["step"], r[key]) for r in rows
           if key in r and isinstance(r[key], (int, float)) and np.isfinite(r[key])]
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts)
    return np.asarray(x, float), np.asarray(y, float)


def _smooth(y: np.ndarray, window: int = 9) -> np.ndarray:
    if len(y) < window or window < 2:
        return y
    k = np.ones(window) / window
    return np.convolve(y, k, mode="same")


def diagnose(train: list[dict], val: list[dict]) -> list[tuple[str, str, str]]:
    """-> danh sach (muc, trang thai, giai thich). Trang thai: OK / CANH BAO / XAU."""
    out: list[tuple[str, str, str]] = []

    def add(name, ok, warn, msg):
        out.append((name, "OK" if ok else ("CANH BAO" if warn else "XAU"), msg))

    # 1. Loss PHUC HOI co giam khong.
    #    KHONG dung loss_total: khi vao Stage B, so hang JEPA duoc cong them nen
    #    tong loss nhay len mot cach hop le. So tong qua ranh gioi stage se bao
    #    "xau" sai. Chi so dung la phan phuc hoi (anh + IMU).
    xi, li = _series(train, "loss_image")
    xu, lu = _series(train, "loss_imu")
    recon = li + lu if len(li) == len(lu) and len(li) else (li if len(li) else lu)
    if len(recon) >= 10:
        n = max(2, len(recon) // 5)
        first, last = float(np.mean(recon[:n])), float(np.mean(recon[-n:]))
        drop = (first - last) / first if first > 0 else 0.0
        add("Loss phuc hoi giam", drop > 0.02, drop > -0.02,
            f"{first:.5f} -> {last:.5f} ({drop:+.1%}); khong tinh so hang JEPA")
    else:
        out.append(("Loss phuc hoi giam", "CHUA DU", f"chi co {len(recon)} diem log"))

    # 2. Gradient co lanh manh khong
    _, gn = _series(train, "grad_norm")
    if len(gn):
        add("Gradient norm", 1e-6 < np.median(gn) < 1e3, np.median(gn) > 0,
            f"trung vi {np.median(gn):.4f}, max {gn.max():.3f}")

    # 3. DIEU QUAN TRONG NHAT: co tot hon input nhieu khong
    for key, label in (("r_image", "Anh"), ("r_acc", "Accel"), ("r_gyro", "Gyro")):
        _, r = _series(val, key)
        if not len(r):
            out.append((f"{label} tot hon identity", "CHUA DU", "chua co validation"))
            continue
        best, final = float(r.min()), float(r[-1])
        add(f"{label} tot hon identity", final < 1.0, best < 1.0,
            f"r cuoi = {final:.4f} (tot nhat {best:.4f}); "
            f"{'giam' if final < 1 else 'TANG'} {abs(1-final):.1%} loi so voi input")

    # 4. Ca ba cung tot
    flags = [r.get("improves_all_three") for r in val if "improves_all_three" in r]
    if flags:
        add("Ca ba nguon cung tot", bool(flags[-1]), any(flags),
            f"{sum(bool(f) for f in flags)}/{len(flags)} lan validation dat")

    # 5. Output co tran ra ngoai [0,1] khong
    _, oor = _series(val, "image_out_of_range_fraction")
    if len(oor):
        add("Anh trong [0,1]", oor[-1] < 0.05, oor[-1] < 0.2,
            f"{oor[-1]:.2%} pixel ngoai khoang truoc khi clamp")

    # 6. Overfit: validation co xau di trong khi train tot len khong
    _, vmae = _series(val, "image_mae")
    _, tl = _series(train, "loss_image")
    if len(vmae) >= 4 and len(tl) >= 10:
        half = len(vmae) // 2
        v_first, v_last = float(np.mean(vmae[:half])), float(np.mean(vmae[half:]))
        n = max(2, len(tl) // 5)
        t_drop = (np.mean(tl[:n]) - np.mean(tl[-n:])) / max(np.mean(tl[:n]), 1e-12)
        overfit = v_last > v_first * 1.02 and t_drop > 0.02
        add("Khong overfit", not overfit, not overfit,
            f"validation MAE {v_first:.5f} -> {v_last:.5f}, train giam {t_drop:+.1%}")

    # 7. Stage B co that su bat JEPA khong
    _, lam = _series(train, "lambda_jepa")
    if len(lam) and lam.max() > 0:
        add("JEPA da bat", True, True, f"lambda_J max = {lam.max():.3f}")
    elif len(lam):
        out.append(("JEPA da bat", "CHUA DU", "lambda_J luon 0 — chua vao Stage B"))
    return out


def plot_training(run_dir: str | Path, out_path: str | Path | None = None,
                  title: str | None = None, dpi: int = 110) -> Path:
    """Ve bang do thi chan doan tu `train_log.jsonl` va `validation.jsonl`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    train = read_jsonl(run_dir / "train_log.jsonl")
    val = read_jsonl(run_dir / "validation.jsonl")
    if not train:
        raise FileNotFoundError(f"khong co du lieu log trong {run_dir}")
    out_path = Path(out_path or run_dir / "training_curves.png")

    has_sens = any("sens_gain_normalized" in r for r in train)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(title or f"Chan doan train — {run_dir.name}", fontsize=14)
    ax = axes.ravel()

    # Ranh gioi Stage A -> B
    switch = next((r["step"] for r in train if r.get("stage") == "B"), None)

    def mark_stage(a):
        if switch:
            a.axvline(switch, color="gray", ls=":", lw=1)
            a.annotate("Stage B", (switch, a.get_ylim()[1]), fontsize=7,
                       color="gray", ha="left", va="top")

    # (1) Loss tong + thanh phan
    for key, lab in (("loss_total", "tong"), ("loss_image", "anh"), ("loss_imu", "IMU")):
        x, y = _series(train, key)
        if len(x):
            ax[0].plot(x, _smooth(y), label=lab, lw=1.3)
    ax[0].set_yscale("log")
    ax[0].set_title("Loss (log) — tong NHAY len khi Stage B bat JEPA")
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("step"); mark_stage(ax[0])

    # (2) accel vs gyro
    for key, lab in (("loss_acc", "accel"), ("loss_gyro", "gyro")):
        x, y = _series(train, key)
        if len(x):
            ax[1].plot(x, _smooth(y), label=lab, lw=1.3)
    ax[1].set_yscale("log"); ax[1].set_title("Loss IMU theo nhom"); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("step")

    # (3) *** DO THI QUAN TRONG NHAT ***
    for key, lab, c in (("r_image", "anh", "tab:blue"),
                        ("r_acc", "accel", "tab:orange"),
                        ("r_gyro", "gyro", "tab:green")):
        x, y = _series(val, key)
        if len(x):
            ax[2].plot(x, y, "o-", label=lab, color=c, ms=3, lw=1.3)
    ax[2].axhline(1.0, color="red", ls="--", lw=1.5)
    ax[2].annotate("r = 1: bang khong lam gi", (0.02, 1.0), xycoords=("axes fraction", "data"),
                   fontsize=8, color="red", va="bottom")
    ax[2].set_title("r = MSE(output)/MSE(nhieu)   —  DUOI 1 moi co ich")
    ax[2].set_xlabel("step"); ax[2].legend(fontsize=8)

    # (4) PSNR / SSIM
    x, y = _series(val, "image_psnr")
    if len(x):
        ax[3].plot(x, y, "o-", ms=3, color="tab:blue", label="PSNR (dB)")
        ax[3].set_ylabel("PSNR (dB)")
    x2, y2 = _series(val, "image_ssim")
    if len(x2):
        a2 = ax[3].twinx(); a2.plot(x2, y2, "s-", ms=3, color="tab:red", label="SSIM")
        a2.set_ylabel("SSIM", color="tab:red")
    ax[3].set_title("Chat luong anh (validation)"); ax[3].set_xlabel("step")

    # (5) Gradient norm
    x, y = _series(train, "grad_norm")
    if len(x):
        ax[4].plot(x, y, lw=0.6, alpha=0.4, color="gray")
        ax[4].plot(x, _smooth(y), lw=1.5, color="tab:purple")
        ax[4].set_yscale("log")
    ax[4].set_title("Gradient norm (bung no / triet tieu?)"); ax[4].set_xlabel("step")

    # (6) LR + lambda_J
    x, y = _series(train, "lr")
    if len(x):
        ax[5].plot(x, y, color="tab:blue", label="learning rate")
        ax[5].set_yscale("log")
    x2, y2 = _series(train, "lambda_jepa")
    if len(x2) and y2.max() > 0:
        a2 = ax[5].twinx(); a2.plot(x2, y2, color="tab:green", label="lambda_JEPA")
        a2.set_ylabel("lambda_JEPA", color="tab:green")
    ax[5].set_title("Lich LR va ramp JEPA"); ax[5].set_xlabel("step")

    # (7) Overfit? Train giam ma validation tang la dau hieu.
    #    KHONG ve RMSE phuc hoi canh RMSE dau vao: hai duong chong khit nhau nen
    #    khong doc duoc gi — ty so da nam o do thi (3).
    xt, yt = _series(train, "loss_image")
    if len(xt):
        ax[6].plot(xt, _smooth(yt), color="tab:blue", lw=1.3, label="train: L1 anh")
        ax[6].set_ylabel("train L1", color="tab:blue")
    xv, yv = _series(val, "image_mae")
    if len(xv):
        a2 = ax[6].twinx()
        a2.plot(xv, yv, "o-", ms=3, color="tab:red", label="validation: MAE anh")
        a2.set_ylabel("validation MAE", color="tab:red")
    ax[6].set_title("Overfit? train giam ma validation tang")
    ax[6].set_xlabel("step"); ax[6].legend(fontsize=7, loc="upper left")

    # (8) JEPA loss hoac do nhay encoder
    if has_sens:
        for key, lab in (("sens_gain_normalized", "normalized"), ("sens_gain_raw", "raw")):
            x, y = _series(train, key)
            if len(x):
                ax[7].plot(x, _smooth(y), label=lab, lw=1.2)
        ax[7].set_yscale("log"); ax[7].set_title("Do nhay encoder (FD gain)")
        ax[7].legend(fontsize=8)
    else:
        for key, lab in (("loss_jepa", "JEPA"), ("jepa_image", "anh"), ("jepa_imu", "IMU")):
            x, y = _series(train, key)
            if len(x):
                ax[7].plot(x, _smooth(y), label=lab, lw=1.2)
        ax[7].set_title("Loss JEPA"); ax[7].legend(fontsize=8)
    ax[7].set_xlabel("step")

    # (9) Bang chan doan bang chu
    ax[8].axis("off")
    rows = diagnose(train, val)
    colors = {"OK": "tab:green", "CANH BAO": "tab:orange", "XAU": "tab:red", "CHUA DU": "gray"}
    ax[8].text(0, 1.0, "KIEM TRA", fontsize=11, weight="bold", va="top")
    for i, (name, status, msg) in enumerate(rows):
        y0 = 0.92 - i * 0.105
        ax[8].text(0, y0, f"{status:9}", fontsize=8.5, family="monospace",
                   color=colors.get(status, "black"), weight="bold", va="top")
        ax[8].text(0.27, y0, name, fontsize=8.5, va="top")
        ax[8].text(0.27, y0 - 0.042, msg, fontsize=7, color="dimgray", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_diagnosis(run_dir: str | Path) -> list[tuple[str, str, str]]:
    run_dir = Path(run_dir)
    rows = diagnose(read_jsonl(run_dir / "train_log.jsonl"),
                    read_jsonl(run_dir / "validation.jsonl"))
    width = max((len(n) for n, _, _ in rows), default=10)
    for name, status, msg in rows:
        print(f"  [{status:8}] {name:{width}}  {msg}")
    return rows


def plot_samples(model, cfg, samples, out_path: str | Path, *, device="cpu",
                 num: int = 4, realization: int = 0, dpi: int = 95) -> Path:
    """Panel clean / bad / restored / |error| tren cac sample CO DINH.

    Dung cung sample va cung realization moi lan goi, de so sanh giua cac
    checkpoint co y nghia (khong chon vai anh dep).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from ..data.dataset import collate
    from ..training.trainer import make_dataset

    model.eval()
    dataset = make_dataset(samples[:num], cfg, realization=realization)
    batch = collate([dataset[i] for i in range(min(num, len(dataset)))])
    with torch.no_grad():
        out = model(
            batch["image_bad"].to(device), batch["imu_bad_phys"].to(device),
            batch["image_time"].to(device), batch["imu_times"].to(device),
        )
    clean = batch["image_clean"].cpu().numpy()
    bad = batch["image_bad"].cpu().numpy()
    hat = out["image_hat"].clamp(0, 1).cpu().numpy()
    imu_c = batch["imu_clean_phys"].cpu().numpy()
    imu_b = batch["imu_bad_phys"].cpu().numpy()
    imu_h = out["imu_hat_phys"].cpu().numpy()
    n = clean.shape[0]

    fig, axes = plt.subplots(n, 6, figsize=(19, 3.1 * n), squeeze=False)
    for i in range(n):
        mse_bad = float(((bad[i] - clean[i]) ** 2).mean())
        mse_hat = float(((hat[i] - clean[i]) ** 2).mean())
        panels = [
            (clean[i].transpose(1, 2, 0), "SACH (target)"),
            (bad[i].transpose(1, 2, 0), f"NHIEU (vao)  MSE {mse_bad:.5f}"),
            (hat[i].transpose(1, 2, 0),
             f"PHUC HOI  MSE {mse_hat:.5f}  r={mse_hat/max(mse_bad,1e-12):.3f}"),
            (np.abs(hat[i] - clean[i]).transpose(1, 2, 0) * 5, "|loi| x5"),
        ]
        for j, (img, t) in enumerate(panels):
            axes[i][j].imshow(np.clip(img, 0, 1))
            axes[i][j].set_title(t, fontsize=8)
            axes[i][j].axis("off")
        # Hai truc IMU: ve SAI SO chu khong ve tin hieu.
        # Nhieu IMU nho hon dao dong that hang tram lan nen ba duong tin hieu
        # chong khit len nhau, khong doc duoc gi. Sai so moi cho thay model co
        # go bot nhieu hay khong.
        for j, (ch, lab) in enumerate(((0, "accel ax (m/s²)"), (3, "gyro gx (rad/s)"))):
            a = axes[i][4 + j]
            e_bad = imu_b[i, ch] - imu_c[i, ch]
            e_hat = imu_h[i, ch] - imu_c[i, ch]
            a.axhline(0, color="black", lw=0.6)
            a.plot(e_bad, lw=0.9, alpha=.75, color="tab:orange", label="loi dau vao")
            a.plot(e_hat, lw=1.1, color="tab:green", label="loi sau phuc hoi")
            rb = float((e_hat**2).mean() / max((e_bad**2).mean(), 1e-12))
            a.set_title(f"SAI SO {lab}   r={rb:.3f}"
                        f"{'  (tot hon)' if rb < 1 else '  (xau hon)'}", fontsize=8)
            a.tick_params(labelsize=6)
            if i == 0:
                a.legend(fontsize=6)
    model.train()
    out_path = Path(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
