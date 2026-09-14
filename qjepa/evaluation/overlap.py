"""Ghep cac cua so IMU chong lan truoc khi tinh metric trajectory (spec muc 18.3).

Moi window tra L mau; cung mot timestamp co the duoc du doan nhieu lan. Ghep
bang trong so tam giac DUONG:

    w[k] = 1 - abs((2k - (L-1)) / (L+1)),   k = 0..L-1

Endpoint van duong (khac Hann), nen khong co vi tri nao bi chia cho 0.

Ghep theo **chi so IMU goc trong manifest**, khong so sanh float timestamp bang
dau bang. Khong ghep xuyen trajectory. Moi sample chi duoc cong mot lan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def triangular_weights(length: int) -> np.ndarray:
    """Trong so tam giac duong cho cua so dai `length`."""
    k = np.arange(length, dtype=np.float64)
    return 1.0 - np.abs((2 * k - (length - 1)) / (length + 1))


@dataclass
class TrajectoryMerger:
    """Cong don du doan cua mot trajectory theo chi so IMU toan cuc."""

    num_rows: int
    channels: int = 6

    total: np.ndarray = field(init=False)
    weight: np.ndarray = field(init=False)
    seen: set = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self.total = np.zeros((self.channels, self.num_rows), dtype=np.float64)
        self.weight = np.zeros(self.num_rows, dtype=np.float64)
        self.seen = set()

    def add(self, prediction_phys: np.ndarray, start: int, sample_id: str | None = None) -> None:
        """Cong mot window `[channels, L]` bat dau tai chi so `start`.

        `sample_id` (neu co) chan cong lap cung mot sample hai lan.
        """
        if prediction_phys.ndim != 2 or prediction_phys.shape[0] != self.channels:
            raise ValueError(
                f"can [{self.channels}, L], nhan {prediction_phys.shape}"
            )
        if sample_id is not None:
            if sample_id in self.seen:
                return
            self.seen.add(sample_id)
        length = prediction_phys.shape[1]
        end = start + length
        if start < 0 or end > self.num_rows:
            raise ValueError(f"window [{start}, {end}) nam ngoai {self.num_rows} hang")
        w = triangular_weights(length)
        self.total[:, start:end] += prediction_phys * w[None, :]
        self.weight[start:end] += w

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """-> (merged `[channels, num_rows]`, mask `[num_rows]` bool).

        Chi vi tri co `weight > 0` moi hop le. Khong dien ground truth vao vung
        khong co du doan.
        """
        mask = self.weight > 0
        merged = np.full((self.channels, self.num_rows), np.nan, dtype=np.float64)
        merged[:, mask] = self.total[:, mask] / self.weight[None, mask]
        return merged, mask

    @property
    def coverage(self) -> float:
        return float(np.count_nonzero(self.weight > 0) / max(self.num_rows, 1))


def merge_predictions(
    predictions: dict[str, list[tuple[np.ndarray, int, str]]],
    rows_per_trajectory: dict[str, int],
    channels: int = 6,
) -> dict[str, dict]:
    """Ghep theo tung trajectory rieng biet.

    Args:
        predictions: `{trajectory_key: [(prediction [C,L], imu_start, sample_id), ...]}`
        rows_per_trajectory: so hang IMU that cua moi trajectory.

    Returns:
        `{trajectory_key: {"merged", "mask", "coverage"}}`
    """
    out: dict[str, dict] = {}
    for key, items in predictions.items():
        merger = TrajectoryMerger(rows_per_trajectory[key], channels)
        for pred, start, sample_id in items:
            merger.add(pred, start, sample_id)
        merged, mask = merger.result()
        out[key] = {"merged": merged, "mask": mask, "coverage": merger.coverage}
    return out


def merged_metrics(
    merged: dict[str, dict],
    clean: dict[str, np.ndarray],
    bad: dict[str, np.ndarray],
) -> dict:
    """Metric theo trajectory roi trung binh voi trong so BANG NHAU (muc 18.5).

    Moi trajectory dong gop nhu nhau bat ke so window, de trajectory dai khong
    lan at ket qua.
    """
    per_traj = {}
    for key, item in merged.items():
        mask = item["mask"]
        if not mask.any():
            continue
        pred = item["merged"][:, mask]
        ref = clean[key][:, mask]
        noisy = bad[key][:, mask]
        mse_r = ((pred - ref) ** 2).mean(axis=1)
        mse_b = ((noisy - ref) ** 2).mean(axis=1)
        per_traj[key] = {
            "coverage": item["coverage"],
            "mse_accel": float(mse_r[:3].mean()),
            "mse_gyro": float(mse_r[3:].mean()),
            "mse_accel_bad": float(mse_b[:3].mean()),
            "mse_gyro_bad": float(mse_b[3:].mean()),
            "r_acc": float(mse_r[:3].mean() / max(mse_b[:3].mean(), 1e-12)),
            "r_gyro": float(mse_r[3:].mean() / max(mse_b[3:].mean(), 1e-12)),
        }
    if not per_traj:
        return {"trajectories": 0}
    keys = ("coverage", "mse_accel", "mse_gyro", "r_acc", "r_gyro")
    summary = {k: float(np.mean([v[k] for v in per_traj.values()])) for k in keys}
    summary["trajectories"] = len(per_traj)
    summary["per_trajectory"] = per_traj
    return summary
