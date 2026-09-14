"""Ghep anh-IMU theo timestamp, chia split theo trajectory, tinh norm stats.

Spec muc 3.2, 3.3 va 4.3. Manifest la nguon su that duy nhat cho sample: moi
dong la mot cap (mot frame, mot cua so IMU L hang) da qua kiem tra.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .tartanair import CAMERA_ID, Trajectory, discover_trajectories

MANIFEST_FIELDS = (
    "sample_id",
    "split",
    "environment",
    "difficulty",
    "trajectory_id",
    "camera",
    "image_path",
    "image_index",
    "image_time",
    "imu_start",
    "imu_end_exclusive",
    "imu_start_time",
    "imu_end_time",
    "center_error_s",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    environment: str
    difficulty: str
    trajectory_id: str
    camera: str
    image_path: str
    image_index: int
    image_time: float
    imu_start: int
    imu_end_exclusive: int
    imu_start_time: float
    imu_end_time: float
    center_error_s: float

    @property
    def trajectory_key(self) -> str:
        return f"{self.environment}/{self.difficulty}/{self.trajectory_id}"


def _window_centers(t: np.ndarray, window: int) -> np.ndarray:
    """c[s] = (t[s] + t[s+L-1]) / 2 cho moi start s hop le."""
    return (t[: len(t) - window + 1] + t[window - 1 :]) / 2.0


def pair_trajectory(
    traj: Trajectory,
    window: int = 128,
    *,
    split: str,
    max_relative_dt_deviation: float = 0.01,
    max_center_error_in_dt: float = 1.0,
) -> tuple[list[Sample], dict]:
    """Tao sample cho mot trajectory theo chinh sach centered offline.

    Loai sample thay vi pad/lap khi khong du context (spec muc 2.4, 3.3).
    """
    u, t = traj.load_imu()
    cam_t = traj.load_camera_times()
    images = traj.image_paths()

    stats = {
        "trajectory": traj.key,
        "split": split,
        "imu_rows": int(len(u)),
        "num_images": len(images),
        "rejected_no_rows": 0,
        "rejected_outside_center_range": 0,
        "rejected_center_error": 0,
        "rejected_support": 0,
        "rejected_non_uniform": 0,
        "accepted": 0,
    }
    if len(images) != len(cam_t):
        raise ValueError(
            f"{traj.key}: {len(images)} anh nhung {len(cam_t)} camera timestamp"
        )
    if len(u) < window:
        stats["rejected_no_rows"] = len(images)
        return [], stats

    dt = np.diff(t)
    median_dt = float(np.median(dt))
    centers = _window_centers(t, window)

    samples: list[Sample] = []
    for idx, (image_path, t_image) in enumerate(zip(images, cam_t)):
        if t_image < centers[0] or t_image > centers[-1]:
            stats["rejected_outside_center_range"] += 1
            continue
        # So hai candidate gan nhat, chon center error nho nhat (muc 3.3 buoc 2).
        pos = int(np.searchsorted(centers, t_image))
        cands = [c for c in (pos - 1, pos) if 0 <= c < len(centers)]
        start = min(cands, key=lambda s: abs(centers[s] - t_image))
        err = abs(float(centers[start] - t_image))
        if err > max_center_error_in_dt * median_dt:
            stats["rejected_center_error"] += 1
            continue
        end = start + window
        if not (t[start] <= t_image <= t[end - 1]):
            stats["rejected_support"] += 1
            continue
        win_dt = dt[start : end - 1]
        if float(np.max(np.abs(win_dt - median_dt)) / median_dt) > max_relative_dt_deviation:
            stats["rejected_non_uniform"] += 1
            continue
        samples.append(
            Sample(
                sample_id=f"{traj.environment}__{traj.difficulty}__{traj.trajectory_id}__{idx:06d}",
                split=split,
                environment=traj.environment,
                difficulty=traj.difficulty,
                trajectory_id=traj.trajectory_id,
                camera=CAMERA_ID,
                image_path=str(image_path),
                image_index=idx,
                image_time=float(t_image),
                imu_start=int(start),
                imu_end_exclusive=int(end),
                imu_start_time=float(t[start]),
                imu_end_time=float(t[end - 1]),
                center_error_s=err,
            )
        )
    stats["accepted"] = len(samples)
    return samples, stats


def assign_splits(
    trajectories: list[Trajectory],
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, str]:
    """Gan split theo NHOM (environment, trajectory_id).

    Data_easy va Data_hard cua cung mot P luon cung split - hai thu muc do la
    cung mot chuyen dong duoc render o hai do kho.

    Neu MOI trajectory da co split_hint (folder view train/valid/test) thi dung
    nguyen split do de khong lam hong holdout da co.
    """
    if all(t.split_hint for t in trajectories):
        return {t.key: t.split_hint for t in trajectories}

    groups: dict[tuple[str, str], list[Trajectory]] = {}
    for t in trajectories:
        groups.setdefault((t.environment, t.trajectory_id), []).append(t)

    ordered = sorted(groups)
    # Thu tu xac dinh boi hash on dinh, khong phai Python hash() (spec muc 5.1).
    def rank(key: tuple[str, str]) -> str:
        raw = f"{seed}|{key[0]}|{key[1]}".encode()
        return hashlib.sha256(raw).hexdigest()

    ordered.sort(key=rank)
    n = len(ordered)
    # Bao dam valid/test moi tap co it nhat mot group khi co tu 3 group tro len,
    # thay vi lam tron roi de mot tap rong.
    if n < 2:
        n_valid = n_test = 0
    elif n == 2:
        n_valid, n_test = 0, 1
    else:
        n_valid = max(1, int(round(split_ratio[1] * n)))
        n_test = max(1, int(round(split_ratio[2] * n)))
        while n - n_valid - n_test < 1 and (n_valid > 1 or n_test > 1):
            if n_valid >= n_test and n_valid > 1:
                n_valid -= 1
            else:
                n_test -= 1
    n_train = n - n_valid - n_test
    assignment: dict[str, str] = {}
    for i, key in enumerate(ordered):
        split = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
        assert split in ("train", "valid", "test")
        for t in groups[key]:
            assignment[t.key] = split
    return assignment


def compute_norm_stats(
    trajectories: list[Trajectory], splits: dict[str, str], channels: int = 6
) -> dict:
    """Mean/std tren TOAN BO hang IMU clean cua train, moi timestamp mot lan.

    Khong duyet theo window: cac window chong lan nhau se dem lai cung hang
    nhieu lan va lam lech thong ke (spec muc 3.2, 4.3).
    """
    count = 0
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    used: list[str] = []
    for traj in trajectories:
        if splits.get(traj.key) != "train":
            continue
        u, _ = traj.load_imu()
        total += u.sum(axis=0)
        total_sq += (u**2).sum(axis=0)
        count += len(u)
        used.append(traj.key)
    if count == 0:
        raise ValueError("khong co trajectory train nao de tinh normalization stats")
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)
    return {
        "source": "train_clean_global_unique_samples",
        "num_rows": int(count),
        "num_trajectories": len(used),
        "trajectories": used,
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "channels": ["ax", "ay", "az", "gx", "gy", "gz"][:channels],
        "units": ["m_s2"] * 3 + ["rad_s"] * 3,
    }


def build_manifest(
    root: str | Path,
    window: int = 128,
    *,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    max_relative_dt_deviation: float = 0.01,
    max_center_error_in_dt: float = 1.0,
) -> dict:
    """Quet root, ghep sample, chia split va tinh norm stats."""
    trajectories = discover_trajectories(root)
    splits = assign_splits(trajectories, split_ratio, seed)

    samples: dict[str, list[Sample]] = {"train": [], "valid": [], "test": []}
    per_traj = []
    for traj in trajectories:
        split = splits[traj.key]
        got, stats = pair_trajectory(
            traj,
            window,
            split=split,
            max_relative_dt_deviation=max_relative_dt_deviation,
            max_center_error_in_dt=max_center_error_in_dt,
        )
        samples[split].extend(got)
        per_traj.append(stats)

    norm = compute_norm_stats(trajectories, splits)
    payload = {
        "root": str(root),
        "window": window,
        "camera": CAMERA_ID,
        "split_ratio": list(split_ratio),
        "seed": seed,
        "splits": {k: len(v) for k, v in samples.items()},
        "trajectories_per_split": {
            s: sorted({t.key for t in trajectories if splits[t.key] == s})
            for s in ("train", "valid", "test")
        },
        "per_trajectory": per_traj,
        "normalization": norm,
    }
    payload["manifest_hash"] = manifest_hash(samples)
    return {"samples": samples, "meta": payload}


def manifest_hash(samples: dict[str, list[Sample]]) -> str:
    """Hash on dinh cua noi dung manifest, luu vao checkpoint (spec muc 3.2)."""
    h = hashlib.sha256()
    for split in ("train", "valid", "test"):
        for s in samples.get(split, []):
            h.update(f"{s.sample_id}|{s.image_path}|{s.imu_start}|{s.imu_end_exclusive}".encode())
    return h.hexdigest()


def write_manifest(built: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in built["samples"].items():
        path = out_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for s in rows:
                writer.writerow(asdict(s))
    (out_dir / "meta.json").write_text(json.dumps(built["meta"], indent=2), encoding="utf-8")


def read_manifest(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    samples: dict[str, list[Sample]] = {}
    for split in ("train", "valid", "test"):
        path = out_dir / f"{split}.csv"
        rows: list[Sample] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rows.append(
                        Sample(
                            sample_id=row["sample_id"],
                            split=row["split"],
                            environment=row["environment"],
                            difficulty=row["difficulty"],
                            trajectory_id=row["trajectory_id"],
                            camera=row["camera"],
                            image_path=row["image_path"],
                            image_index=int(row["image_index"]),
                            image_time=float(row["image_time"]),
                            imu_start=int(row["imu_start"]),
                            imu_end_exclusive=int(row["imu_end_exclusive"]),
                            imu_start_time=float(row["imu_start_time"]),
                            imu_end_time=float(row["imu_end_time"]),
                            center_error_s=float(row["center_error_s"]),
                        )
                    )
        samples[split] = rows
    return {"samples": samples, "meta": meta}
