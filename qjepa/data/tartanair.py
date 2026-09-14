"""Doc trajectory TartanAir-v2 va audit timestamp (spec muc 2.4).

Ho tro ca hai bo cuc thu muc:

    <root>/<env>/<Data_easy|Data_hard>/<Pxxx>/...          (output Kaggle da resize)
    <root>/<train|valid|test>/<env>/<Data_x>/<Pxxx>/...    (folder view co san)

Mot trajectory hop le phai co `imu/imu_time.npy`, `imu/acc.npy`, `imu/gyro.npy`,
`imu/cam_time.npy` va thu muc anh.

Dung `acc.npy` (specific force, CO trong luc) va `gyro.npy` lam target. Khong
tu bo trong luc va khong vi phan pose tho de tao target gia (spec muc 2.4).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

def load_array(imu_dir: Path, name: str) -> np.ndarray:
    """Doc `name`.npy neu co, nguoc lai `name`.txt.

    Notebook resize tren Kaggle CHI copy .png va .txt (bo qua .npy), nen dataset
    da resize khong co file .npy. Hai dinh dang chua cung so lieu float64.
    """
    npy, txt = imu_dir / f"{name}.npy", imu_dir / f"{name}.txt"
    if npy.is_file():
        return np.load(npy)
    if txt.is_file():
        return np.loadtxt(txt, dtype=np.float64)
    raise FileNotFoundError(f"thieu ca {npy.name} lan {txt.name} trong {imu_dir}")


def has_imu_timeline(imu_dir: Path) -> bool:
    return (imu_dir / "imu_time.npy").is_file() or (imu_dir / "imu_time.txt").is_file()


IMAGE_DIRNAME = "image_lcam_front"
CAMERA_ID = "lcam_front"
SPLIT_DIRS = ("train", "valid", "val", "test")
DIFFICULTIES = ("Data_easy", "Data_hard")


@dataclass(frozen=True)
class Trajectory:
    """Mot timeline lien tuc; cua so IMU khong bao gio vuot qua ranh gioi nay."""

    root: Path
    environment: str
    difficulty: str
    trajectory_id: str
    split_hint: str | None = None

    @property
    def key(self) -> str:
        """ID day du; ten Pxxx lap lai giua cac environment (spec muc 3.2)."""
        return f"{self.environment}/{self.difficulty}/{self.trajectory_id}"

    @property
    def imu_dir(self) -> Path:
        return self.root / "imu"

    @property
    def image_dir(self) -> Path:
        return self.root / IMAGE_DIRNAME

    def image_paths(self) -> list[Path]:
        return sorted(self.image_dir.glob(f"*_{CAMERA_ID}.png"))

    def load_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """-> (u[N,6] float64 don vi SI, t[N] float64 giay)."""
        acc = load_array(self.imu_dir, "acc").astype(np.float64)
        gyro = load_array(self.imu_dir, "gyro").astype(np.float64)
        t = load_array(self.imu_dir, "imu_time").astype(np.float64)
        if acc.ndim != 2 or acc.shape[1] != 3 or gyro.shape != acc.shape:
            raise ValueError(f"{self.key}: acc/gyro phai la [N,3], nhan {acc.shape}/{gyro.shape}")
        if t.ndim != 1 or len(t) != len(acc):
            raise ValueError(f"{self.key}: imu_time [{len(t)}] khong khop acc [{len(acc)}]")
        return np.concatenate([acc, gyro], axis=1), t

    def load_camera_times(self) -> np.ndarray:
        return load_array(self.imu_dir, "cam_time").astype(np.float64)


def discover_trajectories(root: str | Path) -> list[Trajectory]:
    """Quet `root` tim moi trajectory hop le, sap xep on dinh."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root khong ton tai: {root}")

    found: list[Trajectory] = []
    seen: set[Path] = set()
    # Folder view cua TartanAir dung symlink o muc Pxxx; Path.rglob KHONG di qua
    # symlink directory nen phai dung os.walk(followlinks=True).
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        resolved = Path(dirpath).resolve()
        if resolved in seen:          # chan vong lap do symlink
            dirnames[:] = []
            continue
        seen.add(resolved)
        if Path(dirpath).name != "imu" or not has_imu_timeline(Path(dirpath)):
            continue
        traj_dir = Path(dirpath).parent
        if not (traj_dir / IMAGE_DIRNAME).is_dir():
            continue
        rel = traj_dir.relative_to(root).parts
        if len(rel) < 3:
            continue
        trajectory_id, difficulty, environment = rel[-1], rel[-2], rel[-3]
        split_hint = None
        for part in rel:
            if part in SPLIT_DIRS:
                split_hint = "valid" if part == "val" else part
                break
        found.append(
            Trajectory(traj_dir, environment, difficulty, trajectory_id, split_hint)
        )
    if not found:
        raise FileNotFoundError(
            f"khong tim thay trajectory nao duoi {root}; can "
            f"<...>/<Pxxx>/imu/imu_time.npy va <...>/<Pxxx>/{IMAGE_DIRNAME}/"
        )
    return sorted(found, key=lambda t: t.key)


def audit_trajectory(traj: Trajectory, window: int = 128) -> dict:
    """Audit mot trajectory theo spec muc 2.4; khong suy rate tu row count."""
    u, t = traj.load_imu()
    cam_t = traj.load_camera_times()
    images = traj.image_paths()

    dt = np.diff(t)
    med = float(np.median(dt)) if len(dt) else float("nan")
    cam_dt = np.diff(cam_t)
    cam_med = float(np.median(cam_dt)) if len(cam_dt) else float("nan")

    report = {
        "key": traj.key,
        "environment": traj.environment,
        "difficulty": traj.difficulty,
        "trajectory_id": traj.trajectory_id,
        "split_hint": traj.split_hint,
        "imu_rows": int(len(u)),
        "measurement_scalars": int(u.size),
        "num_images": len(images),
        "num_camera_times": int(len(cam_t)),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "imu_median_dt_s": med,
        "imu_median_rate_hz": float(1.0 / med) if med > 0 else float("nan"),
        "camera_median_dt_s": cam_med,
        "camera_median_rate_hz": float(1.0 / cam_med) if cam_med > 0 else float("nan"),
        "imu_max_relative_dt_deviation": (
            float(np.max(np.abs(dt - med)) / med) if len(dt) and med > 0 else float("nan")
        ),
        "strictly_increasing": bool(np.all(dt > 0)) if len(dt) else False,
        "finite": bool(np.isfinite(u).all() and np.isfinite(t).all()),
        "enough_rows": bool(len(u) >= window),
        "window_rows": window,
        "uniform_window_duration_s": float((window - 1) * med) if med > 0 else float("nan"),
        "candidate_windows_stride_1": max(0, len(u) - window + 1),
        "images_match_camera_times": len(images) == len(cam_t),
        "acc_abs_max": u[:, :3].__abs__().max(axis=0).tolist(),
        "gyro_abs_max": u[:, 3:].__abs__().max(axis=0).tolist(),
    }
    return report


def audit_dataset(root: str | Path, window: int = 128) -> dict:
    """Audit toan bo root; tra ve tong hop de ghi DATA_AUDIT.md."""
    trajectories = discover_trajectories(root)
    per_traj = [audit_trajectory(t, window) for t in trajectories]
    rates = [r["imu_median_rate_hz"] for r in per_traj if np.isfinite(r["imu_median_rate_hz"])]
    cam_rates = [r["camera_median_rate_hz"] for r in per_traj if np.isfinite(r["camera_median_rate_hz"])]
    return {
        "root": str(root),
        "num_trajectories": len(per_traj),
        "num_environments": len({r["environment"] for r in per_traj}),
        "environments": sorted({r["environment"] for r in per_traj}),
        "total_imu_rows": int(sum(r["imu_rows"] for r in per_traj)),
        "total_images": int(sum(r["num_images"] for r in per_traj)),
        "total_duration_s": float(sum(r["duration_s"] for r in per_traj)),
        "imu_rate_hz_median": float(np.median(rates)) if rates else float("nan"),
        "camera_rate_hz_median": float(np.median(cam_rates)) if cam_rates else float("nan"),
        "all_strictly_increasing": all(r["strictly_increasing"] for r in per_traj),
        "all_finite": all(r["finite"] for r in per_traj),
        "all_images_match_camera_times": all(r["images_match_camera_times"] for r in per_traj),
        "trajectories_without_enough_rows": [r["key"] for r in per_traj if not r["enough_rows"]],
        "window_rows": window,
        "per_trajectory": per_traj,
    }


def write_audit(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
