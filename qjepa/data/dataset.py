"""Dataset cap (mot frame RGB, mot cua so IMU L hang) - spec muc 3, 4.

Moi item tra ve CA ban clean va ban bad. Clean chi duoc dung cho loss va
teacher; duong online/inference khong bao gio doc no (spec muc 1).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..corruptions.image import ImageCorruptionConfig, ImageCorruptor
from ..corruptions.imu import ImuCorruptionConfig, ImuCorruptor
from .manifest import Sample
from .tartanair import Trajectory


class _TrajectoryCache:
    """Giu IMU va timestamp cua vai trajectory gan nhat trong RAM."""

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self._data: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, traj_dir: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        env, difficulty, traj_id = key.split("/")
        value = Trajectory(traj_dir, env, difficulty, traj_id).load_imu()
        self._data[key] = value
        if len(self._data) > self.size:
            self._data.popitem(last=False)
        return value


def load_image(path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """PNG -> [H,W,3] float32 trong [0,1].

    Nguon da dung size thi giu nguyen. Neu khac, resize canh ngan ve 256 roi
    center-crop (spec muc 4.2) - antialias khi giam do phan giai.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size[1], size[0]):
            w, h = im.size
            scale = max(size[1] / w, size[0] / h)
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            left = (im.width - size[1]) // 2
            top = (im.height - size[0]) // 2
            im = im.crop((left, top, left + size[1], top + size[0]))
        return np.asarray(im, dtype=np.float32) / 255.0


class PairedRestorationDataset(Dataset):
    """Sample da ghep; shuffle o DataLoader chi doi thu tu CAC CAP.

    Args:
        samples: danh sach Sample tu manifest.
        realization: doi theo epoch khi train; co dinh o validation/test.
        force_image_clean / force_imu_clean: ep che do cho cac nhom danh gia
            co ten clean/noisy (spec muc 18.1); None = boc theo xac suat.
    """

    def __init__(
        self,
        samples: list[Sample],
        *,
        image_config: ImageCorruptionConfig | None = None,
        imu_config: ImuCorruptionConfig | None = None,
        master_seed: int = 42,
        realization: int = 0,
        image_size: tuple[int, int] = (256, 256),
        force_image_clean: bool | None = None,
        force_imu_clean: bool | None = None,
        cache_size: int = 4,
    ) -> None:
        if not samples:
            raise ValueError("danh sach sample rong")
        self.samples = samples
        self.image_size = tuple(image_size)
        self.realization = realization
        self.force_image_clean = force_image_clean
        self.force_imu_clean = force_imu_clean
        self.image_corruptor = ImageCorruptor(image_config or ImageCorruptionConfig(), master_seed)
        self.imu_corruptor = ImuCorruptor(imu_config or ImuCorruptionConfig(), master_seed, cache_size)
        self._cache = _TrajectoryCache(cache_size)

    def set_realization(self, realization: int) -> None:
        """Doi realization nhieu theo epoch (chi dung cho train)."""
        self.realization = realization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        s = self.samples[index]
        # Anh luon nam o <trajectory>/image_lcam_front/xxx.png, nen thu muc
        # trajectory la parent.parent - khong phu thuoc co hay khong thu muc split.
        u_all, t_all = self._cache.get(Path(s.image_path).parent.parent, s.trajectory_key)

        imu_clean = np.asarray(u_all[s.imu_start : s.imu_end_exclusive], dtype=np.float64)
        imu_times = np.asarray(t_all[s.imu_start : s.imu_end_exclusive], dtype=np.float64)
        image_clean = load_image(s.image_path, self.image_size)

        image_bad, image_params = self.image_corruptor(
            image_clean,
            split=s.split,
            realization=self.realization,
            trajectory=s.trajectory_key,
            image_time=s.image_time,
            image_index=s.image_index,
            force_clean=self.force_image_clean,
        )
        imu_bad, imu_params = self.imu_corruptor(
            imu_clean,
            split=s.split,
            realization=self.realization,
            trajectory=s.trajectory_key,
            times=t_all,
            start=s.imu_start,
            end=s.imu_end_exclusive,
            force_clean=self.force_imu_clean,
        )

        return {
            # Anh: [3,H,W] channels-first cho model.
            "image_clean": torch.from_numpy(np.ascontiguousarray(image_clean.transpose(2, 0, 1))),
            "image_bad": torch.from_numpy(np.ascontiguousarray(image_bad.transpose(2, 0, 1))),
            # IMU storage layout [L,6]; collate se transpose sang [B,6,L].
            "imu_clean_phys": torch.from_numpy(imu_clean).float(),
            "imu_bad_phys": torch.from_numpy(imu_bad).float(),
            "image_time": torch.tensor(s.image_time, dtype=torch.float64),
            "imu_times": torch.from_numpy(imu_times),
            "imu_start": torch.tensor(s.imu_start, dtype=torch.long),
            "sample_id": s.sample_id,
            "trajectory_key": s.trajectory_key,
            "image_clean_flag": torch.tensor(bool(image_params["clean"])),
            "imu_clean_flag": torch.tensor(bool(imu_params["clean"])),
        }


def collate(batch: list[dict]) -> dict:
    """Gop batch va chuyen IMU tu [B,L,6] sang [B,6,L] (spec muc 15)."""
    out: dict = {}
    for key in (
        "image_clean",
        "image_bad",
        "image_time",
        "imu_times",
        "imu_start",
        "image_clean_flag",
        "imu_clean_flag",
    ):
        out[key] = torch.stack([b[key] for b in batch])
    for key in ("imu_clean_phys", "imu_bad_phys"):
        out[key] = torch.stack([b[key] for b in batch]).transpose(1, 2).contiguous()
    out["image_time"] = out["image_time"].float()
    out["imu_times"] = out["imu_times"].float()
    for key in ("sample_id", "trajectory_key"):
        out[key] = [b[key] for b in batch]
    return out
