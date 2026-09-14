"""Gate G5: resume tai epoch boundary phai trung khop tung bit.

Chay tren CPU: GPU khong deterministic (cuDNN chon thuat toan convolution khac
nhau giua cac lan chay), nen so sanh tren GPU do duoc nhieu phan cung chu khong
phai logic resume. Xem TRAINING_REPORT muc 3.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from qjepa.config import load_config
from qjepa.data.manifest import read_manifest
from qjepa.data.normalize import ImuNormalizer
from qjepa.training.trainer import Trainer, set_seed

import pytest

MANIFEST = "outputs/manifest_local"


def _available() -> bool:
    from pathlib import Path

    return (Path(MANIFEST) / "meta.json").exists()


pytestmark = pytest.mark.skipif(
    not _available(), reason=f"can manifest tai {MANIFEST} (chay build-manifest truoc)"
)


def _run(tmp_path, name, total, resume_from=None, samples=16):
    built = read_manifest(MANIFEST)
    stats = built["meta"]["normalization"]
    subset = built["samples"]["train"][:samples]
    cfg = load_config("configs/main_qwt.yaml")
    cfg = replace(
        cfg,
        output_dir=str(tmp_path / name),
        train=replace(
            cfg.train,
            stage_a_optimizer_steps=40,
            stage_b_optimizer_steps=0,
            max_optimizer_steps=40,
            validation_every_updates=0,
            checkpoint_every_updates=0,
            batch_size=2,
            gradient_accumulation=1,
        ),
    )
    set_seed(cfg.seed)
    normalizer = ImuNormalizer(
        mean=torch.tensor(stats["mean"]).float(), std=torch.tensor(stats["std"]).float()
    )
    trainer = Trainer(
        cfg,
        {"train": subset, "valid": subset, "test": []},
        normalizer,
        device="cpu",
        manifest_hash=built["meta"]["manifest_hash"],
    )
    if resume_from:
        trainer.load_checkpoint(resume_from)
    trainer.train(max_steps=total, log_every=10**6)
    return trainer


def _max_param_diff(a, b) -> float:
    sa, sb = a.model.state_dict(), b.model.state_dict()
    return max(
        (sa[k].float() - sb[k].float()).abs().max().item()
        for k in sa
        if sa[k].dtype.is_floating_point
    )


def test_resume_at_epoch_boundary_is_bit_exact(tmp_path):
    # 16 sample, batch 2 -> mot epoch dung 8 step, nen step 8 la epoch boundary.
    first = _run(tmp_path, "a", 8)
    checkpoint = first.out_dir / "last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["resume_exact"] is True
    assert payload["step"] == 8

    resumed = _run(tmp_path, "a", 16, resume_from=checkpoint)
    continuous = _run(tmp_path, "b", 16)
    assert resumed.step == continuous.step == 16
    assert _max_param_diff(resumed, continuous) == 0.0


def test_checkpoint_records_mid_epoch_as_not_exact(tmp_path):
    """Dung giua epoch phai duoc danh dau resume_exact=false, khong hua hen sai."""
    trainer = _run(tmp_path, "c", 5)          # 5 < 8 step cua mot epoch
    payload = torch.load(trainer.out_dir / "last.pt", map_location="cpu", weights_only=False)
    assert payload["resume_exact"] is False
    assert payload["steps_into_epoch"] == 5
    assert "phat lai" in payload["data_replay_point"]


def test_resume_rejects_different_manifest(tmp_path):
    trainer = _run(tmp_path, "d", 8)
    checkpoint = trainer.out_dir / "last.pt"
    other = _run(tmp_path, "e", 0)
    other.manifest_hash = "khac"
    with pytest.raises(ValueError, match="manifest_hash"):
        other.load_checkpoint(checkpoint)


def test_resume_rejects_different_image_transform(tmp_path):
    trainer = _run(tmp_path, "f", 8)
    checkpoint = trainer.out_dir / "last.pt"
    other = _run(tmp_path, "g", 0)
    other.cfg = replace(other.cfg, transform=replace(other.cfg.transform, image="dwt_haar_baseline"))
    with pytest.raises(ValueError, match="transform"):
        other.load_checkpoint(checkpoint)
