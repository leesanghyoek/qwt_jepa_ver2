"""Train decoder danh gia tren backbone DONG BANG (themjacobian muc 13).

Backbone (encoder + fusion + normalizer + transform) bi freeze va o `.eval()`.
Chi hai nhanh cua `RepresentationProbeDecoder` duoc cap nhat. Khong JEPA, khong
encoder sensitivity, khong EMA trong phase nay.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import collate
from ..data.manifest import Sample
from ..evaluation.metrics import image_metrics, imu_metrics
from ..models.joint_jepa import JointRestorationJEPA
from ..models.representation_probe import RepresentationProbeDecoder
from .losses import image_loss, imu_loss
from .trainer import lr_lambda, make_dataset, set_seed


def backbone_fingerprint(model: torch.nn.Module) -> str:
    """Hash parameters + buffers cua backbone de chung minh no khong doi."""
    h = hashlib.sha256()
    for name, tensor in sorted(
        list(model.named_parameters()) + list(model.named_buffers())
    ):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class ProbeTrainer:
    """Train `RepresentationProbeDecoder`; backbone chi dung de sinh ZI/ZU."""

    def __init__(
        self,
        cfg: Config,
        backbone: JointRestorationJEPA,
        samples: dict[str, list[Sample]],
        *,
        device: str | torch.device = "cpu",
        backbone_hash: str = "",
        output_dir: str | None = None,
    ) -> None:
        pc = cfg.representation_probe
        if not pc.enabled:
            raise ValueError("representation_probe.enabled=false")
        cfg.validate()
        self.cfg = cfg
        self.pc = pc
        self.device = torch.device(device)
        self.samples = samples
        self.backbone_hash = backbone_hash
        self.out_dir = Path(output_dir or cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Backbone DONG BANG hoan toan.
        self.backbone = backbone.to(self.device)
        self.backbone.freeze_backbone()
        self.backbone.eval()
        self._frozen_fingerprint = backbone_fingerprint(self.backbone)

        # Decoder moi: cung seed -> control va treatment co cung initial state.
        set_seed(pc.initialization_seed)
        self.probe = RepresentationProbeDecoder(
            backbone.image_transform,
            backbone.imu_transform,
            image_size=tuple(cfg.data.image_size),
            imu_window=cfg.data.imu_window_samples,
            feature_channels=pc.feature_channels,
            decoder_channels=tuple(pc.decoder_channels),
            imu_channels=backbone.imu_channels,
        ).to(self.device)
        self.init_hash = self.probe.state_hash()

        # Optimizer CHI chua tham so cua probe decoder.
        probe_params = [p for p in self.probe.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            probe_params, lr=pc.learning_rate, weight_decay=pc.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: lr_lambda(
                s, pc.max_successful_updates,
                pc.warmup_updates / max(pc.max_successful_updates, 1),
                pc.learning_rate, pc.minimum_lr,
            ),
        )
        self.step = 0
        self.epoch = 0
        self.best: dict[str, float] = {}

    def _latents(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """ZI/ZU tu backbone dong bang. Decoder KHONG nam trong no_grad."""
        with torch.no_grad():
            return self.backbone.encode_and_fuse(
                batch["image_bad"].to(self.device),
                batch["imu_bad_phys"].to(self.device),
                batch["image_time"].to(self.device),
                batch["imu_times"].to(self.device),
            )

    def _loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        zi, zu = self._latents(batch)
        out = self.probe(zi, zu)
        image_clean = batch["image_clean"].to(self.device)
        imu_clean = batch["imu_clean_phys"].to(self.device)
        imu_clean_norm = self.backbone.normalizer.normalize(imu_clean)
        l_img = image_loss(out["image_hat"], image_clean)
        l_imu, l_acc, l_gyro = imu_loss(
            out["imu_hat_norm"], imu_clean_norm, self.cfg.loss.smooth_l1_beta
        )
        total = self.cfg.loss.image_weight * l_img + self.cfg.loss.imu_weight * l_imu
        return total, {
            "loss_total": total.item(), "loss_image": l_img.item(),
            "loss_imu": l_imu.item(), "loss_acc": l_acc.item(), "loss_gyro": l_gyro.item(),
        }

    def train(self, max_steps: int | None = None, log_every: int = 50) -> dict:
        pc = self.pc
        target = max_steps or pc.max_successful_updates
        accum = max(1, pc.gradient_accumulation)
        started = time.time()

        while self.step < target:
            dataset = make_dataset(self.samples["train"], self.cfg, realization=self.epoch)
            loader = DataLoader(
                dataset, batch_size=pc.batch_size, shuffle=True,
                num_workers=self.cfg.train.num_workers, collate_fn=collate,
                pin_memory=(self.device.type == "cuda"),
            )
            self.probe.train()
            self.backbone.eval()          # backbone KHONG bao gio ve train mode
            pending: list[dict] = []
            total_batches, consumed = len(loader), 0
            for batch in loader:
                consumed += 1
                pending.append(batch)
                if len(pending) < accum:
                    continue
                self._update(pending, log_every, started)
                pending = []
                if self.step >= target:
                    break
            finished = consumed == total_batches
            if finished and pending:
                self._update(pending, log_every, started)
            if not finished:
                break
            self.epoch += 1
        self.save("probe_last.pt")
        return {
            "steps": self.step, "epochs": self.epoch, "seconds": time.time() - started,
            "backbone_unchanged": self.verify_backbone_frozen(),
        }

    def _update(self, microbatches: list[dict], log_every: int, started: float) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        n = len(microbatches)
        agg: dict[str, float] = {}
        for mb in microbatches:
            loss, logs = self._loss(mb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"probe loss khong huu han tai step {self.step}")
            (loss / n).backward()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v / n
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.probe.parameters(), self.cfg.train.grad_clip_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1

        if self.step % log_every == 0 or self.step == 1:
            agg.update({
                "step": self.step, "epoch": self.epoch,
                "grad_norm": float(grad_norm),
                "lr": self.optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - started,
            })
            print(
                f"[probe] step {self.step:5d}/{self.pc.max_successful_updates} "
                f"loss {agg['loss_total']:.5f} img {agg['loss_image']:.5f} "
                f"imu {agg['loss_imu']:.5f} gn {agg['grad_norm']:.3f} "
                f"{agg['elapsed_s']:.0f}s", flush=True,
            )
            with (self.out_dir / "probe_train_log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(agg) + "\n")
        if self.pc.validation_every_updates and self.step % self.pc.validation_every_updates == 0:
            self.validate()

    @torch.no_grad()
    def validate(self, split: str = "valid", max_batches: int = 32) -> dict:
        rows = self.samples.get(split) or []
        if not rows:
            return {}
        dataset = make_dataset(
            rows, self.cfg, realization=self.cfg.corruption.validation_realization
        )
        loader = DataLoader(
            dataset, batch_size=self.pc.batch_size, shuffle=False,
            num_workers=self.cfg.train.num_workers, collate_fn=collate,
        )
        self.probe.eval()
        acc: dict[str, list[float]] = {}
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            zi, zu = self._latents(batch)
            out = self.probe(zi, zu)
            image_clean = batch["image_clean"].to(self.device)
            imu_clean = batch["imu_clean_phys"].to(self.device)
            im = image_metrics(out["image_hat"], image_clean)
            um = imu_metrics(self.backbone.normalizer.denormalize(out["imu_hat_norm"]), imu_clean)
            for k, v in im.items():
                if isinstance(v, float) and math.isfinite(v):
                    acc.setdefault(f"image_{k}", []).append(v)
            for k in ("mse_accel", "mse_gyro", "rmse_accel", "rmse_gyro"):
                acc.setdefault(f"imu_{k}", []).append(um[k])
        self.probe.train()
        result = {k: float(np.mean(v)) for k, v in acc.items()}
        result.update({"step": self.step, "system": "frozen_probe"})
        print(
            f"  [probe val {self.step}] psnr {result.get('image_psnr', float('nan')):.3f} "
            f"ssim {result.get('image_ssim', float('nan')):.4f} "
            f"accel_rmse {result.get('imu_rmse_accel', float('nan')):.4f} "
            f"gyro_rmse {result.get('imu_rmse_gyro', float('nan')):.5f}", flush=True,
        )
        with (self.out_dir / "probe_validation.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
        score = result.get("image_mse", float("inf")) + result.get("imu_mse_accel", 0.0)
        if "best" not in self.best or score < self.best["best"]:
            self.best["best"] = score
            self.save("probe_best_joint_validation.pt")
        return result

    def verify_backbone_frozen(self) -> bool:
        """Backbone phai KHONG doi sau moi optimizer step cua probe."""
        return backbone_fingerprint(self.backbone) == self._frozen_fingerprint

    def save(self, name: str) -> Path:
        payload = {
            "kind": "representation_probe",
            "probe": self.probe.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step, "epoch": self.epoch,
            "backbone_hash": self.backbone_hash,
            "frozen_fingerprint": self._frozen_fingerprint,
            "probe_init_hash": self.init_hash,
            "config": self.cfg.to_dict(),
        }
        path = self.out_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path
