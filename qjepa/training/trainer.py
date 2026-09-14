"""Vong huan luyen hai stage, checkpoint va resume (spec muc 14, 20).

Stage A: chi phuc hoi (lambda_J = 0).
Stage B: khoi tao teacher, bat JEPA theo ramp, cap nhat EMA sau MOI optimizer
step thanh cong.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..config import Config, config_hash, save_resolved
from . import distributed as dstr
from ..corruptions.image import ImageCorruptionConfig
from ..corruptions.imu import ImuCorruptionConfig
from ..data.dataset import PairedRestorationDataset, collate
from ..data.manifest import Sample
from ..data.normalize import ImuNormalizer
from ..evaluation.metrics import error_reduction, image_metrics, imu_metrics
from ..models.joint_jepa import (
    JointRestorationJEPA,
    image_tokens,
    imu_tokens,
    teacher_momentum,
)
from .encoder_sensitivity import (
    EncoderSensitivityConfig,
    encoder_fd_loss,
    encoder_weight,
    make_encoder_probe,
    raw_feature_fd_gain,
    seeded_rademacher,
    source_for_update,
)
from .losses import (
    LossWeights,
    edge_loss,
    image_loss,
    imu_delta_loss,
    imu_loss,
    jepa_loss,
    jepa_weight,
    variance_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config, normalizer: ImuNormalizer | None = None) -> JointRestorationJEPA:
    return JointRestorationJEPA(
        image_transform=cfg.transform.image,
        imu_transform=cfg.transform.imu,
        encoder_channels=tuple(cfg.model.encoder_channels),
        embedding_dim=cfg.model.embedding_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        predictor_hidden_dim=cfg.model.predictor_hidden_dim,
        imu_summary_bins=cfg.model.imu_summary_bins,
        time_metadata_dim=cfg.model.time_metadata_dim,
        cross_modal=cfg.model.cross_modal,
        gate_bias_init=cfg.model.gate_bias_init,
        groupnorm_groups=cfg.model.groupnorm_groups,
        imu_window=cfg.data.imu_window_samples,
        image_size=tuple(cfg.data.image_size),
        normalizer=normalizer,
    )


def parameter_groups(model: JointRestorationJEPA, weight_decay: float, no_decay_bias_norm: bool):
    """Tach nhom: khong weight decay cho bias va affine cua normalization."""
    if not no_decay_bias_norm:
        return [{"params": model.online_parameters(), "weight_decay": weight_decay}]
    decay, no_decay = [], []
    teacher_ids = set()
    for t in (model.teacher_image_encoder, model.teacher_imu_encoder):
        if t is not None:
            teacher_ids.update(id(p) for p in t.parameters())
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in teacher_ids:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_lambda(step: int, total: int, warmup_fraction: float, base_lr: float, min_lr: float):
    """Warmup tuyen tinh roi cosine; tinh theo optimizer update THANH CONG."""
    warmup = max(1, int(round(warmup_fraction * total)))
    if step < warmup:
        return (step + 1) / warmup
    # Chia cho (total - warmup - 1) de step CUOI dat dung min_lr, khong dung gan.
    progress = min((step - warmup) / max(total - warmup - 1, 1), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return (min_lr + (base_lr - min_lr) * cosine) / base_lr


def make_dataset(
    samples: list[Sample], cfg: Config, *, realization: int, force_image_clean=None, force_imu_clean=None
) -> PairedRestorationDataset:
    ic = cfg.corruption.image
    uc = cfg.corruption.imu
    return PairedRestorationDataset(
        samples,
        image_config=ImageCorruptionConfig(
            blur_sigma_px=tuple(ic.blur_sigma_px),
            gaussian_std=tuple(ic.gaussian_std),
            downsample_probability=ic.downsample_probability,
            downsample_scale=tuple(ic.downsample_scale),
            jpeg_probability=ic.jpeg_probability,
            clean_probability=cfg.corruption.clean_probability_each_modality,
        ),
        imu_config=ImuCorruptionConfig(
            accel_noise_std=tuple(uc.accel_noise_std),
            gyro_noise_std=tuple(uc.gyro_noise_std),
            bias_enabled=uc.bias_enabled,
            accel_bias_bound=uc.accel_bias_bound,
            gyro_bias_bound=uc.gyro_bias_bound,
            drift_enabled=uc.drift_enabled,
            accel_bias_random_walk=uc.accel_bias_random_walk,
            gyro_bias_random_walk=uc.gyro_bias_random_walk,
            spike_enabled=uc.spike_enabled,
            spike_rate_per_second_per_triplet=uc.spike_rate_per_second_per_triplet,
            accel_spike_amplitude=tuple(uc.accel_spike_amplitude),
            gyro_spike_amplitude=tuple(uc.gyro_spike_amplitude),
            clean_probability=cfg.corruption.clean_probability_each_modality,
        ),
        master_seed=cfg.seed,
        realization=realization,
        image_size=tuple(cfg.data.image_size),
        force_image_clean=force_image_clean,
        force_imu_clean=force_imu_clean,
    )


class Trainer:
    """Dieu phoi train/validate/checkpoint cho mot run."""

    def __init__(
        self,
        cfg: Config,
        samples: dict[str, list[Sample]],
        normalizer: ImuNormalizer,
        *,
        device: str | torch.device = "cpu",
        manifest_hash: str = "",
        dist: dstr.DistInfo | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.dist = dist or dstr.DistInfo()
        self.device = torch.device(device)
        self.samples = samples
        self.manifest_hash = manifest_hash
        self.out_dir = Path(cfg.output_dir)
        if self.dist.is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # Cung seed tren moi rank -> trong so khoi tao giong nhau truoc khi DDP dong bo.
        set_seed(cfg.seed)
        self.model = build_model(cfg, normalizer).to(self.device)
        self.ddp = None
        if self.dist.enabled:
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank] if self.device.type == "cuda" else None,
                output_device=self.dist.local_rank if self.device.type == "cuda" else None,
                # O Stage A hai predictor KHONG tham gia loss (lambda_J = 0) nen
                # DDP can co nay. Run chay THANG vao Stage B (stage_a = 0, vi du
                # cac phase v2) thi predictor luon duoc dung -> tat di de bo mot
                # lan duyet autograd graph moi iteration.
                find_unused_parameters=cfg.train.stage_a_optimizer_steps > 0,
            )
        # Giu effective batch khong doi khi so GPU thay doi.
        self.accum, note = dstr.resolve_accumulation(
            max(1, cfg.train.gradient_accumulation), self.dist.world_size
        )
        if note and self.dist.is_main:
            print(f"  [dist] {note}")
        if self.dist.is_main and self.dist.enabled:
            eff = cfg.train.batch_size * self.accum * self.dist.world_size
            print(f"  [dist] {self.dist.world_size} GPU | batch/rank {cfg.train.batch_size} "
                  f"| accum {self.accum} | effective batch {eff}")
        self.optimizer = torch.optim.AdamW(
            parameter_groups(self.model, cfg.train.weight_decay, cfg.train.no_decay_bias_and_norm),
            lr=cfg.train.learning_rate,
            betas=tuple(cfg.train.betas),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: lr_lambda(
                s,
                cfg.train.max_optimizer_steps,
                cfg.train.warmup_fraction,
                cfg.train.learning_rate,
                cfg.train.minimum_lr,
            ),
        )
        self.amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[
            cfg.train.precision
        ]
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=(cfg.train.precision == "fp16")
        )
        self.weights = LossWeights(
            image=cfg.loss.image_weight,
            imu=cfg.loss.imu_weight,
            edge=cfg.loss.edge_weight,
            imu_delta=cfg.loss.imu_delta_weight,
            variance=cfg.loss.variance_weight,
            variance_gamma=cfg.loss.variance_gamma,
            smooth_l1_beta=cfg.loss.smooth_l1_beta,
        )
        es = cfg.encoder_sensitivity
        self.sens_cfg = EncoderSensitivityConfig(
            enabled=es.enabled,
            image_epsilon=es.image_epsilon,
            imu_normalized_epsilon=es.imu_normalized_epsilon,
            alpha=es.alpha,
            layer_norm_eps=es.layer_norm_eps,
            minimum_input_energy=es.minimum_input_energy,
            weight_max=es.weight_max,
            ramp_updates=es.ramp_updates,
            probe_seed=es.probe_seed,
            modality_multipliers=tuple(es.modality_multipliers),
        )
        self.step = 0              # optimizer update THANH CONG
        self.jepa_steps = 0        # update co JEPA, dung cho EMA momentum
        self.sens_steps = 0        # update co encoder sensitivity (ramp rieng)
        self.epoch = 0
        self.steps_in_epoch = 0    # de biet checkpoint co nam o epoch boundary khong
        self.best: dict[str, float] = {}
        self.history: list[dict] = []
        if self.dist.is_main:
            save_resolved(cfg, self.out_dir / "resolved_config.yaml")

    # -- stage ------------------------------------------------------------
    @property
    def stage(self) -> str:
        return "A" if self.step < self.cfg.train.stage_a_optimizer_steps else "B"

    def current_jepa_weight(self) -> float:
        if self.stage == "A":
            return 0.0
        local = self.step - self.cfg.train.stage_a_optimizer_steps
        return jepa_weight(
            local,
            self.cfg.loss.jepa_ramp_updates,
            self.cfg.loss.jepa_start_weight,
            self.cfg.loss.jepa_max_weight,
        )

    def ensure_teacher(self) -> None:
        if self.stage == "B" and not bool(self.model.teacher_initialized):
            self.model.initialize_teacher()
            self.model.to(self.device)

    # -- loss -------------------------------------------------------------
    def compute_loss(
        self,
        batch: dict,
        lambda_j: float,
        *,
        sens_source: str | None = None,
        lambda_enc: float = 0.0,
        microbatch_index: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        image_bad = batch["image_bad"].to(self.device, non_blocking=True)
        image_clean = batch["image_clean"].to(self.device, non_blocking=True)
        imu_bad = batch["imu_bad_phys"].to(self.device, non_blocking=True)
        imu_clean = batch["imu_clean_phys"].to(self.device, non_blocking=True)
        image_time = batch["image_time"].to(self.device, non_blocking=True)
        imu_times = batch["imu_times"].to(self.device, non_blocking=True)

        # Duong forward CHINH phai di qua wrapper DDP, neu khong reducer khong
        # bao gio all-reduce gradient va moi rank se train doc lap.
        net = self.ddp if self.ddp is not None else self.model
        imu_clean_norm = self.model.normalizer.normalize(imu_clean)

        # Chuan bi duong PHU truoc, de tat ca di chung MOT lan goi forward:
        # DDP yeu cau moi tham so can gradient phai duoc dung ben trong forward.
        if lambda_j > 0:
            self.ensure_teacher()
        probe_kw, sens_meta = {}, None
        if sens_source is not None and lambda_enc > 0:
            probe_kw, sens_meta = self._prepare_probe(
                batch, imu_bad, sens_source, microbatch_index
            )
        out = net(
            image_bad, imu_bad, image_time, imu_times,
            predict_latents=(lambda_j > 0), **probe_kw,
        )

        l_image = image_loss(out["image_hat"], image_clean)
        l_imu, l_acc, l_gyro = imu_loss(
            out["imu_hat_norm"], imu_clean_norm, self.weights.smooth_l1_beta
        )
        total = self.weights.image * l_image + self.weights.imu * l_imu
        logs = {
            "loss_image": l_image.item(),
            "loss_imu": l_imu.item(),
            "loss_acc": l_acc.item(),
            "loss_gyro": l_gyro.item(),
        }
        # Chi tinh loss phu khi weight > 0 (muc 13: tranh 0*NaN va compute thua).
        if self.weights.edge > 0:
            l_edge = edge_loss(out["image_hat"], image_clean)
            total = total + self.weights.edge * l_edge
            logs["loss_edge"] = l_edge.item()
        if self.weights.imu_delta > 0:
            l_delta = imu_delta_loss(out["imu_hat_norm"], imu_clean_norm)
            total = total + self.weights.imu_delta * l_delta
            logs["loss_imu_delta"] = l_delta.item()
        if self.weights.variance > 0:
            l_var = 0.5 * (
                variance_loss(image_tokens(out["fi"]), self.weights.variance_gamma)
                + variance_loss(imu_tokens(out["fu"]), self.weights.variance_gamma)
            )
            total = total + self.weights.variance * l_var
            logs["loss_variance"] = l_var.item()

        if lambda_j > 0:
            ti, tu = self.model.teacher_targets(image_clean, imu_clean)
            l_jepa, ji, ju = jepa_loss(
                out["pred_image"],
                image_tokens(ti),
                out["pred_imu"],
                imu_tokens(tu),
                beta=self.weights.smooth_l1_beta,
                eps=self.cfg.loss.jepa_target_norm_eps,
            )
            total = total + lambda_j * l_jepa
            logs.update({"loss_jepa": l_jepa.item(), "jepa_image": ji.item(), "jepa_imu": ju.item()})

        if sens_meta is not None:
            l_enc, enc_logs = self._encoder_sensitivity_term(out, sens_source, sens_meta)
            multiplier = dict(zip(("image", "imu"), self.sens_cfg.modality_multipliers))[sens_source]
            total = total + lambda_enc * multiplier * l_enc
            logs.update(enc_logs)

        logs["loss_total"] = total.item()
        logs["lambda_jepa"] = lambda_j
        return total, logs

    def _prepare_probe(
        self, batch: dict, imu_bad: torch.Tensor, source: str, microbatch_index: int
    ) -> tuple[dict, dict]:
        """Tao dau vao perturbed. Goi TRUOC forward de probe di chung mot lan goi."""
        cfg = self.sens_cfg
        if source == "image":
            x = batch["image_bad"].to(self.device, non_blocking=True)
        else:
            # Perturb IMU o mien NORMALIZED — mien ma encoder that su nhan.
            x = self.model.normalizer.normalize(imu_bad)
        context = (
            self.cfg.experiment_name, self.step, self.epoch,
            batch["sample_id"][0], microbatch_index, source,
        )
        direction = seeded_rademacher(x, probe_seed=cfg.probe_seed, context=context)
        xp, energy, clipped = make_encoder_probe(
            x, source, direction,
            image_epsilon=cfg.image_epsilon,
            imu_epsilon=cfg.imu_normalized_epsilon,
            alpha=cfg.alpha,
            minimum_energy=cfg.minimum_input_energy,
        )
        key = "probe_image" if source == "image" else "probe_imu_norm"
        return {key: xp}, {"energy": energy, "clipped": clipped}

    def _encoder_sensitivity_term(
        self, out: dict, source: str, meta: dict
    ) -> tuple[torch.Tensor, dict]:
        """FD sensitivity tai FI/FU truoc fusion (themjacobian muc 6, 9.1)."""
        cfg = self.sens_cfg
        f_base = out["fi"] if source == "image" else out["fu"]
        f_probe = out["probe_fi"] if source == "image" else out["probe_fu"]
        energy = meta["energy"]
        l_enc, gain = encoder_fd_loss(f_base, f_probe, energy, ln_eps=cfg.layer_norm_eps)
        raw_gain = raw_feature_fd_gain(f_base, f_probe, energy)
        return l_enc, {
            "sens_source": source,
            "sens_gain_normalized": float(gain.mean()),
            "sens_gain_raw": float(raw_gain.mean()),
            "sens_input_energy": float(energy.mean()),
            "sens_clipped_fraction": meta["clipped"],
        }

    # -- train ------------------------------------------------------------
    def train(self, max_steps: int | None = None, log_every: int = 25) -> dict:
        cfg = self.cfg
        target = max_steps or cfg.train.max_optimizer_steps
        accum = self.accum
        started = time.time()

        while self.step < target:
            dataset = make_dataset(
                self.samples["train"], cfg, realization=self.epoch
            )
            sampler = None
            if self.dist.enabled:
                # Moi rank nhan mot phan khac nhau; set_epoch de shuffle doi moi epoch.
                sampler = DistributedSampler(
                    dataset, num_replicas=self.dist.world_size, rank=self.dist.rank,
                    shuffle=cfg.data.train_shuffle_windows, drop_last=True,
                )
                sampler.set_epoch(self.epoch)
            loader = DataLoader(
                dataset,
                batch_size=cfg.train.batch_size,
                shuffle=(sampler is None and cfg.data.train_shuffle_windows),
                sampler=sampler,
                num_workers=cfg.train.num_workers,
                collate_fn=collate,
                drop_last=self.dist.enabled,
                pin_memory=(self.device.type == "cuda"),
            )
            self.model.train()
            self.steps_in_epoch = 0
            total_batches, consumed = len(loader), 0
            pending: list[dict] = []
            for batch in loader:
                consumed += 1
                pending.append(batch)
                if len(pending) < accum:
                    continue
                self._optimizer_step(pending, log_every, started)
                pending = []
                if self.step >= target:
                    break
            # Epoch chi HOAN THANH khi da duyet het loader. Dat target dung o
            # batch cuoi cung van la hoan thanh; dat target som hon thi khong.
            finished_epoch = consumed == total_batches
            if finished_epoch and pending:
                # Nhom cuoi it hon accum: chia dung so microbatch THUC (muc 14.2).
                self._optimizer_step(pending, log_every, started)
            if not finished_epoch:
                break
            self.epoch += 1
            self.steps_in_epoch = 0
            # Epoch boundary: diem duy nhat resume duoc chinh xac.
            if self.dist.is_main:
                self.save_checkpoint("last.pt")
        if self.dist.is_main:
            self.save_checkpoint("last.pt")
        dstr.barrier()
        return {"steps": self.step, "epochs": self.epoch, "seconds": time.time() - started,
                "world_size": self.dist.world_size}

    def _optimizer_step(self, microbatches: list[dict], log_every: int, started: float) -> None:
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)
        lambda_j = self.current_jepa_weight()
        # Ghi lai stage TAI LUC tinh loss: self.step tang o cuoi ham, nen doc
        # self.stage sau do se bao sai stage cho chinh step vua chay.
        stage_of_this_step = self.stage
        # Mot source cho MOI optimizer update; moi microbatch trong cung
        # accumulation group dung cung source (themjacobian muc 7).
        if self.sens_cfg.enabled and self.sens_cfg.weight_max > 0:
            sens_source = source_for_update(self.sens_steps)
            lambda_enc = encoder_weight(self.sens_steps, self.sens_cfg)
        else:
            sens_source, lambda_enc = None, 0.0
        n = len(microbatches)
        agg: dict[str, float] = {}
        sens_text: dict[str, str] = {}
        import contextlib

        for idx, mb in enumerate(microbatches):
            kw = dict(sens_source=sens_source, lambda_enc=lambda_enc, microbatch_index=idx)
            # Chi all-reduce gradient o microbatch CUOI cua nhom accumulation.
            last = idx == n - 1
            sync_ctx = (
                self.ddp.no_sync() if (self.ddp is not None and not last)
                else contextlib.nullcontext()
            )
            with sync_ctx:
                if self.amp_dtype is not None:
                    with torch.autocast(self.device.type, dtype=self.amp_dtype):
                        loss, logs = self.compute_loss(mb, lambda_j, **kw)
                else:
                    loss, logs = self.compute_loss(mb, lambda_j, **kw)
                sens_text.update({k: v for k, v in logs.items() if isinstance(v, str)})
                logs = {k: v for k, v in logs.items() if not isinstance(v, str)}
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"loss khong huu han tai step {self.step}: {loss.item()}"
                    )
                self.scaler.scale(loss / n).backward()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v / n

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.online_parameters(), cfg.train.grad_clip_norm
        )
        before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # fp16 co the BO step; khi do khong dem step, khong scheduler, khong EMA.
        did_step = self.scaler.get_scale() >= before or self.amp_dtype is not torch.float16
        self.optimizer.zero_grad(set_to_none=True)

        if not did_step:
            return
        self.scheduler.step()
        self.step += 1
        self.steps_in_epoch += 1
        if sens_source is not None:
            # Counter rieng: ramp va luan phien source theo update THANH CONG.
            self.sens_steps += 1
        if lambda_j > 0:
            momentum = teacher_momentum(
                self.jepa_steps,
                cfg.train.stage_b_optimizer_steps,
                cfg.train.teacher_momentum_start,
                cfg.train.teacher_momentum_end,
            )
            self.model.update_teacher(momentum)
            self.jepa_steps += 1

        agg.update(
            {
                "step": self.step,
                "epoch": self.epoch,
                "stage": stage_of_this_step,
                "grad_norm": float(grad_norm),
                "lambda_encoder": lambda_enc,
                **sens_text,
                "lr": self.optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - started,
            }
        )
        if (self.step % log_every == 0 or self.step == 1) and self.dist.is_main:
            self.history.append(agg)
            print(
                f"[{agg['stage']}] step {self.step:6d}/{cfg.train.max_optimizer_steps} "
                f"loss {agg['loss_total']:.5f} img {agg['loss_image']:.5f} "
                f"imu {agg['loss_imu']:.5f} lam_J {lambda_j:.3f} "
                f"gn {agg['grad_norm']:.3f} lr {agg['lr']:.2e} "
                f"{agg['elapsed_s']:.0f}s",
                flush=True,
            )
            self._append_jsonl("train_log.jsonl", agg)
        if cfg.train.validation_every_updates and self.step % cfg.train.validation_every_updates == 0:
            self.validate()
        if (cfg.train.checkpoint_every_updates
                and self.step % cfg.train.checkpoint_every_updates == 0
                and self.dist.is_main):
            self.save_checkpoint("last.pt")

    # -- validation -------------------------------------------------------
    @torch.no_grad()
    def validate(self, split: str = "valid", max_batches: int | None = None) -> dict:
        cfg = self.cfg
        rows = self.samples.get(split) or []
        if not rows:
            return {}
        dataset = make_dataset(rows, cfg, realization=cfg.corruption.validation_realization)
        loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            collate_fn=collate,
        )
        limit = max_batches or cfg.train.validation_batches
        self.model.eval()
        acc: dict[str, list[float]] = {}
        for i, batch in enumerate(loader):
            if i >= limit:
                break
            image_bad = batch["image_bad"].to(self.device)
            image_clean = batch["image_clean"].to(self.device)
            imu_bad = batch["imu_bad_phys"].to(self.device)
            imu_clean = batch["imu_clean_phys"].to(self.device)
            out = self.model(
                image_bad, imu_bad, batch["image_time"].to(self.device), batch["imu_times"].to(self.device)
            )
            im = image_metrics(
                out["image_hat"], image_clean,
                data_range=cfg.evaluation.image_data_range,
                clamp=cfg.evaluation.image_metric_clamp,
                border_crop=cfg.evaluation.image_border_crop,
            )
            im_bad = image_metrics(
                image_bad, image_clean,
                data_range=cfg.evaluation.image_data_range,
                clamp=cfg.evaluation.image_metric_clamp,
                border_crop=cfg.evaluation.image_border_crop,
            )
            um = imu_metrics(out["imu_hat_phys"], imu_clean)
            um_bad = imu_metrics(imu_bad, imu_clean)
            for k, v in im.items():
                if math.isfinite(v):
                    acc.setdefault(f"image_{k}", []).append(v)
            acc.setdefault("image_mse_bad", []).append(im_bad["mse"])
            for k in ("mse_accel", "mse_gyro", "rmse_accel", "rmse_gyro", "mae_accel", "mae_gyro"):
                acc.setdefault(f"imu_{k}", []).append(um[k])
                acc.setdefault(f"imu_{k}_bad", []).append(um_bad[k])
        self.model.train()

        result = {k: float(np.mean(v)) for k, v in acc.items()}
        result["step"] = self.step
        result["stage"] = self.stage
        # Ty so so voi dau vao nhieu: <1 nghia la tot hon identity (muc 18.5).
        result["r_image"] = result["image_mse"] / max(result["image_mse_bad"], 1e-12)
        result["r_acc"] = result["imu_mse_accel"] / max(result["imu_mse_accel_bad"], 1e-12)
        result["r_gyro"] = result["imu_mse_gyro"] / max(result["imu_mse_gyro_bad"], 1e-12)
        result["joint_score"] = float(
            np.mean([math.log(max(result[k], 1e-12)) for k in ("r_image", "r_acc", "r_gyro")])
        )
        result["improves_all_three"] = bool(
            result["r_image"] < 1 and result["r_acc"] < 1 and result["r_gyro"] < 1
        )
        if not self.dist.is_main:
            return result
        print(
            f"  [val step {self.step}] psnr {result.get('image_psnr', float('nan')):.3f} "
            f"ssim {result.get('image_ssim', float('nan')):.4f} | "
            f"r_image {result['r_image']:.4f} r_acc {result['r_acc']:.4f} "
            f"r_gyro {result['r_gyro']:.4f} | joint {result['joint_score']:.4f}"
            f" | all<1 {result['improves_all_three']}",
            flush=True,
        )
        self._append_jsonl("validation.jsonl", result)
        self._update_best(result)
        return result

    def _update_best(self, result: dict) -> None:
        checks = (
            ("best_image", "r_image", True),
            ("best_imu", None, True),
            ("best_joint", "joint_score", result["improves_all_three"]),
        )
        imu_score = 0.5 * (result["r_acc"] + result["r_gyro"])
        for name, key, allowed in checks:
            value = imu_score if key is None else result[key]
            if not allowed:
                continue
            if name not in self.best or value < self.best[name]:
                self.best[name] = value
                self.save_checkpoint(f"{name}.pt", extra={"validation": result})

    # -- checkpoint -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "kind": "training",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "jepa_steps": self.jepa_steps,
            "sens_steps": self.sens_steps,
            "epoch": self.epoch,
            "stage": self.stage,
            # Khong lưu sampler/worker cursor, nen resume chi CHINH XAC khi
            # checkpoint nam o epoch boundary (spec muc 20.1, G5).
            "resume_exact": self.steps_in_epoch == 0,
            "steps_into_epoch": self.steps_in_epoch,
            "data_replay_point": (
                f"dau epoch {self.epoch}" if self.steps_in_epoch == 0
                else f"dau epoch {self.epoch} (phat lai {self.steps_in_epoch} step da chay)"
            ),
            "teacher_initialized": bool(self.model.teacher_initialized),
            "best": self.best,
            "config": self.cfg.to_dict(),
            "config_hash": config_hash(self.cfg),
            "manifest_hash": self.manifest_hash,
            "normalization": {
                "mu": self.model.normalizer.mu.tolist(),
                "scale": self.model.normalizer.scale.tolist(),
            },
            "image_transform": self.cfg.transform.image,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }

    def save_checkpoint(self, name: str, extra: dict | None = None) -> Path:
        """Chi rank 0 ghi file. Rank khac tra ve duong dan ma khong ghi, de hai
        process khong cung viet `last.tmp` roi dam nhau khi rename."""
        path = self.out_dir / name
        if not self.dist.is_main:
            return path
        payload = self.state_dict()
        if extra:
            payload.update(extra)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)   # atomic: khong de lai file hong khi bi ngat
        return path

    def load_checkpoint(self, path: str | Path, *, strict_config: bool = True) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if strict_config:
            if payload.get("manifest_hash") and payload["manifest_hash"] != self.manifest_hash:
                raise ValueError(
                    "manifest_hash khac: checkpoint duoc train tren manifest khac. "
                    "Dung --init de bat dau thi nghiem moi thay vi --resume."
                )
            if payload.get("image_transform") != self.cfg.transform.image:
                raise ValueError(
                    f"checkpoint dung transform {payload.get('image_transform')!r}, "
                    f"config dung {self.cfg.transform.image!r}; khong ghep metric hai backend"
                )
        if payload.get("teacher_initialized"):
            self.model.initialize_teacher()
            self.model.to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step = payload["step"]
        self.jepa_steps = payload.get("jepa_steps", 0)
        self.sens_steps = payload.get("sens_steps", 0)
        self.epoch = payload["epoch"]
        self.steps_in_epoch = 0
        self.best = payload.get("best", {})
        if not payload.get("resume_exact", True):
            print(
                f"  canh bao: checkpoint luu GIUA epoch "
                f"({payload.get('steps_into_epoch')} step vao epoch {payload['epoch']}). "
                f"Resume se phat lai tu {payload.get('data_replay_point')}; "
                "khong phai resume chinh xac tung bit."
            )
        rng = payload.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        return payload

    def export_inference(self, name: str = "inference.pt") -> Path:
        """Bo teacher va predictor (spec muc 20.2)."""
        payload = {
            "kind": "inference",
            "model": self.model.export_state(),
            "config": self.cfg.to_dict(),
            "manifest_hash": self.manifest_hash,
            "image_transform": self.cfg.transform.image,
            "normalization": {
                "mu": self.model.normalizer.mu.tolist(),
                "scale": self.model.normalizer.scale.tolist(),
            },
        }
        path = self.out_dir / name
        torch.save(payload, path)
        return path

    def _append_jsonl(self, name: str, row: dict) -> None:
        with (self.out_dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
