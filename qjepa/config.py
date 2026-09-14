"""Schema config strict (spec muc 16): reject moi field khong duoc khai bao.

Config la hop dong. Mot field go sai ten phai bao loi ngay thay vi bi bo qua
roi chay bang gia tri mac dinh khac y dinh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any, get_type_hints

import yaml

SCHEMA_VERSION = 3


def _build(cls, payload: Any, path: str = ""):
    """Dung dataclass tu dict, reject unknown field va sai kieu co ban."""
    if not is_dataclass(cls):
        return payload
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"{path or 'config'}: can mapping, nhan {type(payload).__name__}")
    # `from __future__ import annotations` bien annotation thanh STRING, nen
    # phai resolve lai; neu khong, nested dataclass se lot qua ma khong validate.
    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(
            f"{path or 'config'}: field khong duoc ho tro {sorted(unknown)}; "
            f"hop le: {sorted(known)}"
        )
    kwargs = {}
    for name, f in known.items():
        if name not in payload:
            continue
        value = payload[name]
        sub = f"{path}.{name}" if path else name
        annotation = hints.get(name, f.type)
        if isinstance(annotation, type) and is_dataclass(annotation):
            kwargs[name] = _build(annotation, value, sub)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "tartanair"
    root: str | None = None
    manifest: str | None = None
    camera_id: str = "lcam_front"
    image_size: tuple[int, int] = (256, 256)
    imu_window_samples: int = 128
    imu_rate_hz: float | None = None
    image_rate_hz: float | None = None
    window_mode: str = "centered_offline"
    max_relative_dt_deviation: float = 0.01
    max_center_error_in_imu_dt: float = 1.0
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    train_shuffle_windows: bool = True
    imu_std_floor: tuple[float, ...] = (1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4)
    allow_imu_padding: bool = False
    allow_automatic_window_change: bool = False


@dataclass(frozen=True)
class TransformConfig:
    image: str = "qwt"
    image_levels: int = 1
    imu: str = "dwt_haar_1d"
    imu_levels: int = 1
    compute_dtype: str = "float32"
    allow_silent_fallback: bool = False


@dataclass(frozen=True)
class ModelConfig:
    encoder_channels: tuple[int, ...] = (32, 64, 96, 128)
    embedding_dim: int = 128
    groupnorm_groups: int = 8
    imu_summary_bins: int = 4
    fusion: str = "gated_mlp"
    fusion_hidden_dim: int = 256
    cross_modal: bool = True
    gate_bias_init: float = -2.0
    predictor_hidden_dim: int = 256
    dropout: float = 0.0
    zero_init_coefficient_heads: bool = True
    mask_ratio: float = 0.0
    time_metadata_dim: int = 3


@dataclass(frozen=True)
class ImageCorruption:
    blur_sigma_px: tuple[float, float] = (0.3, 1.0)
    gaussian_std: tuple[float, float] = (2 / 255, 8 / 255)
    downsample_probability: float = 0.0
    downsample_scale: tuple[float, float] = (0.5, 1.0)
    jpeg_probability: float = 0.0


@dataclass(frozen=True)
class ImuCorruption:
    accel_noise_std: tuple[float, float] = (0.02, 0.10)
    gyro_noise_std: tuple[float, float] = (0.001, 0.005)
    bias_enabled: bool = False
    accel_bias_bound: float = 0.05
    gyro_bias_bound: float = 0.003
    drift_enabled: bool = False
    accel_bias_random_walk: float = 0.005
    gyro_bias_random_walk: float = 0.0002
    spike_enabled: bool = False
    spike_rate_per_second_per_triplet: float = 0.2
    accel_spike_amplitude: tuple[float, float] = (0.5, 2.0)
    gyro_spike_amplitude: tuple[float, float] = (0.02, 0.10)


@dataclass(frozen=True)
class CorruptionConfig:
    profile: str = "mild"
    independent_modalities: bool = True
    clean_probability_each_modality: float = 0.1
    image_parameter_scope: str = "one_second_camera_segment"
    imu_parameter_scope: str = "trajectory_realization"
    validation_realization: int = 0
    image: ImageCorruption = field(default_factory=ImageCorruption)
    imu: ImuCorruption = field(default_factory=ImuCorruption)


@dataclass(frozen=True)
class LossConfig:
    smooth_l1_beta: float = 1.0
    image_weight: float = 1.0
    imu_weight: float = 1.0
    jepa_start_weight: float = 0.01
    jepa_max_weight: float = 0.10
    jepa_ramp_updates: int = 1000
    jepa_target_norm_eps: float = 1e-5
    edge_weight: float = 0.0
    imu_delta_weight: float = 0.0
    variance_weight: float = 0.0
    variance_gamma: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    no_decay_bias_and_norm: bool = True
    batch_size: int = 4
    gradient_accumulation: int = 2
    grad_clip_norm: float = 1.0
    precision: str = "fp32"
    num_workers: int = 0
    max_optimizer_steps: int = 10000
    stage_a_optimizer_steps: int = 2000
    stage_b_optimizer_steps: int = 8000
    scheduler: str = "warmup_cosine"
    warmup_fraction: float = 0.05
    minimum_lr: float = 1e-6
    teacher_momentum_start: float = 0.99
    teacher_momentum_end: float = 0.999
    initialize_teacher_at_stage_b: bool = True
    validation_every_updates: int = 500
    checkpoint_every_updates: int = 500
    validation_batches: int = 32


@dataclass(frozen=True)
class EvaluationConfig:
    image_data_range: float = 1.0
    image_border_crop: int = 0
    image_metric_clamp: bool = True
    imu_merge: str = "normalized_triangular_overlap_add"
    trajectory_equal_weight: bool = True
    minimum_fixed_feature_bank: int = 64
    test_realizations: tuple[int, ...] = (0, 1, 2)
    clean_image_mae_tolerance: float = 1 / 255
    clean_accel_rmse_tolerance: float = 0.02
    clean_gyro_rmse_tolerance: float = 0.001


@dataclass(frozen=True)
class EncoderSensitivityCfg:
    """themjacobian muc 14. Tap tai FI/FU dense TRUOC fusion."""

    enabled: bool = False
    method: str = "finite_difference"
    target: str = "online_dense_before_fusion"
    measurement_normalization: str = "channel_layer_norm_no_affine"
    layer_norm_eps: float = 1e-5
    feature_to_fusion: str = "raw_unchanged"
    image_epsilon: float = 1.0 / 255.0
    imu_normalized_epsilon: float = 0.01
    alpha: float = 1.0
    direction: str = "rademacher"
    image_boundary: str = "clamp_and_measure_actual_delta"
    input_energy: str = "mean_actual_delta_squared_core_units"
    minimum_input_energy: float = 1e-12
    detach_base_feature: bool = False
    source_schedule: str = "alternate_successful_optimizer_update"
    modality_multipliers: tuple[float, float] = (1.0, 1.0)
    weight_max: float = 1e-4
    ramp_updates: int = 200
    probe_seed: int = 73129
    precision: str = "fp32"
    output_sensitivity_loss_weight: float = 0.0
    fusion_sensitivity_loss_weight: float = 0.0


@dataclass(frozen=True)
class RepresentationPhaseCfg:
    name: str = "encoder_sensitivity_finetune_v2"
    parent_checkpoint: str | None = None
    parent_stage_required: str = "stable_stage_b"
    init_mode: str = "weights_teachers_normalizer_from_parent"
    reset_optimizer_on_new_phase: bool = True
    learning_rate: float = 2e-5
    warmup_updates: int = 100
    minimum_lr: float = 1e-6
    max_successful_updates: int = 1000
    preserve_parent_jepa_weight: bool = True
    validation_every_updates: int = 100
    backbone_checkpoint_for_probe: str = "last_fixed_budget"


@dataclass(frozen=True)
class LatentMonitorCfg:
    enabled: bool = True
    reference_split: str = "train"
    target_bank_samples: int = 64
    log_raw_and_normalized: bool = True
    collapse_relative_threshold: float = 0.1
    collapse_consecutive_evaluations: int = 3
    raw_rms_ratio_warning: tuple[float, float] = (0.1, 10.0)


@dataclass(frozen=True)
class RepresentationProbeCfg:
    enabled: bool = False
    input: str = "fused_dense_only"
    freeze_encoders: bool = True
    freeze_fusion: bool = True
    encoder_skips: bool = False
    input_coefficient_residual: bool = False
    teacher_features: bool = False
    predictor_features: bool = False
    coefficients: str = "absolute_prediction"
    initialization_seed: int = 73131
    feature_channels: int = 128
    decoder_channels: tuple[int, ...] = (96, 64, 32)
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_updates: int = 200
    minimum_lr: float = 1e-6
    batch_size: int = 4
    gradient_accumulation: int = 2
    precision: str = "fp32"
    max_successful_updates: int = 2000
    validation_every_updates: int = 200


@dataclass(frozen=True)
class SensitivityDiagnosticsCfg:
    enabled: bool = False
    train_loss: bool = False
    measure_encoder_features: bool = True
    measure_full_restoration_outputs: bool = True
    output_cross_gains: bool = True
    seed: int = 73130
    target_bank_samples: int = 64
    directions_per_source: int = 4
    alpha_sweep: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class HamiltonAuditCfg:
    required: bool = True
    learned_hamilton_layers: bool = False
    replace_learned_layers_in_this_migration: bool = False


@dataclass(frozen=True)
class Config:
    schema_version: int = SCHEMA_VERSION
    experiment_name: str = "qwt_jepa_rgb256_imu128x6_gated_mlp"
    seed: int = 42
    output_dir: str = "outputs/run"
    data: DataConfig = field(default_factory=DataConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # --- themjacobian v2 (mac dinh TAT; config cu khong co section nay van chay) ---
    migration_schema_version: int = 1
    experiment_suffix: str = ""
    encoder_sensitivity: EncoderSensitivityCfg = field(default_factory=EncoderSensitivityCfg)
    representation_phase: RepresentationPhaseCfg = field(default_factory=RepresentationPhaseCfg)
    latent_monitor: LatentMonitorCfg = field(default_factory=LatentMonitorCfg)
    representation_probe: RepresentationProbeCfg = field(default_factory=RepresentationProbeCfg)
    sensitivity_diagnostics: SensitivityDiagnosticsCfg = field(default_factory=SensitivityDiagnosticsCfg)
    hamilton_audit: HamiltonAuditCfg = field(default_factory=HamiltonAuditCfg)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {self.schema_version}, can {SCHEMA_VERSION}")
        if self.data.imu_window_samples != 128:
            raise ValueError(
                f"imu_window_samples={self.data.imu_window_samples}: cau hinh chinh la 128 "
                "va co dinh theo yeu cau (spec muc 0.8). L khac chi duoc chay nhu run "
                "nghien cuu rieng co cau hinh/report doc lap."
            )
        if self.data.imu_window_samples % 16:
            raise ValueError("imu_window_samples phai chia het 16 (ba lan downsample + Haar)")
        if any(s % 2 for s in self.data.image_size):
            raise ValueError("image_size phai chan cho wavelet mot level")
        if self.model.encoder_channels[-1] != self.model.embedding_dim:
            raise ValueError("encoder_channels[-1] phai bang embedding_dim")
        if any(c % self.model.groupnorm_groups for c in self.model.encoder_channels):
            raise ValueError("moi encoder channel phai chia het groupnorm_groups")
        if self.transform.image not in ("qwt", "dwt_haar_baseline"):
            raise ValueError(f"transform.image={self.transform.image!r} khong hop le")
        if self.transform.allow_silent_fallback:
            raise ValueError("allow_silent_fallback phai la false (spec muc 6.3)")
        if self.data.allow_imu_padding or self.data.allow_automatic_window_change:
            raise ValueError("khong duoc bat padding/doi window tu dong (spec muc 3.3)")
        total = self.train.stage_a_optimizer_steps + self.train.stage_b_optimizer_steps
        if total != self.train.max_optimizer_steps:
            raise ValueError(
                f"stage_a + stage_b = {total} != max_optimizer_steps "
                f"{self.train.max_optimizer_steps}"
            )
        if self.train.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"precision={self.train.precision!r} khong hop le")
        self._validate_migration_v2()

    def _validate_migration_v2(self) -> None:
        """Rang buoc cua themjacobian muc 14."""
        es, rp = self.encoder_sensitivity, self.representation_probe
        if self.migration_schema_version not in (1, 2):
            raise ValueError(f"migration_schema_version={self.migration_schema_version}, can 1 hoac 2")
        if es.enabled and self.migration_schema_version != 2:
            raise ValueError(
                "encoder_sensitivity.enabled=true can migration_schema_version=2; "
                "khong dien giai lai config v1 thanh v2 am tham"
            )
        if es.enabled:
            # Main v2 chi cho tap TRUOC fusion, do bang LN khong affine.
            if es.target != "online_dense_before_fusion":
                raise ValueError(
                    f"encoder_sensitivity.target={es.target!r}: main v2 chi ho tro "
                    "'online_dense_before_fusion'. Target o output/predictor la cau hinh v1."
                )
            if es.method != "finite_difference":
                raise ValueError(f"method={es.method!r} chua ho tro")
            if es.measurement_normalization != "channel_layer_norm_no_affine":
                raise ValueError(
                    "measurement_normalization phai la 'channel_layer_norm_no_affine': "
                    "LayerNorm co affine cho phep mang thu nho scale de giam loss"
                )
            if es.detach_base_feature:
                raise ValueError("detach_base_feature phai false: gradient can di qua ca hai nhanh")
            if es.feature_to_fusion != "raw_unchanged":
                raise ValueError("feature_to_fusion phai la 'raw_unchanged': LN chi de DO")
            if es.output_sensitivity_loss_weight or es.fusion_sensitivity_loss_weight:
                raise ValueError(
                    "main v2 khong cho output/fusion sensitivity loss khac 0 "
                    "(do la namespace v1). Chay combined ablation bang run type rieng."
                )
            if es.direction != "rademacher":
                raise ValueError(f"direction={es.direction!r} chua ho tro")
            if es.precision != "fp32":
                raise ValueError("encoder_sensitivity.precision phai fp32 o ban main")
            if es.image_epsilon <= 0 or es.imu_normalized_epsilon <= 0 or es.alpha <= 0:
                raise ValueError("epsilon va alpha phai duong")
            if es.weight_max < 0 or es.ramp_updates < 1:
                raise ValueError("weight_max >= 0 va ramp_updates >= 1")
        if rp.enabled:
            if rp.input != "fused_dense_only":
                raise ValueError(f"representation_probe.input={rp.input!r}: chi nhan ZI/ZU")
            if rp.encoder_skips or rp.input_coefficient_residual:
                raise ValueError(
                    "probe decoder bi cam encoder_skips va input_coefficient_residual — "
                    "do la duong ro ri thong tin. Muon finetune decoder chinh thi goi phase khac."
                )
            if rp.teacher_features or rp.predictor_features:
                raise ValueError("probe decoder khong duoc nhan feature teacher/predictor")
            if not (rp.freeze_encoders and rp.freeze_fusion):
                raise ValueError("probe decoder yeu cau backbone DONG BANG hoan toan")
            if rp.coefficients != "absolute_prediction":
                raise ValueError("probe decoder du doan he so TUYET DOI, khong residual")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if overrides:
        payload = _deep_merge(payload, overrides)
    cfg = _build(Config, payload)
    cfg.validate()
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_resolved(cfg: Config, path: str | Path) -> None:
    """Ghi resolved_config.yaml - moi mac dinh da duoc resolve (spec muc 0.10)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")


def config_hash(cfg: Config) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
