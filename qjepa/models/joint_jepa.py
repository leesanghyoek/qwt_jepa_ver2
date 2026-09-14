"""Model chung camera-IMU: phuc hoi joint + muc tieu JEPA (spec muc 7, 12, 15).

Mot model, mot checkpoint inference. Teacher va predictor chi ton tai khi
train va bi bo khi export.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from ..data.normalize import ImuNormalizer
from ..transforms import build_image_transform, build_imu_transform
from .decoders import LatentPredictor, image_decoder, imu_decoder
from .encoders import image_encoder, imu_encoder
from .fusion import SharedGatedFusion, build_time_metadata


def image_tokens(f: torch.Tensor) -> torch.Tensor:
    """[B,D,H,W] -> [B,H*W,D], flatten row-major theo (H, W)."""
    return f.flatten(2).transpose(1, 2)


def imu_tokens(f: torch.Tensor) -> torch.Tensor:
    """[B,D,N] -> [B,N,D]."""
    return f.transpose(1, 2)


class JointRestorationJEPA(nn.Module):
    """Mot frame RGB 256x256 + mot cua so IMU 128x6.

    Args:
        image_transform: 'qwt' hoac 'dwt_haar_baseline'.
        imu_window: so hang IMU L (cau hinh chinh 128).
        cross_modal: bat/tat trao doi thong tin giua hai nhanh.
    """

    def __init__(
        self,
        *,
        image_transform: str = "qwt",
        imu_transform: str = "dwt_haar_1d",
        encoder_channels: tuple[int, ...] = (32, 64, 96, 128),
        embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
        predictor_hidden_dim: int = 256,
        imu_summary_bins: int = 4,
        time_metadata_dim: int = 3,
        cross_modal: bool = True,
        gate_bias_init: float = -2.0,
        groupnorm_groups: int = 8,
        imu_channels: int = 6,
        imu_window: int = 128,
        image_size: tuple[int, int] = (256, 256),
        normalizer: ImuNormalizer | None = None,
    ) -> None:
        super().__init__()
        if encoder_channels[-1] != embedding_dim:
            raise ValueError(
                f"encoder_channels[-1]={encoder_channels[-1]} phai bang "
                f"embedding_dim={embedding_dim}"
            )
        self.image_transform_name = image_transform
        self.imu_window = imu_window
        self.image_size = tuple(image_size)
        self.imu_channels = imu_channels
        self.cross_modal = cross_modal

        self.image_transform = build_image_transform(image_transform)
        self.imu_transform = build_imu_transform(imu_transform, channels=imu_channels)
        self.normalizer = normalizer if normalizer is not None else ImuNormalizer(channels=imu_channels)

        ci = self.image_transform.coeff_channels
        cu = self.imu_transform.coeff_channels
        self.image_encoder = image_encoder(ci, encoder_channels, groupnorm_groups)
        self.imu_encoder = imu_encoder(cu, encoder_channels, groupnorm_groups)
        self.shared_fusion = SharedGatedFusion(
            embedding_dim,
            fusion_hidden_dim,
            imu_summary_bins=imu_summary_bins,
            time_metadata_dim=time_metadata_dim,
            cross_modal=cross_modal,
            gate_bias_init=gate_bias_init,
        )
        self.image_decoder = image_decoder(ci, encoder_channels, groupnorm_groups)
        self.imu_decoder = imu_decoder(cu, encoder_channels, groupnorm_groups)
        self.image_predictor = LatentPredictor(embedding_dim, predictor_hidden_dim)
        self.imu_predictor = LatentPredictor(embedding_dim, predictor_hidden_dim)

        self.teacher_image_encoder: nn.Module | None = None
        self.teacher_imu_encoder: nn.Module | None = None
        self.register_buffer("teacher_initialized", torch.tensor(False))

    # -- online -----------------------------------------------------------
    def forward(
        self,
        image_bad: torch.Tensor,
        imu_bad_phys: torch.Tensor,
        image_time: torch.Tensor,
        imu_times: torch.Tensor,
        *,
        predict_latents: bool = False,
        probe_image: torch.Tensor | None = None,
        probe_imu_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Duong online; chi nhan du lieu NHIEU (spec muc 15).

        `predict_latents` va `probe_*` gop cac duong forward PHU vao CUNG mot lan
        goi forward. Voi DDP dieu nay la bat buoc: moi tham so can gradient phai
        duoc dung BEN TRONG forward cua module duoc boc, neu khong reducer bao
        "mark a variable ready only once" (themjacobian muc 10).
        """
        b = image_bad.shape[0]
        if image_bad.dim() != 4 or image_bad.shape[1] != 3:
            raise ValueError(f"image_bad can [B,3,H,W], nhan {tuple(image_bad.shape)}")
        if tuple(image_bad.shape[2:]) != self.image_size:
            raise ValueError(
                f"image_bad {tuple(image_bad.shape[2:])} khac image_size {self.image_size}"
            )
        if imu_bad_phys.shape[1:] != (self.imu_channels, self.imu_window):
            raise ValueError(
                f"imu_bad_phys can [B,{self.imu_channels},{self.imu_window}], "
                f"nhan {tuple(imu_bad_phys.shape)}"
            )
        if imu_bad_phys.shape[0] != b or image_time.shape != (b,) or imu_times.shape != (b, self.imu_window):
            raise ValueError("batch size hoac timestamp shape khong khop")

        imu_bad_norm = self.normalizer.normalize(imu_bad_phys)
        ci, image_layout = self.image_transform.analysis(image_bad)
        cu, imu_layout = self.imu_transform.analysis(imu_bad_norm)

        fi, image_skips = self.image_encoder(ci)
        fu, imu_skips = self.imu_encoder(cu)
        meta = build_time_metadata(image_time, imu_times)
        zi, zu = self.shared_fusion(fi, fu, meta)

        delta_ci = self.image_decoder(zi, image_skips)
        delta_cu = self.imu_decoder(zu, imu_skips)
        image_hat = self.image_transform.synthesis(ci + delta_ci, image_layout)
        imu_hat_norm = self.imu_transform.synthesis(cu + delta_cu, imu_layout)

        out = {
            "image_hat": image_hat,
            "imu_hat_norm": imu_hat_norm,
            "imu_hat_phys": self.normalizer.denormalize(imu_hat_norm),
            "fi": fi,
            "fu": fu,
            "zi": zi,
            "zu": zu,
        }
        if predict_latents:
            out["pred_image"] = self.image_predictor(image_tokens(zi))
            out["pred_imu"] = self.imu_predictor(imu_tokens(zu))
        if probe_image is not None:
            out["probe_fi"] = self.encode_image_dense(probe_image)
        if probe_imu_norm is not None:
            out["probe_fu"] = self.encode_imu_dense_normalized(probe_imu_norm)
        return out

    # -- API tach cho do do nhay (themjacobian muc 5) ----------------------
    def encode_image_dense(self, image: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> FI [B,D,Hi,Wi], dense truoc fusion.

        Dung DUNG instance transform va encoder nhu duong base — khong deepcopy,
        khong detach. Bo cac gia tri tra ve khong dung (skips/coefficients).
        """
        ci, _ = self.image_transform.analysis(image)
        fi, _ = self.image_encoder(ci)
        return fi

    def encode_imu_dense_normalized(self, imu_norm: torch.Tensor) -> torch.Tensor:
        """[B,6,L] da normalize -> FU [B,D,Lu], dense truoc fusion."""
        cu, _ = self.imu_transform.analysis(imu_norm)
        fu, _ = self.imu_encoder(cu)
        return fu

    def encode_and_fuse(
        self,
        image_bad: torch.Tensor,
        imu_bad_phys: torch.Tensor,
        image_time: torch.Tensor,
        imu_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (ZI, ZU) sau fusion. Dung cho probe decoder tren backbone dong bang."""
        imu_norm = self.normalizer.normalize(imu_bad_phys)
        ci, _ = self.image_transform.analysis(image_bad)
        cu, _ = self.imu_transform.analysis(imu_norm)
        fi, _ = self.image_encoder(ci)
        fu, _ = self.imu_encoder(cu)
        meta = build_time_metadata(image_time, imu_times)
        return self.shared_fusion(fi, fu, meta)

    def freeze_backbone(self) -> None:
        """Dong bang encoder/fusion/normalizer/transform cho phase probe (muc 13)."""
        for module in (
            self.image_encoder, self.imu_encoder, self.shared_fusion,
            self.image_transform, self.imu_transform, self.normalizer,
        ):
            module.requires_grad_(False)
            module.eval()

    # -- teacher ----------------------------------------------------------
    def initialize_teacher(self) -> None:
        """Deepcopy encoder online lam teacher (goi dau Stage B, spec muc 12.2)."""
        self.teacher_image_encoder = copy.deepcopy(self.image_encoder)
        self.teacher_imu_encoder = copy.deepcopy(self.imu_encoder)
        for teacher in (self.teacher_image_encoder, self.teacher_imu_encoder):
            teacher.requires_grad_(False)
            teacher.eval()
        self.teacher_initialized = torch.tensor(True, device=self.teacher_initialized.device)

    def train(self, mode: bool = True):
        """Giu teacher o eval ngay ca khi parent chuyen sang train (muc 12.2)."""
        super().train(mode)
        for teacher in (self.teacher_image_encoder, self.teacher_imu_encoder):
            if teacher is not None:
                teacher.eval()
        return self

    @torch.no_grad()
    def teacher_targets(
        self, image_clean: torch.Tensor, imu_clean_phys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Target latent tu du lieu SACH; stop-gradient (spec muc 12.2)."""
        if not bool(self.teacher_initialized):
            raise RuntimeError("teacher chua khoi tao; goi initialize_teacher() truoc Stage B")
        ci, _ = self.image_transform.analysis(image_clean)
        cu, _ = self.imu_transform.analysis(self.normalizer.normalize(imu_clean_phys))
        ti, _ = self.teacher_image_encoder(ci)
        tu, _ = self.teacher_imu_encoder(cu)
        return ti, tu

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA sau MOI optimizer step thanh cong (spec muc 12.2)."""
        if not bool(self.teacher_initialized):
            raise RuntimeError("teacher chua khoi tao")
        pairs = (
            (self.teacher_image_encoder, self.image_encoder),
            (self.teacher_imu_encoder, self.imu_encoder),
        )
        for teacher, online in pairs:
            for pt, po in zip(teacher.parameters(), online.parameters()):
                pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)
            for bt, bo in zip(teacher.buffers(), online.buffers()):
                bt.copy_(bo)

    # -- helpers ----------------------------------------------------------
    def online_parameters(self):
        """Tham so duoc optimizer cap nhat; KHONG gom teacher."""
        teachers = set()
        for teacher in (self.teacher_image_encoder, self.teacher_imu_encoder):
            if teacher is not None:
                teachers.update(id(p) for p in teacher.parameters())
        return [p for p in self.parameters() if id(p) not in teachers and p.requires_grad]

    def export_state(self) -> dict:
        """State dict cho inference: bo teacher va predictor (spec muc 20.2)."""
        drop = ("teacher_image_encoder", "teacher_imu_encoder", "image_predictor", "imu_predictor")
        return {
            k: v for k, v in self.state_dict().items()
            if not any(k.startswith(p) for p in drop)
        }


def teacher_momentum(step: int, total_steps: int, start: float = 0.99, end: float = 0.999) -> float:
    """Lich cosine tang dan (spec muc 12.2). step bat dau tu 0 o update JEPA dau."""
    p = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * p))
