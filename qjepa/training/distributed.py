"""Ho tro da GPU bang DistributedDataParallel (themjacobian muc 10).

Quy uoc:
  - 1 GPU (hoac khong co GPU): chay thang, khong khoi tao process group.
  - N GPU: `torchrun --nproc_per_node=N`, moi process giu mot GPU.

`batch_size` trong config la batch **moi rank**. De giu **effective batch** khong
doi giua 1 va N GPU, `gradient_accumulation` duoc chia cho world_size khi chia
het; neu khong chia het thi giu nguyen va **canh bao**, vi effective batch doi
se lam hai run khong con so sanh duoc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def dist_info() -> DistInfo:
    """Doc bien moi truong do torchrun dat. Khong co thi coi nhu single process."""
    if "WORLD_SIZE" not in os.environ:
        return DistInfo()
    return DistInfo(
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )


def setup(info: DistInfo | None = None) -> DistInfo:
    """Khoi tao process group neu dang chay da process."""
    info = info or dist_info()
    if not info.enabled or dist.is_initialized():
        return info
    # NCCL chi dung duoc khi MOI rank co GPU rieng. Neu so process nhieu hon so
    # GPU (vi du test 2 process tren may 1 GPU) thi phai roi ve gloo/CPU.
    n_gpu = torch.cuda.device_count()
    use_cuda = n_gpu > 0 and info.world_size <= n_gpu
    backend = "nccl" if use_cuda and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend)
    if use_cuda:
        torch.cuda.set_device(info.local_rank)
    return info


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def resolve_device(info: DistInfo, requested: str = "auto") -> torch.device:
    """Chon device, nhat quan voi lua chon backend trong `setup()`.

    Neu so process NHIEU HON so GPU thi khong rank nao dung CUDA: hai process
    dung chung mot GPU vua khong dung y dinh vua lam NCCL that bai.
    """
    if requested not in ("auto", "cuda"):
        return torch.device(requested)
    n_gpu = torch.cuda.device_count()
    if info.world_size > n_gpu:
        return torch.device("cpu")
    if info.enabled:
        return torch.device(f"cuda:{info.local_rank}")
    return torch.device("cuda" if n_gpu > 0 else "cpu")


def resolve_accumulation(accum: int, world_size: int) -> tuple[int, str]:
    """Giu effective batch khong doi khi tang so GPU.

    -> (accumulation moi, ghi chu). Chia het thi chia; khong thi giu va canh bao.
    """
    if world_size <= 1:
        return accum, ""
    if accum % world_size == 0:
        return accum // world_size, (
            f"gradient_accumulation {accum} -> {accum // world_size} "
            f"de giu effective batch khong doi tren {world_size} GPU"
        )
    return accum, (
        f"CANH BAO: gradient_accumulation={accum} khong chia het world_size={world_size}. "
        f"Effective batch tang {world_size} lan so voi run 1 GPU — hai run se KHONG "
        f"so sanh duoc truc tiep."
    )


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Trung binh mot scalar qua moi rank (dung cho log/metric)."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
