"""Da GPU: effective batch, gating rank, va dong bo gradient that (2 process gloo).

Test dong bo chay THAT bang torchrun 2 process tren CPU. No khong kiem duoc NCCL
tren 2 GPU roi — phan do ghi ro la chua kiem trong README.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from qjepa.training.distributed import DistInfo, dist_info, resolve_accumulation, resolve_device


def test_single_process_by_default():
    info = dist_info()
    assert not info.enabled and info.is_main and info.world_size == 1


def test_effective_batch_preserved_when_divisible():
    """accum chia het world_size -> effective batch khong doi."""
    for accum, world in ((2, 2), (4, 2), (4, 4), (8, 2)):
        new, note = resolve_accumulation(accum, world)
        assert accum * 1 == new * world, "effective batch phai giu nguyen"
        assert "giu effective batch" in note


def test_warns_when_effective_batch_would_change():
    new, note = resolve_accumulation(1, 2)
    assert new == 1 and note.startswith("CANH BAO")
    assert "KHONG so sanh duoc" in note


def test_no_change_for_single_gpu():
    assert resolve_accumulation(2, 1) == (2, "")


def test_device_falls_back_to_cpu_when_processes_exceed_gpus():
    """Nhieu process hon so GPU -> CPU, thay vi hai rank dung chung cuda:0."""
    n = torch.cuda.device_count()
    crowded = DistInfo(rank=1, world_size=n + 1, local_rank=1)
    assert resolve_device(crowded, "auto").type == "cpu"
    # Yeu cau device cu the thi luon duoc ton trong.
    assert resolve_device(DistInfo(), "cpu").type == "cpu"


def test_each_rank_gets_its_own_gpu_when_enough():
    """Du GPU thi rank i dung cuda:i."""
    n = torch.cuda.device_count()
    if n < 2:
        pytest.skip(f"can >=2 GPU, may nay co {n}")
    for r in range(2):
        d = resolve_device(DistInfo(rank=r, world_size=2, local_rank=r), "auto")
        assert d.type == "cuda" and d.index == r


def test_shipped_kaggle_configs_keep_effective_batch_on_two_gpus():
    """Config Kaggle phai chia het cho 2 de 1 GPU va 2 GPU cung effective batch."""
    from qjepa.config import load_config

    for path in sorted(Path("configs").glob("kaggle_*.yaml")):
        cfg = load_config(path)
        accum = cfg.train.gradient_accumulation
        assert accum % 2 == 0, (
            f"{path.name}: gradient_accumulation={accum} khong chia het 2, "
            "chay 2 GPU se lam effective batch gap doi"
        )


@pytest.mark.skipif(
    not Path("outputs/manifest_local/meta.json").exists(),
    reason="can manifest tai outputs/manifest_local",
)
@pytest.mark.skipif(
    os.environ.get("QJEPA_SKIP_DDP") == "1", reason="QJEPA_SKIP_DDP=1"
)
def test_two_process_ddp_keeps_ranks_identical(tmp_path):
    """Chay THAT 2 process: sau train, moi rank phai co tham so GIONG HET nhau.

    Neu ai do goi `self.model(...)` thay vi wrapper DDP thi gradient khong duoc
    all-reduce va test nay fail.
    """
    script = tmp_path / "ddp_job.py"
    script.write_text(
        "import sys, hashlib, torch\n"
        f"sys.path.insert(0, {str(Path.cwd())!r})\n"
        "from dataclasses import replace\n"
        "from qjepa.config import load_config\n"
        "from qjepa.data.manifest import read_manifest\n"
        "from qjepa.data.normalize import ImuNormalizer\n"
        "from qjepa.training import distributed as dstr\n"
        "from qjepa.training.trainer import Trainer, set_seed\n"
        "info = dstr.setup()\n"
        "built = read_manifest('outputs/manifest_local')\n"
        "n = built['meta']['normalization']\n"
        "cfg = load_config('configs/main_qwt_balanced.yaml')\n"
        f"cfg = replace(cfg, output_dir={str(tmp_path / 'run')!r}, train=replace(cfg.train,\n"
        "    stage_a_optimizer_steps=2, stage_b_optimizer_steps=8, max_optimizer_steps=10,\n"
        "    validation_every_updates=0, checkpoint_every_updates=0,\n"
        "    batch_size=2, gradient_accumulation=2, num_workers=0))\n"
        "set_seed(cfg.seed)\n"
        "norm = ImuNormalizer(mean=torch.tensor(n['mean']).float(),"
        " std=torch.tensor(n['std']).float())\n"
        "t = Trainer(cfg, {'train': built['samples']['train'][:32], 'valid': [], 'test': []},\n"
        "            norm, device=dstr.resolve_device(info, 'cpu'),\n"
        "            manifest_hash=built['meta']['manifest_hash'], dist=info)\n"
        "t.train(max_steps=6, log_every=10**6)\n"
        "def h(sd):\n"
        "    x = hashlib.sha256()\n"
        "    for k, v in sorted(sd.items()): x.update(v.detach().cpu().numpy().tobytes())\n"
        "    return x.hexdigest()\n"
        "print('HASH', info.rank, h(t.model.state_dict()), t.accum, t.stage, flush=True)\n"
        "dstr.cleanup()\n"
    )
    # torch da import trong process pytest -> phai ep MKL_THREADING_LAYER=GNU,
    # neu khong process con chet voi loi xung dot libgomp.
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=2", "--standalone", str(script)],
        capture_output=True, text=True, timeout=900, cwd=Path.cwd(), env=env,
    )
    lines = [l.split() for l in proc.stdout.splitlines() if l.startswith("HASH")]
    assert len(lines) == 2, f"can 2 rank, nhan {len(lines)}\n{proc.stdout}\n{proc.stderr[-2000:]}"
    hashes = {l[1]: l[2] for l in lines}
    assert hashes["0"] == hashes["1"], "hai rank lech nhau -> gradient KHONG duoc all-reduce"
    assert all(l[3] == "1" for l in lines), "accum phai la 2//2 = 1"
    assert all(l[4] == "B" for l in lines), "phai vao Stage B (kiem teacher duoi DDP)"
