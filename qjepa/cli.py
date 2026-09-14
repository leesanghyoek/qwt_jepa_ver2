"""Giao dien dong lenh (spec muc 21).

    python -m qjepa audit-data     --root DATA
    python -m qjepa build-manifest --root DATA --out DIR
    python -m qjepa check-transform
    python -m qjepa smoke-train    --config C --manifest DIR
    python -m qjepa overfit        --config C --manifest DIR --samples 16
    python -m qjepa train          --config C --manifest DIR [--resume CKPT | --init CKPT]
    python -m qjepa evaluate       --config C --manifest DIR --checkpoint CKPT
    python -m qjepa export         --config C --manifest DIR --checkpoint CKPT
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import Config, load_config
from .data.manifest import build_manifest, read_manifest, write_manifest
from .data.normalize import ImuNormalizer
from .data.tartanair import audit_dataset, write_audit
from .training.trainer import Trainer, make_dataset, set_seed


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load(args) -> tuple[Config, dict, ImuNormalizer]:
    cfg = load_config(args.config)
    if getattr(args, "output_dir", None):
        cfg = replace(cfg, output_dir=args.output_dir)
    built = read_manifest(args.manifest)
    norm = built["meta"]["normalization"]
    normalizer = ImuNormalizer(
        mean=torch.tensor(norm["mean"], dtype=torch.float32),
        std=torch.tensor(norm["std"], dtype=torch.float32),
        std_floor=tuple(cfg.data.imu_std_floor),
    )
    return cfg, built, normalizer


def cmd_audit_data(args) -> None:
    report = audit_dataset(args.root, window=args.window)
    out = Path(args.out or "outputs/audit")
    out.mkdir(parents=True, exist_ok=True)
    write_audit(report, out / "audit.json")
    print(json.dumps({k: v for k, v in report.items() if k != "per_trajectory"}, indent=2))
    print(f"\nchi tiet tung trajectory -> {out / 'audit.json'}")


def cmd_build_manifest(args) -> None:
    built = build_manifest(
        args.root,
        window=args.window,
        split_ratio=tuple(args.split_ratio),
        seed=args.seed,
    )
    write_manifest(built, args.out)
    meta = built["meta"]
    print(json.dumps({k: meta[k] for k in ("splits", "manifest_hash", "window")}, indent=2))
    print("normalization mean:", [round(x, 5) for x in meta["normalization"]["mean"]])
    print("normalization std :", [round(x, 5) for x in meta["normalization"]["std"]])
    print(f"-> {args.out}")


def cmd_check_transform(args) -> None:
    """Gate G1: round-trip va gradient cua ca hai transform."""
    from .transforms import build_image_transform, build_imu_transform

    torch.manual_seed(0)
    rows = []
    for name in ("qwt", "dwt_haar_baseline"):
        tr = build_image_transform(name)
        for case, x in (
            ("random", torch.rand(2, 3, 64, 64)),
            ("zeros", torch.zeros(1, 3, 32, 32)),
            ("constant", torch.full((1, 3, 32, 32), 0.5)),
            ("rgb_distinct", torch.stack([torch.full((32, 32), v) for v in (0.2, 0.5, 0.9)])[None]),
        ):
            c, lay = tr.analysis(x)
            r = tr.synthesis(c, lay)
            den = x.norm().item()
            rows.append(
                {
                    "transform": name,
                    "case": case,
                    "coeff_channels": c.shape[1],
                    "relative_error": ((r - x).norm() / den).item() if den else None,
                    "max_abs_error": (r - x).abs().max().item(),
                }
            )
    tr = build_imu_transform("dwt_haar_1d")
    for case, u in (
        ("random", torch.randn(2, 6, 128)),
        ("ramp", torch.linspace(-1, 1, 128).expand(1, 6, 128).contiguous()),
    ):
        c, lay = tr.analysis(u)
        r = tr.synthesis(c, lay)
        rows.append(
            {
                "transform": "dwt_haar_1d",
                "case": case,
                "coeff_channels": c.shape[1],
                "relative_error": ((r - u).norm() / u.norm()).item(),
                "max_abs_error": (r - u).abs().max().item(),
            }
        )
    for row in rows:
        rel = row["relative_error"]
        rel_s = "n/a" if rel is None else f"{rel:.3e}"
        print(
            f"{row['transform']:20} {row['case']:14} ch={row['coeff_channels']:3d} "
            f"rel={rel_s:>10} maxabs={row['max_abs_error']:.3e}"
        )
    worst = max(r["relative_error"] or 0.0 for r in rows)
    worst_abs = max(r["max_abs_error"] for r in rows)
    ok = worst <= 1e-5 and worst_abs <= 1e-5
    print(f"\nG1 round-trip: {'PASS' if ok else 'FAIL'} (worst rel {worst:.3e}, target <= 1e-5)")


def _trainer(args) -> tuple[Trainer, Config]:
    cfg, built, normalizer = _load(args)
    set_seed(cfg.seed)
    trainer = Trainer(
        cfg,
        built["samples"],
        normalizer,
        device=_device(args.device),
        manifest_hash=built["meta"]["manifest_hash"],
    )
    return trainer, cfg


def cmd_smoke_train(args) -> None:
    """Gate G2: vai chuc step, kiem tra shape/gradient/finite."""
    trainer, cfg = _trainer(args)
    print(f"device={trainer.device} transform={cfg.transform.image} params="
          f"{sum(p.numel() for p in trainer.model.online_parameters())/1e6:.2f}M")
    before = [p.detach().clone() for p in trainer.model.online_parameters()]
    trainer.train(max_steps=args.steps, log_every=max(1, args.steps // 5))
    after = trainer.model.online_parameters()
    changed = sum(1 for a, b in zip(before, after) if not torch.equal(a, b))
    print(f"\nG2: {changed}/{len(before)} tensor tham so da doi sau {trainer.step} step")
    print("G2 finite:", all(torch.isfinite(p).all().item() for p in after))


def cmd_overfit(args) -> None:
    """Gate G3: overfit mot nhom sample co dinh, nhieu khoa cung."""
    trainer, cfg = _trainer(args)
    subset = trainer.samples["train"][: args.samples]
    trainer.samples = {"train": subset, "valid": subset, "test": []}
    # Muc 14.2: overfit debug dung LR CO DINH, khong warmup/cosine cua pilot.
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.cfg = replace(trainer.cfg, train=replace(trainer.cfg.train, validation_every_updates=0))
    print(f"overfit {len(subset)} sample co dinh, {args.steps} step, LR co dinh "
          f"{trainer.cfg.train.learning_rate}")
    trainer.train(max_steps=args.steps, log_every=max(1, args.steps // 10))
    result = trainer.validate(split="valid", max_batches=10**6)
    print("\nG3 error reduction so voi input nhieu (can >= 0.30 moi nhom):")
    for key, label in (("r_image", "image"), ("r_acc", "accel"), ("r_gyro", "gyro")):
        print(f"  {label:6}: r={result[key]:.4f}  giam {100*(1-result[key]):+.2f}%")


def cmd_train(args) -> None:
    trainer, cfg = _trainer(args)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resume tu {args.resume} tai step {trainer.step} (stage {trainer.stage})")
    elif args.init:
        payload = torch.load(args.init, map_location=trainer.device, weights_only=False)
        if payload.get("teacher_initialized"):
            trainer.model.initialize_teacher()
            trainer.model.to(trainer.device)
        trainer.model.load_state_dict(payload["model"], strict=True)
        print(f"init trong so tu {args.init}; step bat dau lai tu 0")
    summary = trainer.train(max_steps=args.steps, log_every=args.log_every)
    print("\ntrain xong:", summary)
    trainer.validate()
    print("export:", trainer.export_inference())


def cmd_evaluate(args) -> None:
    trainer, cfg = _trainer(args)
    trainer.load_checkpoint(args.checkpoint, strict_config=not args.allow_mismatch)
    result = trainer.validate(split=args.split, max_batches=args.max_batches or 10**6)
    print(json.dumps(result, indent=2, default=str))


def cmd_export(args) -> None:
    trainer, cfg = _trainer(args)
    trainer.load_checkpoint(args.checkpoint, strict_config=not args.allow_mismatch)
    print("export ->", trainer.export_inference(args.name))


def cmd_infer(args) -> None:
    """Chay tren MOT trajectory that: xuat anh phuc hoi + IMU da ghep (muc 20.2)."""
    import time

    import numpy as np
    from PIL import Image

    from .data.dataset import collate, PairedRestorationDataset
    from .data.manifest import pair_trajectory
    from .data.tartanair import Trajectory, discover_trajectories
    from .evaluation.overlap import TrajectoryMerger

    cfg, built, normalizer = _load(args)
    device = _device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    from .training.trainer import build_model

    model = build_model(cfg, normalizer).to(device)
    state = payload["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    blocked = [k for k in missing if not k.startswith(("image_predictor", "imu_predictor",
                                                       "teacher_image_encoder", "teacher_imu_encoder"))]
    if blocked or unexpected:
        raise ValueError(f"checkpoint khong khop: thieu {blocked}, thua {list(unexpected)}")
    model.eval()

    traj = next(t for t in discover_trajectories(args.root) if t.key == args.trajectory)
    samples, stats = pair_trajectory(traj, cfg.data.imu_window_samples, split="test")
    if not samples:
        raise ValueError(f"{traj.key}: khong ghep duoc sample nao ({stats})")

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    # Inference doc du lieu NHIEU that; khong sinh corruption cua train.
    dataset = PairedRestorationDataset(
        samples, master_seed=cfg.seed, realization=cfg.corruption.validation_realization,
        image_size=tuple(cfg.data.image_size),
        force_image_clean=args.clean_input, force_imu_clean=args.clean_input,
    )
    u_all, t_all = traj.load_imu()
    merger = TrajectoryMerger(len(u_all), cfg.data.imu_window_samples and 6)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    written = 0
    with torch.inference_mode():
        for i in range(0, len(dataset), cfg.train.batch_size):
            batch = collate([dataset[j] for j in range(i, min(i + cfg.train.batch_size, len(dataset)))])
            out = model(
                batch["image_bad"].to(device), batch["imu_bad_phys"].to(device),
                batch["image_time"].to(device), batch["imu_times"].to(device),
            )
            images = out["image_hat"].clamp(0, 1).cpu().numpy()
            imus = out["imu_hat_phys"].cpu().numpy().astype(np.float64)
            for k, sid in enumerate(batch["sample_id"]):
                arr = (images[k].transpose(1, 2, 0) * 255).round().astype(np.uint8)
                Image.fromarray(arr).save(out_dir / "images" / f"{sid}.png")
                merger.add(imus[k], int(batch["imu_start"][k]), sid)
                written += 1

    merged, mask = merger.result()
    np.save(out_dir / "imu_restored.npy", merged.T)          # luu [N,6] theo storage layout
    np.save(out_dir / "imu_timestamps.npy", t_all)
    np.save(out_dir / "imu_coverage_mask.npy", mask)
    with (out_dir / "imu_restored.csv").open("w", encoding="utf-8") as fh:
        fh.write("timestamp,ax,ay,az,gx,gy,gz,covered\n")
        for r in range(len(t_all)):
            vals = ",".join("" if not mask[r] else f"{merged[c, r]:.9g}" for c in range(6))
            fh.write(f"{t_all[r]:.9f},{vals},{int(mask[r])}\n")

    report = {
        "trajectory": traj.key,
        "checkpoint": str(args.checkpoint),
        "images_written": written,
        "images_skipped_at_border": stats["rejected_outside_center_range"],
        "imu_rows": len(t_all),
        "imu_rows_covered": int(mask.sum()),
        "imu_coverage": merger.coverage,
        "runtime_s": time.time() - started,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        "units": ["m_s2"] * 3 + ["rad_s"] * 3,
        "channel_order": ["ax", "ay", "az", "gx", "gy", "gz"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cmd_train_probe(args) -> None:
    """Train decoder danh gia tren backbone dong bang (themjacobian muc 13)."""
    from .training.probe_trainer import ProbeTrainer, backbone_fingerprint
    from .training.trainer import build_model

    cfg, built, normalizer = _load(args)
    device = _device(args.device)
    payload = torch.load(args.backbone, map_location=device, weights_only=False)
    backbone = build_model(cfg, normalizer).to(device)
    if payload.get("teacher_initialized"):
        backbone.initialize_teacher()
        backbone.to(device)
    backbone.load_state_dict(payload["model"], strict=True)

    trainer = ProbeTrainer(
        cfg, backbone, built["samples"], device=device,
        backbone_hash=backbone_fingerprint(backbone),
        output_dir=args.output_dir,
    )
    print(f"backbone hash {trainer.backbone_hash[:16]} | probe init hash {trainer.init_hash[:16]}")
    print(f"probe params {sum(p.numel() for p in trainer.probe.parameters())/1e6:.2f}M "
          f"| backbone dong bang: {trainer.verify_backbone_frozen()}")
    summary = trainer.train(max_steps=args.steps, log_every=args.log_every)
    print("\nprobe xong:", summary)
    if not summary["backbone_unchanged"]:
        raise RuntimeError("backbone DA BI DOI trong phase probe — vi pham freeze")
    trainer.validate()


def cmd_diagnose_sensitivity(args) -> None:
    """Do do nhay encoder va output tren mot bank co dinh (muc 15)."""
    from .evaluation.representation import sensitivity_report
    from .training.trainer import build_model

    cfg, built, normalizer = _load(args)
    device = _device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg, normalizer).to(device)
    if payload.get("teacher_initialized"):
        model.initialize_teacher()
        model.to(device)
    model.load_state_dict(payload["model"], strict=True)
    report = sensitivity_report(
        model, cfg, built["samples"][args.split], device=device,
        bank_samples=args.bank_samples,
    )
    out = Path(args.out or Path(args.checkpoint).parent) / "sensitivity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}, indent=2))
    print("->", out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qjepa", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit-data", help="audit dataset theo spec muc 2.4")
    a.add_argument("--root", required=True)
    a.add_argument("--window", type=int, default=128)
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_audit_data)

    b = sub.add_parser("build-manifest", help="ghep anh-IMU, chia split, tinh norm stats")
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--window", type=int, default=128)
    b.add_argument("--split-ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=cmd_build_manifest)

    c = sub.add_parser("check-transform", help="gate G1 round-trip/gradient")
    c.set_defaults(func=cmd_check_transform)

    def common(sp):
        sp.add_argument("--config", required=True)
        sp.add_argument("--manifest", required=True)
        sp.add_argument("--device", default="auto")
        sp.add_argument("--output-dir", default=None)

    s = sub.add_parser("smoke-train", help="gate G2 forward/backward")
    common(s); s.add_argument("--steps", type=int, default=30)
    s.set_defaults(func=cmd_smoke_train)

    o = sub.add_parser("overfit", help="gate G3 overfit nhom nho")
    common(o); o.add_argument("--steps", type=int, default=500)
    o.add_argument("--samples", type=int, default=16)
    o.set_defaults(func=cmd_overfit)

    t = sub.add_parser("train", help="train Stage A roi Stage B")
    common(t)
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--log-every", type=int, default=25)
    g = t.add_mutually_exclusive_group()
    g.add_argument("--resume", default=None, help="tiep tuc dung run (giu step/optimizer)")
    g.add_argument("--init", default=None, help="chi nap trong so, bat dau thi nghiem moi")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="danh gia checkpoint")
    common(e)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", default="valid")
    e.add_argument("--max-batches", type=int, default=None)
    e.add_argument("--allow-mismatch", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    i = sub.add_parser("infer", help="chay tren mot trajectory, xuat anh + IMU da ghep")
    common(i)
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--root", required=True)
    i.add_argument("--trajectory", required=True, help="vi du AmericanDiner/Data_easy/P003")
    i.add_argument("--out", required=True)
    i.add_argument("--clean-input", action="store_true",
                   help="dua anh/IMU SACH vao de do muc pha du lieu sach")
    i.set_defaults(func=cmd_infer)

    tp = sub.add_parser("train-probe", help="train decoder danh gia, backbone dong bang")
    common(tp)
    tp.add_argument("--backbone", required=True, help="checkpoint backbone da khoa")
    tp.add_argument("--steps", type=int, default=None)
    tp.add_argument("--log-every", type=int, default=50)
    tp.set_defaults(func=cmd_train_probe)

    dg = sub.add_parser("diagnose-sensitivity", help="do nhay encoder/output, chi diagnostic")
    common(dg)
    dg.add_argument("--checkpoint", required=True)
    dg.add_argument("--split", default="valid")
    dg.add_argument("--bank-samples", type=int, default=64)
    dg.add_argument("--out", default=None)
    dg.set_defaults(func=cmd_diagnose_sensitivity)

    x = sub.add_parser("export", help="export checkpoint inference")
    common(x)
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--name", default="inference.pt")
    x.add_argument("--allow-mismatch", action="store_true")
    x.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
