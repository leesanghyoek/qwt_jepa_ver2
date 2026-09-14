"""G3-G4: probe decoder khong ro ri, backbone that su frozen, config/checkpoint."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from qjepa.config import Config
from qjepa.models.joint_jepa import JointRestorationJEPA
from qjepa.models.representation_probe import RepresentationProbeDecoder
from qjepa.training.probe_trainer import backbone_fingerprint


def make_probe(backbone=None):
    m = backbone or JointRestorationJEPA().eval()
    return m, RepresentationProbeDecoder(m.image_transform, m.imu_transform)


def latents(b=2):
    return torch.randn(b, 128, 16, 16), torch.randn(b, 128, 8)


# ---------------- G3: khong ro ri thong tin ----------------
def test_forward_signature_accepts_only_latents():
    """Signature khong duoc co anh/IMU/skip/coefficients."""
    _, probe = make_probe()
    names = probe.forward.__code__.co_varnames[: probe.forward.__code__.co_argcount]
    assert names == ("self", "zi", "zu")


def test_output_depends_only_on_latents():
    """Cung ZI/ZU nhung doi input goc -> output KHONG doi (khong co bypass)."""
    _, probe = make_probe()
    probe.eval()
    zi, zu = latents()
    with torch.no_grad():
        a = probe(zi, zu)["image_hat"]
        b = probe(zi.clone(), zu.clone())["image_hat"]
    assert torch.equal(a, b)


def test_permuting_latents_permutes_output():
    _, probe = make_probe()
    probe.eval()
    zi, zu = latents(3)
    idx = torch.tensor([2, 0, 1])
    with torch.no_grad():
        out = probe(zi, zu)["image_hat"]
        permuted = probe(zi[idx], zu[idx])["image_hat"]
    assert torch.allclose(out[idx], permuted, atol=1e-6)


def test_identical_latents_give_identical_outputs():
    """Latent hang so -> moi sample ra giong nhau; khong tai hien anh goc."""
    _, probe = make_probe()
    probe.eval()
    zi = torch.ones(3, 128, 16, 16)
    zu = torch.ones(3, 128, 8)
    with torch.no_grad():
        out = probe(zi, zu)["image_hat"]
    assert torch.allclose(out[0], out[1], atol=1e-6)
    assert torch.allclose(out[0], out[2], atol=1e-6)


def test_predicts_absolute_coefficients_not_residual():
    """He so du doan KHONG duoc cong he so nhieu dau vao."""
    m, probe = make_probe()
    probe.eval()
    zi, zu = latents(1)
    with torch.no_grad():
        coeffs = probe(zi, zu)["image_coefficients"]
        # Neu la residual thi zero-latent se tra ve chinh he so cua mot anh nao do.
        zero = probe(torch.zeros_like(zi), torch.zeros_like(zu))["image_coefficients"]
    assert not torch.allclose(coeffs, zero)
    # Head khong zero-init (do la decoder tuyet doi, khong phai correction head).
    assert probe.image_branch.head.weight.abs().sum() > 0


def test_probe_has_no_reference_to_encoder_skips():
    _, probe = make_probe()
    names = [n for n, _ in probe.named_modules()]
    assert not any("skip" in n for n in names)
    source = open("qjepa/models/representation_probe.py").read()
    body = source.split("def forward(self, zi")[1].split("def state_hash")[0]
    for banned in ("image_skips", "imu_skips", "ci_bad", "cu_bad", "teacher", "predictor"):
        assert banned not in body, f"probe forward khong duoc nhac toi {banned}"


# ---------------- G3: backbone that su frozen ----------------
def test_freeze_backbone_blocks_gradient_and_keeps_eval():
    m = JointRestorationJEPA()
    m.freeze_backbone()
    for mod in (m.image_encoder, m.imu_encoder, m.shared_fusion):
        assert all(not p.requires_grad for p in mod.parameters())
        assert not mod.training


def test_backbone_unchanged_after_probe_optimizer_step():
    m = JointRestorationJEPA().eval()
    m.freeze_backbone()
    before = backbone_fingerprint(m)
    _, probe = make_probe(m)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    img = torch.rand(2, 3, 256, 256)
    imu = torch.randn(2, 6, 128)
    t = torch.full((2,), 1.0)
    ut = torch.stack([torch.linspace(0.365, 1.635, 128) for _ in range(2)])
    with torch.no_grad():
        zi, zu = m.encode_and_fuse(img, imu, t, ut)
    out = probe(zi, zu)
    out["image_hat"].square().mean().backward()
    opt.step()

    assert backbone_fingerprint(m) == before, "backbone bi doi trong phase probe"
    assert all(p.grad is None for p in m.image_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in probe.parameters())


def test_gradient_flows_through_synthesis_to_probe():
    _, probe = make_probe()
    zi, zu = latents(1)
    out = probe(zi, zu)
    (out["image_hat"].mean() + out["imu_hat_norm"].mean()).backward()
    grads = [p.grad for p in probe.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_shared_initialization_hash_for_control_and_treatment():
    """Control va treatment phai bat dau tu CUNG initial state."""
    from qjepa.training.trainer import set_seed

    m = JointRestorationJEPA().eval()
    set_seed(73131); a = RepresentationProbeDecoder(m.image_transform, m.imu_transform)
    set_seed(73131); b = RepresentationProbeDecoder(m.image_transform, m.imu_transform)
    assert a.state_hash() == b.state_hash()
    set_seed(99999); c = RepresentationProbeDecoder(m.image_transform, m.imu_transform)
    assert c.state_hash() != a.state_hash()


# ---------------- G4: config va checkpoint ----------------
def test_v2_config_conflicts_are_rejected():
    base = Config()
    es, rp = base.encoder_sensitivity, base.representation_probe

    with pytest.raises(ValueError, match="migration_schema_version=2"):
        replace(base, encoder_sensitivity=replace(es, enabled=True)).validate()
    with pytest.raises(ValueError, match="online_dense_before_fusion"):
        replace(base, migration_schema_version=2,
                encoder_sensitivity=replace(es, enabled=True, target="restored_outputs")).validate()
    with pytest.raises(ValueError, match="affine"):
        replace(base, migration_schema_version=2,
                encoder_sensitivity=replace(es, enabled=True,
                                            measurement_normalization="layer_norm")).validate()
    with pytest.raises(ValueError, match="detach_base_feature"):
        replace(base, migration_schema_version=2,
                encoder_sensitivity=replace(es, enabled=True, detach_base_feature=True)).validate()
    with pytest.raises(ValueError, match="namespace v1|output/fusion"):
        replace(base, migration_schema_version=2,
                encoder_sensitivity=replace(es, enabled=True,
                                            output_sensitivity_loss_weight=0.1)).validate()
    with pytest.raises(ValueError, match="encoder_skips"):
        replace(base, representation_probe=replace(rp, enabled=True, encoder_skips=True)).validate()
    with pytest.raises(ValueError, match="DONG BANG"):
        replace(base, representation_probe=replace(rp, enabled=True, freeze_fusion=False)).validate()
    with pytest.raises(ValueError, match="TUYET DOI|absolute"):
        replace(base, representation_probe=replace(rp, enabled=True,
                                                   coefficients="residual")).validate()


def test_old_config_without_v2_sections_still_valid():
    """Config cu khong co section moi -> v2 tat, lich train cu khong doi."""
    cfg = Config()
    cfg.validate()
    assert cfg.migration_schema_version == 1
    assert not cfg.encoder_sensitivity.enabled
    assert not cfg.representation_probe.enabled


def test_shipped_v2_configs_load():
    from pathlib import Path

    from qjepa.config import load_config

    for path in sorted(Path("configs").glob("v2_*.yaml")):
        cfg = load_config(path)
        assert cfg.migration_schema_version == 2
