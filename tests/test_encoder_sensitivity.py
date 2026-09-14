"""G0-G2: regression, toan hoc FD/LN va gradient routing (themjacobian muc 16)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from qjepa.models.joint_jepa import JointRestorationJEPA
from qjepa.training.encoder_sensitivity import (
    EncoderSensitivityConfig,
    encoder_fd_loss,
    encoder_weight,
    make_encoder_probe,
    mean_per_sample,
    normalized_dense_for_measurement,
    raw_feature_fd_gain,
    seeded_rademacher,
    source_for_update,
)


def batch(b=2):
    return (
        torch.rand(b, 3, 256, 256), torch.randn(b, 6, 128),
        torch.full((b,), 1.0),
        torch.stack([torch.linspace(0.365, 1.635, 128) for _ in range(b)]),
    )


# ---------------- G0: vi tri feature va regression ----------------
def test_dense_api_matches_forward_features_exactly():
    """encode_*_dense phai tra DUNG FI/FU ma forward dung, cung instance."""
    m = JointRestorationJEPA().eval()
    img, imu, t, ut = batch()
    with torch.no_grad():
        base = m(img, imu, t, ut)
        fi = m.encode_image_dense(img)
        fu = m.encode_imu_dense_normalized(m.normalizer.normalize(imu))
    assert torch.equal(base["fi"], fi)
    assert torch.equal(base["fu"], fu)


def test_fi_depends_only_on_image_and_fu_only_on_imu():
    """FI/FU lay TRUOC fusion, nen doc lap modality kia."""
    m = JointRestorationJEPA().eval()
    img, imu, t, ut = batch()
    with torch.no_grad():
        a = m(img, imu, t, ut)
        b = m(img, imu + 7.0, t, ut)
        c = m(torch.rand_like(img), imu, t, ut)
    assert torch.equal(a["fi"], b["fi"])          # doi IMU khong doi FI
    assert torch.equal(a["fu"], c["fu"])          # doi anh khong doi FU
    assert not torch.equal(a["fu"], b["fu"])


def test_disabled_adds_no_parameters():
    """Helper moi khong them tham so hoc vao backbone."""
    m = JointRestorationJEPA()
    assert sum(p.numel() for p in m.parameters()) == 2_244_668 + 0


# ---------------- G1: toan hoc FD va normalization ----------------
def test_layer_norm_measurement_has_no_affine_and_right_axis():
    f = torch.randn(3, 128, 5, 7) * 4 + 2
    h = normalized_dense_for_measurement(f)
    assert h.shape == f.shape
    # Chuan hoa theo CHANNEL tai tung vi tri (h,w), khong theo batch/spatial.
    assert torch.allclose(h.mean(dim=1), torch.zeros(3, 5, 7), atol=1e-5)
    assert torch.allclose(h.var(dim=1, unbiased=False), torch.ones(3, 5, 7), atol=1e-3)
    # Doi chieu voi reference tinh tay.
    manual = (f - f.mean(1, keepdim=True)) / (f.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    assert torch.allclose(h, manual, atol=1e-5)


def test_layer_norm_is_scale_invariant_for_positive_scaling():
    """f' = c*f voi variance du lon -> LN gain gan bat bien scale (muc 16 G1.6)."""
    f = torch.randn(2, 128, 4, 4) * 3
    h1 = normalized_dense_for_measurement(f)
    h2 = normalized_dense_for_measurement(f * 5.0)
    assert torch.allclose(h1, h2, atol=1e-4)


def test_probe_uses_actual_delta_after_clamp_for_image():
    """Anh o bien bi clamp -> energy phai la delta THUC, nho hon nominal."""
    x = torch.ones(1, 3, 8, 8)                 # toan bo o bien tren
    v = torch.ones_like(x)                     # day len tren -> bi clamp het
    with pytest.raises(ValueError, match="perturbation bi mat"):
        make_encoder_probe(x, "image", v)

    x2 = torch.full((1, 3, 8, 8), 0.5)
    xp, energy, clipped = make_encoder_probe(x2, "image", torch.ones_like(x2))
    assert clipped == 0.0
    assert torch.allclose(energy, torch.tensor([(1 / 255) ** 2]), rtol=1e-5)


def test_probe_does_not_clamp_imu():
    x = torch.full((1, 6, 128), 3.0)           # ngoai [0,1] -> hop le voi IMU
    xp, energy, clipped = make_encoder_probe(x, "imu", torch.ones_like(x))
    assert clipped == 0.0
    assert torch.allclose(energy, torch.tensor([0.01**2]), rtol=1e-4)
    assert (xp > 1.0).all()                    # khong bi clamp ve [0,1]


def test_large_imu_magnitude_loses_perturbation_in_fp32():
    """Canh bao that: o bien do IMU lon, fp32 khong giu du 0.01.

    Code do delta THUC nen van dung, nhung energy lech khoi nominal. Ghi lai
    bang test de khong ai tuong nham energy luon bang epsilon^2.
    """
    x = torch.full((1, 6, 128), 50.0)
    _, energy, _ = make_encoder_probe(x, "imu", torch.ones_like(x))
    assert energy.item() < 0.01**2                       # nho hon nominal
    assert energy.item() == pytest.approx(0.01**2, rel=1e-3)


def test_probe_does_not_mutate_input():
    x = torch.rand(2, 3, 16, 16)
    original = x.clone()
    make_encoder_probe(x, "image", seeded_rademacher(x, probe_seed=1, context=("a",)))
    assert torch.equal(x, original)


def test_denominator_is_mean_delta_squared_not_v1_formula():
    """Mau so la mean(delta^2) trong core units, khong phai cong thuc output gain v1."""
    x = torch.full((1, 6, 128), 0.0)
    xp, energy, _ = make_encoder_probe(x, "imu", torch.ones_like(x), alpha=2.0)
    assert torch.allclose(energy, torch.tensor([(2.0 * 0.01) ** 2]), rtol=1e-5)


def test_fd_ratio_matches_analytic_linear_reference():
    """Voi f(x)=A x tuyen tinh (chua LN), ratio FD khop |A delta|^2/|delta|^2."""
    torch.manual_seed(0)
    a = torch.randn(4, 4)
    x = torch.randn(1, 4, 1)
    delta = torch.full_like(x, 0.01)
    f, fp = a @ x[:, :, 0].T, a @ (x + delta)[:, :, 0].T
    energy = mean_per_sample(delta.square())
    got = raw_feature_fd_gain(f.T[..., None], fp.T[..., None], energy)
    want = (a @ delta[:, :, 0].T).square().mean() / delta.square().mean()
    assert torch.allclose(got, want.reshape(1), rtol=1e-4)


def test_constant_feature_gives_zero_loss_but_is_detectable():
    """h hang so -> loss 0; collapse monitor phai phat hien duoc, khong phai loss."""
    f = torch.ones(4, 128, 4, 4)
    loss, gain = encoder_fd_loss(f, f.clone(), torch.full((4,), 1e-4))
    assert loss.item() == 0.0
    from qjepa.evaluation.representation import latent_statistics
    stats = latent_statistics(f)
    assert stats["std_across_samples_mean"] == 0.0      # co the phat hien


def test_rejects_bad_inputs():
    x = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="source"):
        make_encoder_probe(x, "audio", torch.ones_like(x))
    with pytest.raises(ValueError, match="Rademacher"):
        make_encoder_probe(x, "image", torch.full_like(x, 0.5))
    with pytest.raises(ValueError, match="direction"):
        make_encoder_probe(x, "image", torch.ones(1, 3, 4, 4))
    with pytest.raises(ValueError, match="FP32"):
        make_encoder_probe(x.double(), "image", torch.ones_like(x).double())
    with pytest.raises(ValueError, match="NaN|Inf|huu han|finite"):
        bad = x.clone(); bad[0, 0, 0, 0] = float("nan")
        make_encoder_probe(bad, "image", torch.ones_like(x))
    with pytest.raises(ValueError, match="shape lech"):
        encoder_fd_loss(torch.randn(2, 8, 4), torch.randn(2, 8, 5), torch.ones(2))


def test_ramp_and_source_alternation():
    cfg = EncoderSensitivityConfig(weight_max=1e-4, ramp_updates=200)
    assert encoder_weight(0, cfg) == pytest.approx(1e-4 / 200)
    assert encoder_weight(199, cfg) == pytest.approx(1e-4)
    assert encoder_weight(10_000, cfg) == pytest.approx(1e-4)
    assert [source_for_update(s) for s in range(4)] == ["image", "imu", "image", "imu"]


def test_seeded_directions_are_stateless_and_distinct():
    x = torch.rand(2, 3, 8, 8)
    a = seeded_rademacher(x, probe_seed=7, context=("run", 3, "image"))
    assert torch.equal(a, seeded_rademacher(x, probe_seed=7, context=("run", 3, "image")))
    assert not torch.equal(a, seeded_rademacher(x, probe_seed=7, context=("run", 4, "image")))
    assert not torch.equal(a, seeded_rademacher(x, probe_seed=8, context=("run", 3, "image")))
    # Khong tieu thu global RNG.
    torch.manual_seed(0); before = torch.rand(3)
    torch.manual_seed(0); seeded_rademacher(x, probe_seed=7, context=("z",)); after = torch.rand(3)
    assert torch.equal(before, after)


# ---------------- G2: gradient routing ----------------
def test_gradient_reaches_only_the_perturbed_encoder():
    """Loss do nhay chi train encoder cua source bi perturb (muc 7)."""
    m = JointRestorationJEPA()
    img, imu, t, ut = batch()
    out = m(img, imu, t, ut)
    imu_norm = m.normalizer.normalize(imu)
    v = seeded_rademacher(img, probe_seed=1, context=("g",))
    xp, energy, _ = make_encoder_probe(img, "image", v)
    loss, _ = encoder_fd_loss(out["fi"], m.encode_image_dense(xp), energy)

    groups = {
        "image_encoder": m.image_encoder, "imu_encoder": m.imu_encoder,
        "shared_fusion": m.shared_fusion, "image_decoder": m.image_decoder,
        "imu_decoder": m.imu_decoder, "image_predictor": m.image_predictor,
    }
    grads = {
        name: torch.autograd.grad(loss, list(mod.parameters()),
                                  retain_graph=True, allow_unused=True)
        for name, mod in groups.items()
    }
    def has_grad(gs):
        return any(g is not None and g.abs().sum() > 0 for g in gs)
    assert has_grad(grads["image_encoder"]), "encoder bi perturb phai nhan gradient"
    for name in ("imu_encoder", "shared_fusion", "image_decoder", "imu_decoder", "image_predictor"):
        assert not has_grad(grads[name]), f"{name} KHONG duoc nhan gradient tu loss do nhay"


def test_base_feature_is_not_detached():
    """Neu ai do detach nhanh base, gradient se khac -> test phai bat duoc."""
    m = JointRestorationJEPA()
    img, imu, t, ut = batch(1)
    out = m(img, imu, t, ut)
    v = seeded_rademacher(img, probe_seed=2, context=("d",))
    xp, energy, _ = make_encoder_probe(img, "image", v)
    probe_f = m.encode_image_dense(xp)

    params = list(m.image_encoder.parameters())
    g_full = torch.autograd.grad(
        encoder_fd_loss(out["fi"], probe_f, energy)[0], params, retain_graph=True)
    g_detached = torch.autograd.grad(
        encoder_fd_loss(out["fi"].detach(), probe_f, energy)[0], params, retain_graph=True)
    diff = max((a - b).abs().max().item() for a, b in zip(g_full, g_detached))
    assert diff > 1e-9, "detach base feature phai lam doi gradient"


def test_teacher_gets_no_gradient_from_sensitivity():
    m = JointRestorationJEPA()
    m.initialize_teacher()
    img, imu, t, ut = batch(1)
    out = m(img, imu, t, ut)
    v = seeded_rademacher(img, probe_seed=3, context=("t",))
    xp, energy, _ = make_encoder_probe(img, "image", v)
    encoder_fd_loss(out["fi"], m.encode_image_dense(xp), energy)[0].backward()
    assert all(p.grad is None for p in m.teacher_image_encoder.parameters())


def test_autograd_matches_finite_difference_on_toy_encoder():
    """Toy f_theta(x)=A(theta)x qua DUNG duong LN nhu loss that."""
    torch.manual_seed(0)
    theta = torch.randn(6, 6, dtype=torch.float64, requires_grad=True)
    x = torch.randn(1, 6, 3, dtype=torch.float64)
    delta = torch.full_like(x, 1e-3)

    def scalar_loss(th):
        f = torch.einsum("ij,bjt->bit", th, x)
        fp = torch.einsum("ij,bjt->bit", th, x + delta)
        return encoder_fd_loss(f, fp, mean_per_sample(delta.square()))[0]

    analytic = torch.autograd.grad(scalar_loss(theta), theta)[0]
    eps, numeric = 1e-6, torch.zeros_like(theta)
    with torch.no_grad():
        for i in range(6):
            for j in range(6):
                up, dn = theta.clone(), theta.clone()
                up[i, j] += eps; dn[i, j] -= eps
                numeric[i, j] = (scalar_loss(up) - scalar_loss(dn)) / (2 * eps)
    assert torch.allclose(analytic, numeric, atol=1e-6, rtol=1e-4)


# ---------------- G0: parity khi tat ----------------
def _short_run(tmp_path, name, enabled):
    """Chay 8 step tren CPU voi CUNG config, chi doi co enabled."""
    from dataclasses import replace

    from qjepa.config import load_config
    from qjepa.data.manifest import read_manifest
    from qjepa.data.normalize import ImuNormalizer
    from qjepa.training.trainer import Trainer, set_seed

    built = read_manifest("outputs/manifest_local")
    stats = built["meta"]["normalization"]
    cfg = load_config("configs/v2_treatment.yaml")
    cfg = replace(
        cfg, output_dir=str(tmp_path / name),
        encoder_sensitivity=replace(cfg.encoder_sensitivity, enabled=enabled),
        train=replace(cfg.train, stage_a_optimizer_steps=20, stage_b_optimizer_steps=0,
                      max_optimizer_steps=20, validation_every_updates=0,
                      checkpoint_every_updates=0, batch_size=2, gradient_accumulation=1),
    )
    set_seed(cfg.seed)
    normalizer = ImuNormalizer(
        mean=torch.tensor(stats["mean"]).float(), std=torch.tensor(stats["std"]).float()
    )
    trainer = Trainer(
        cfg, {"train": built["samples"]["train"][:16],
              "valid": built["samples"]["train"][:16], "test": []},
        normalizer, device="cpu", manifest_hash=built["meta"]["manifest_hash"],
    )
    trainer.train(max_steps=8, log_every=10**6)
    return trainer


def _max_diff(a, b):
    sa, sb = a.model.state_dict(), b.model.state_dict()
    return max((sa[k].float() - sb[k].float()).abs().max().item()
               for k in sa if sa[k].dtype.is_floating_point)


@pytest.mark.skipif(
    not __import__("pathlib").Path("outputs/manifest_local/meta.json").exists(),
    reason="can manifest tai outputs/manifest_local",
)
def test_disabled_is_bit_exact_and_enabled_differs(tmp_path):
    """Tat v2 -> khong extra forward, khong tieu thu RNG, ket qua khong doi."""
    off_a = _short_run(tmp_path, "off_a", False)
    off_b = _short_run(tmp_path, "off_b", False)
    on = _short_run(tmp_path, "on", True)

    assert _max_diff(off_a, off_b) == 0.0, "tat v2 phai tai lap tung bit tren CPU"
    assert _max_diff(off_a, on) > 0.0, "bat v2 phai doi ket qua (loss co them so hang)"
    assert off_a.sens_steps == 0
    assert on.sens_steps == 8
