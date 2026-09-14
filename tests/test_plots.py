"""Kiem module ve do thi chan doan."""

from __future__ import annotations

import json

import pytest

from qjepa.evaluation.plots import diagnose, plot_training, read_jsonl


def _write(tmp_path, train, val):
    (tmp_path / "train_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train), encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val), encoding="utf-8")
    return tmp_path


def _train_rows(n=40, image_start=0.05, image_end=0.02, jepa_from=None):
    rows = []
    for i in range(n):
        f = i / max(n - 1, 1)
        r = {
            "step": i + 1, "epoch": 0, "stage": "A",
            "loss_image": image_start + (image_end - image_start) * f,
            "loss_imu": 3e-5, "loss_acc": 3e-5, "loss_gyro": 2e-5,
            "grad_norm": 0.1, "lr": 2e-4, "lambda_jepa": 0.0,
        }
        if jepa_from is not None and i >= jepa_from:
            r["stage"] = "B"
            r["lambda_jepa"] = 0.1
            r["loss_jepa"] = 0.5
        r["loss_total"] = r["loss_image"] + r["loss_imu"] + r["lambda_jepa"] * r.get("loss_jepa", 0)
        rows.append(r)
    return rows


def _val_rows(r_image, r_acc, r_gyro, n=4):
    return [{"step": (i + 1) * 10, "stage": "A",
             "r_image": r_image, "r_acc": r_acc, "r_gyro": r_gyro,
             "image_psnr": 25.0, "image_ssim": 0.8, "image_mae": 0.04,
             "image_mse": 0.001, "image_mse_bad": 0.001,
             "image_out_of_range_fraction": 0.01,
             "improves_all_three": r_image < 1 and r_acc < 1 and r_gyro < 1}
            for i in range(n)]


def _status(rows, name):
    return next(s for n, s, _ in rows if n == name)


def test_detects_healthy_run():
    rows = diagnose(_train_rows(), _val_rows(0.7, 0.8, 0.9))
    assert _status(rows, "Loss phuc hoi giam") == "OK"
    assert _status(rows, "Anh tot hon identity") == "OK"
    assert _status(rows, "Ca ba nguon cung tot") == "OK"


def test_flags_model_worse_than_identity():
    """r > 1 nghia la model lam XAU hon input — phai bao XAU."""
    rows = diagnose(_train_rows(), _val_rows(1.2, 1.1, 1.05))
    for name in ("Anh tot hon identity", "Accel tot hon identity", "Gyro tot hon identity"):
        assert _status(rows, name) == "XAU"
    assert _status(rows, "Ca ba nguon cung tot") == "XAU"


def test_flags_loss_going_up():
    rows = diagnose(_train_rows(image_start=0.02, image_end=0.05), _val_rows(0.9, 0.9, 0.9))
    assert _status(rows, "Loss phuc hoi giam") == "XAU"


def test_jepa_term_does_not_fake_a_rising_loss():
    """Vao Stage B, loss_total nhay len mot cach hop le.

    Chan doan phai nhin phan PHUC HOI, khong duoc bao 'XAU' chi vi cong them
    so hang JEPA.
    """
    train = _train_rows(jepa_from=20)
    assert train[-1]["loss_total"] > train[0]["loss_total"], "fixture: tong phai tang"
    rows = diagnose(train, _val_rows(0.9, 0.9, 0.9))
    assert _status(rows, "Loss phuc hoi giam") == "OK"


def test_detects_overfitting():
    train = _train_rows()
    val = _val_rows(0.9, 0.9, 0.9, n=6)
    for i, r in enumerate(val):
        r["image_mae"] = 0.03 + 0.01 * i          # validation xau dan
    rows = diagnose(train, val)
    assert _status(rows, "Khong overfit") == "XAU"


def test_reports_insufficient_data_instead_of_guessing():
    rows = diagnose(_train_rows(n=3), [])
    assert _status(rows, "Loss phuc hoi giam") == "CHUA DU"
    assert _status(rows, "Anh tot hon identity") == "CHUA DU"


def test_read_jsonl_skips_broken_lines(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a":1}\nkhong phai json\n\n{"a":2}\n', encoding="utf-8")
    assert read_jsonl(f) == [{"a": 1}, {"a": 2}]
    assert read_jsonl(tmp_path / "khong-ton-tai.jsonl") == []


def test_plot_training_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    _write(tmp_path, _train_rows(jepa_from=20), _val_rows(0.8, 0.95, 1.02))
    out = plot_training(tmp_path)
    assert out.exists() and out.stat().st_size > 10_000


def test_plot_training_fails_loudly_without_logs(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(FileNotFoundError):
        plot_training(tmp_path)
