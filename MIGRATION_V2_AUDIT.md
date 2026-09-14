# MIGRATION_V2_AUDIT.md

Audit truoc khi sua source theo [themjacobian.md](themjacobian.md) v2.0.
Moi dong duoi day duoc **kiem tra bang lenh tren repo that**, khong suy doan.

## 1. Bang audit (themjacobian muc 2)

| Hang muc | File/ham that | Trang thai | Bang chung | Viec phai sua |
| --- | --- | --- | --- | --- |
| API/shape/data | `qjepa/models/joint_jepa.py::forward` | **PASS** | anh `[B,3,256,256]`, IMU `[B,6,128]`, 84 test | khong |
| FI/FU truoc fusion | `CoefficientEncoder.forward` tra `(dense, skips)` | **PASS** | `test_dense_api_matches_forward_features_exactly` | them `encode_*_dense` |
| QWT/Hamilton | `qjepa/transforms/qwt.py` | **PASS** transform, **ABSENT** Hamilton | round-trip 1.09e-07; Hilbert 82.8% | `learned_hamilton_layers=false` |
| Loss output v1 | — | **ABSENT** | `grep -rn "sensitivity\|output_gain\|jacobian" qjepa/` khong ra ket qua | them v2 truc tiep |
| JEPA/EMA | `initialize_teacher`, `update_teacher` | **PASS** | `test_ema_matches_hand_computation` | khong |
| Parent checkpoint | `outputs/*/last.pt` | **INVALID** | ba checkpoint deu `stage A`, `teacher=False`, 800 step overfit tren 16 sample | can Stage B that truoc khi so C/T |

## 2. Duong xu ly da chon

Theo muc 2, repo roi vao truong hop **"Chua implement migration v1"**:

> them v2 truc tiep; khong can xay output regularizer truoc

Khong co namespace v1 nao phai go bo. `output_sensitivity_loss_weight` va
`fusion_sensitivity_loss_weight` ton tai trong schema **chi de validator TU CHOI**
khi ai do dat khac 0 — khong co code cong chung vao loss.

## 3. Hamilton audit (muc 3)

| Cau hoi | Tra loi |
| --- | --- |
| Filter bank co dung QWT khong? | Co — dual-tree db4, hai cay lech mot mau; round-trip/gradient PASS; xem [QWT_AUDIT.md](QWT_AUDIT.md) |
| Co lop hoc dung Hamilton product khong? | **Khong.** `grep -rn "hamilton\|quaternion_mul\|qmul"` khong ra ket qua |
| Ket luan | `learned_hamilton_layers = false` |

QWT pack he so vao CNN **so thuc**. Khong doi ten `Conv2d` thuong thanh
quaternion convolution. Theo muc 3, migration nay **khong bat buoc them** lop
Hamilton, va `hamilton_audit.replace_learned_layers_in_this_migration = false`.

## 4. Parent checkpoint — han che quan trong

Khong co checkpoint nao dat `parent_stage_required: stable_stage_b`:

```
outputs/overfit/last.pt        step 800  stage A  teacher=False
outputs/ovf_balanced/last.pt   step 800  stage A  teacher=False
outputs/ovf_smallbeta/last.pt  step 800  stage A  teacher=False
```

Ba checkpoint nay la run **overfit tren 16 sample co dinh**, khong phai pilot
held-out. Theo muc 2, tinh huong nay la **"Mới có source, chưa train"**:

> chay gate cu, Stage A phuc hoi va Stage B JEPA truoc phase so sanh v2

Vi vay **G5 (pilot va danh gia that) = NOT_RUN**. Toan bo code duong C/T va probe
da chay duoc va co test, nhung **chua co so lieu chat luong nao** de ket luan
Jacobian regularization co loi hay khong.

## 5. Source da sua (muc 18.2)

| Nhom | File | Thay doi |
| --- | --- | --- |
| Config | `qjepa/config.py` | 6 section v2, `migration_schema_version`, validator chan xung dot v1/v2 |
| Encoder API | `qjepa/models/joint_jepa.py` | `encode_image_dense`, `encode_imu_dense_normalized`, `encode_and_fuse`, `freeze_backbone` |
| Loss | `qjepa/training/encoder_sensitivity.py` (**moi**) | LN khong affine + FD energy that, ramp, luan phien source, seed SHA-256 |
| Trainer backbone | `qjepa/training/trainer.py` | `_encoder_sensitivity_term`, source co dinh theo accumulation group, counter `sens_steps` |
| Decoder moi | `qjepa/models/representation_probe.py` (**moi**) | he so tuyet doi tu ZI/ZU, khong skip/residual |
| Trainer probe | `qjepa/training/probe_trainer.py` (**moi**) | freeze backbone, optimizer chi probe, `backbone_fingerprint` |
| Eval | `qjepa/evaluation/representation.py` (**moi**) | 4 output gain, latent stats, alpha sweep — chi diagnostic |
| CLI | `qjepa/cli.py` | `train-probe`, `diagnose-sensitivity` |
| Tests | `tests/test_encoder_sensitivity.py`, `tests/test_representation_probe.py` (**moi**) | 32 test cho G0–G4 |

Backbone **khong** them tham so hoc nao: van 2.244.668. Probe decoder la model
rieng (0,61 M) voi optimizer va checkpoint rieng.

## 6. Trang thai gate

| Gate | Trang thai | Bang chung |
| --- | --- | --- |
| G0 regression va vi tri feature | **PASS** | tat v2 tai lap **tung bit** (diff 0.0 tren CPU); `sens_steps=0`; FI doc lap IMU va nguoc lai |
| G1 toan hoc FD va normalization | **PASS** | LN mean≈0/var=1 theo channel, khop reference tinh tay; delta thuc sau clamp; ratio khop analytic tuyen tinh |
| G2 gradient routing | **PASS** | gradient chi toi encoder bi perturb; fusion/decoder/predictor/teacher/encoder kia deu khong; autograd khop central FD o float64 |
| G3 probe khong ro ri va frozen | **PASS** | signature chi `(zi, zu)`; backbone fingerprint khong doi sau optimizer step; init hash chung C/T |
| G4 checkpoint va config | **PASS** | 8 quy tac xung dot v1/v2 deu bi chan; config cu van chay |
| G5 pilot va danh gia that | **NOT_RUN** | khong co parent Stage B hop le (muc 4) |

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q` → **84 passed**.
