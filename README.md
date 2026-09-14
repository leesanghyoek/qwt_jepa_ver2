# QWT–JEPA v2 — phuc hoi chung anh RGB 256×256 va IMU 128×6

Trien khai [kientruc2.md](kientruc2.md) (dac ta v3.0). Moi sample la **mot frame
RGB 256×256** va **mot cua so IMU 128 hang × 6 kenh** quanh timestamp cua frame
do. Model phuc hoi ca hai.

```
anh nhieu  ──> QWT 2D dual-tree ──> CNN2D encoder ─┐
                                                   ├─> shared gated MLP fusion ─┬─> decoder anh ──> iQWT  ──> anh phuc hoi
IMU nhieu  ──> Haar 1D          ──> CNN1D encoder ─┘                            └─> decoder IMU ──> iHaar ──> IMU phuc hoi
                                          │                                              │
                                          └──> predictor ──> loss JEPA <── teacher EMA (du lieu sach, chi khi train)
```

## Trang thai da kiem chung

| Gate | Noi dung | Trang thai |
| --- | --- | --- |
| G0 | Audit du lieu, pairing, split, corruption doc lap | **PASS** — [DATA_AUDIT.md](DATA_AUDIT.md) |
| G1 | QWT/Haar round-trip va gradient cua inverse | **PASS** — [QWT_AUDIT.md](QWT_AUDIT.md) |
| G2 | Forward/backward, EMA, teacher tach biet | **PASS** |
| G3 | Overfit nhom co dinh | **MOT PHAN** — anh dat, IMU chua; xem [TRAINING_REPORT.md](TRAINING_REPORT.md) |
| G4 | Pilot held-out + JEPA | **NOT_RUN** — can GPU Kaggle |
| G5 | Save/load/resume/export | **PASS** |

84 test: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q`

> GPU **khong deterministic** (cuDNN autotune): hai run giong het nhau lech
> ~9,6e-05 o muc tham so. Tren CPU sai khac la 0,0 va resume tai epoch boundary
> trung khop **tung bit**. Dung chenh lech nho hon ~1e-4 de ket luan khac biet.

**Chua co bang chung** ve: chat luong phuc hoi tren held-out, loi ich cua JEPA,
loi ich cua cross-modal fusion, hay loi ich cua QWT so voi Haar. Nhung cau hoi
do can cac run trong bang ablation ben duoi.

> Kien truc chi tiet — luong du lieu, cong thuc tung khoi, so tham so tung module:
> **[KIEN_TRUC.md](KIEN_TRUC.md)**

## Migration v2: Jacobian tai encoder + probe decoder

Theo [themjacobian.md](themjacobian.md) v2.0. **Mac dinh TAT** — config cu chay
khong doi (parity tung bit da kiem).

| Thanh phan | File | Trang thai |
| --- | --- | --- |
| FD sensitivity tai FI/FU truoc fusion | `qjepa/training/encoder_sensitivity.py` | G0–G2 **PASS** |
| Probe decoder tren backbone dong bang | `qjepa/models/representation_probe.py` | G3 **PASS** |
| Trainer probe | `qjepa/training/probe_trainer.py` | G3 **PASS** |
| Diagnostic do nhay/collapse | `qjepa/evaluation/representation.py` | implemented |

```bash
# treatment: bat Jacobian regularization tai encoder
python -m qjepa train --config configs/v2_treatment.yaml --manifest outputs/manifest     --init <parent_stage_b.pt>
# control: cung parent, cung budget, khong sensitivity
python -m qjepa train --config configs/v2_control.yaml --manifest outputs/manifest     --init <parent_stage_b.pt>
# probe: backbone dong bang, chi train decoder danh gia
python -m qjepa train-probe --config configs/v2_probe.yaml --manifest outputs/manifest     --backbone outputs/v2_treatment/last.pt
# diagnostic (khong train)
python -m qjepa diagnose-sensitivity --config configs/v2_treatment.yaml     --manifest outputs/manifest --checkpoint outputs/v2_treatment/last.pt
```

Bao cao: [MIGRATION_V2_AUDIT.md](MIGRATION_V2_AUDIT.md),
[ENCODER_SENSITIVITY_REPORT.md](ENCODER_SENSITIVITY_REPORT.md),
[REPRESENTATION_PROBE_REPORT.md](REPRESENTATION_PROBE_REPORT.md).

> **G5 = NOT_RUN.** Chua co parent Stage B hop le nen phase control/treatment
> chua chay. Chua co bang chung nao ve loi ich cua Jacobian regularization.

## Kich thuoc tensor (da xac minh bang test)

| Tensor | Shape |
| --- | --- |
| Anh vao / ra | `[B,3,256,256]` |
| He so QWT | `[B,48,128,128]` — 3 RGB × 4 band × 4 quaternion component |
| `FI` (dac trung anh) | `[B,128,16,16]` |
| IMU vao / ra (model) | `[B,6,128]` — storage tren dia la `[128,6]` |
| He so Haar IMU | `[B,12,64]` |
| `FU` (dac trung IMU) | `[B,128,8]` |
| Summary `gI`, `gU`, `s` | `[B,128]` moi cai |
| Metadata thoi gian `m` | `[B,3]` |
| Dau vao shared MLP | `[B,259]` = 128 + 128 + 3 |
| Token JEPA anh / IMU | `[B,256,128]` / `[B,8,128]` |

Tham so: **2,24 M** online (+1,00 M teacher chi khi train).

> `128` trong `[B,128,8]` la so **channels**, con `8` la chieu thoi gian con lai.
> Model van phuc hoi du 128 hang: decoder upsample ve 64 vi tri he so roi iHaar
> tra lai 128 hang.

## Chay tren Kaggle

Notebook san sang: [kaggle/qwt_jepa_train.ipynb](kaggle/qwt_jepa_train.ipynb)
(Accelerator `GPU T4 x2`, Internet `ON`, add dataset `tartanair-v2-256`).

Do duoc tren GPU: batch 8 ton **0,9 GB** VRAM va dat throughput cao nhat (87,6
sample/s); tang batch chi ton them VRAM ma khong nhanh hon. Nut that that su la
**data loading**, nen config Kaggle dat `num_workers=4` (61 -> 184 sample/s).

Loader doc `.npy` neu co, nguoc lai `.txt` — **dataset resize tren Kaggle chi co
`.txt`** vi cell resize bo qua `.npy`. Hai dinh dang da doi chieu trung khop tuyet doi.

## Chay o may local

```bash
python -m qjepa audit-data     --root /home/buidinhkhoi/Datasets/tartanair-v2-jepa
python -m qjepa check-transform
python -m qjepa build-manifest --root /home/buidinhkhoi/Datasets/tartanair-v2-jepa \
                               --out outputs/manifest --window 128
python -m qjepa smoke-train --config configs/main_qwt.yaml --manifest outputs/manifest --steps 30
python -m qjepa overfit     --config configs/main_qwt.yaml --manifest outputs/manifest --samples 16 --steps 800
python -m qjepa train       --config configs/main_qwt.yaml --manifest outputs/manifest
python -m qjepa evaluate    --config configs/main_qwt.yaml --manifest outputs/manifest \
                            --checkpoint outputs/main_qwt/last.pt
python -m qjepa export      --config configs/main_qwt.yaml --manifest outputs/manifest \
                            --checkpoint outputs/main_qwt/last.pt
```

## Config

| File | Vai tro |
| --- | --- |
| `main_qwt.yaml` | **cau hinh muc tieu** theo dac ta muc 16 |
| `kaggle_t4.yaml` | main_qwt + batch 8, fp16, 2 worker |
| `main_qwt_balanced.yaml` | `imu_weight=100` — sua mat can bang gradient (xem duoi) |
| `main_qwt_smallbeta.yaml` | `smooth_l1_beta=0.01`, `imu_weight=5` — cach sua thu hai |
| `kaggle_balanced.yaml` | **khuyen dung tren Kaggle**: kaggle_t4 + `imu_weight=100` |
| `haar_baseline.yaml` | thay QWT bang Haar 12 kenh — **khong hoan thanh yeu cau QWT** |
| `no_cross.yaml` | `cross_modal=false` — do dong gop cua fusion |
| `no_jepa.yaml` | control `lambda_J=0` toan bo — do dong gop cua JEPA |
| `stage_c.yaml` | nhieu nang: bias + drift + spike |

Schema **strict**: field go sai ten bi reject o moi cap long nhau, khong im lang
chay bang mac dinh khac y dinh. `imu_window_samples` khac 128 cung bi reject.

## Phat hien quan trong: mat can bang gradient hai nhanh

Voi `w_image = w_imu = 1` nhu dac ta, gradient vao hai decoder lech **~88 lan**:

| Nhanh | Loss | Grad norm vao decoder |
| --- | ---: | ---: |
| `L_image` | 2.874e-02 | 1.042e-01 |
| `L_imu` | 3.985e-05 | 1.186e-03 |

Nguyen nhan so hoc: nhieu IMU sau normalize chi khoang `0.006` rms, ma
`SmoothL1(beta=1.0)` o do van nam trong **vung quadratic** nen dao ham chi
`x/beta ≈ 0.006` thay vi `1.0`. Hau qua: trong gate G3 anh giam 29,6% MSE con
accel/gyro gan nhu khong doi.

Dac ta muc 13 cho phep chinh weight bang pilot va ghi vao **config moi** — do la
`main_qwt_balanced.yaml` va `main_qwt_smallbeta.yaml`. Ket qua doi chung o
[TRAINING_REPORT.md](TRAINING_REPORT.md).

## Ban do ma nguon

| File | Vai tro |
| --- | --- |
| `qjepa/transforms/qwt.py` | QWT 2D dual-tree (db4, hai cay lech mot mau) |
| `qjepa/transforms/haar.py` | Haar 1D cho IMU |
| `qjepa/transforms/haar_image.py` | Haar 2D — baseline debug co ten ro rang |
| `qjepa/models/encoders.py` | CNN2D / CNN1D encoder |
| `qjepa/models/fusion.py` | shared gated MLP + metadata thoi gian |
| `qjepa/models/decoders.py` | decoder he so + predictor JEPA |
| `qjepa/models/joint_jepa.py` | model chung, teacher EMA, export |
| `qjepa/data/tartanair.py` | quet trajectory, audit, doc `.npy`/`.txt` |
| `qjepa/data/manifest.py` | ghep centered window, split, norm stats |
| `qjepa/corruptions/` | suy giam anh va IMU doc lap, seed SHA-256 |
| `qjepa/training/trainer.py` | Stage A/B, EMA, checkpoint, resume |
| `qjepa/evaluation/metrics.py` | MAE/PSNR/SSIM, metric IMU vat ly |
| `qjepa/training/encoder_sensitivity.py` | FD Jacobian tai FI/FU truoc fusion (v2) |
| `qjepa/models/representation_probe.py` | decoder danh gia, chi doc ZI/ZU (v2) |
| `qjepa/training/probe_trainer.py` | train probe, backbone dong bang (v2) |
| `qjepa/evaluation/representation.py` | do nhay va latent collapse (v2) |
| `qjepa/cli.py` | CLI |

## Gioi han da biet

- Mot anh don khong quan sat truc tiep chuyen dong giua cac frame, nen **loi ich
  cua fusion anh–IMU cho khu nhieu chua duoc bao dam** — phai do bang `no_cross.yaml`.
- `centered_offline` nhin truoc ~0,635 s quanh anh: **khong phai pipeline causal
  thoi gian thuc**.
- Cac window chong lan ~92%: khong coi so window la so quan sat doc lap.
- Chua doi chieu he so QWT voi mot implementation thu ba doc lap (xem QWT_AUDIT muc 7).
