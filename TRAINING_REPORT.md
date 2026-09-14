# TRAINING_REPORT.md

## 0. KET QUA PILOT DAU TIEN — Kaggle 2x T4, 14 environment

Chay ngay 15/09/2026. **Day la ket qua that dau tien tren du lieu day du**, thay
cho trang thai `NOT_RUN` truoc do cua G4.

| | |
| --- | --- |
| Du lieu | 14 environment, 166 trajectory, ~63.000 sample |
| Cau hinh | `kaggle_balanced.yaml` (`imu_weight=100`), QWT, cross-modal, JEPA |
| Phan cung | Kaggle **2x T4**, DDP/NCCL, fp16 |
| Budget | 10.000 step (Stage A 2.000 + Stage B 8.000) = **1,58 epoch** |
| Thoi gian | **1.298 s (21,6 phut)** — 129,8 ms/step, 61,6 sample/s |

### Ket qua validation

| step | PSNR | SSIM | `r_image` | `r_acc` | `r_gyro` | ca ba < 1 |
| ---: | ---: | ---: | ---: | ---: | ---: | :--: |
| 1.000 | 34,08 | 0,9136 | 0,6396 | 0,9363 | 1,0047 | khong |
| 2.000 | 35,51 | 0,9273 | 0,5585 | 0,8106 | 1,0031 | khong |
| 4.000 | 36,33 | 0,9386 | 0,5112 | 0,6866 | 1,0023 | khong |
| **6.000** | 36,82 | 0,9426 | 0,4944 | 0,6101 | **0,9992** | **CO** |
| 8.000 | **37,59** | 0,9453 | **0,4880** | 0,5705 | 0,9973 | CO |
| 10.000 | 37,49 | **0,9467** | 0,4925 | **0,5636** | **0,9962** | CO |

**Ket luan cuoi run:**

| Nguon | `r` | Giam MSE |
| --- | ---: | ---: |
| Anh | 0,4925 | **50,8%** |
| Accel | 0,5636 | **43,6%** |
| Gyro | 0,9962 | **0,4%** |

Ca ba ty so **deu < 1** tu step 6.000, nen `best_joint.pt` da duoc luu. Khong co
dau hieu overfit: validation MAE 0,01539 -> 0,01422 trong khi train giam 22,9%.
Gradient norm on dinh (trung vi 0,067, max 0,165).

### Hai du doan truoc do: mot dung, mot sai

**Dung — gyro gan nhu thoai hoa o profile `mild`.** Muc 4.2 du doan dieu nay tu
SNR: gyro 45 dB, va nhieu "gap ghenh" chi bang 0,09x tin hieu sach. Ket qua that:
`r_gyro = 0,9962`, tuc **0,4%**. Khong phai loi kien truc — bai toan gyro o muc
nhieu nay gan nhu khong co gi de khu.

**Sai — toi da danh gia thap nhanh accel.** Gate overfit tren 16 sample cho
accel +9,58% (muc 4), va toi da dung con so do de du doan. Tren du lieu day du
voi 10.000 step, accel dat **+43,6%**. Overfit tren 16 sample la phep thu
*memorize*, **khong du doan duoc** hanh vi khi co 63.000 sample va nhieu step hon.
Khong nen ngoai suy tu gate G3 sang ket qua that.

### Chua ket luan duoc gi tu run nay

- **Mot seed, mot run.** Chua co uncertainty theo trajectory hay multi-seed.
- **Chua co ablation nao.** Khong the quy 50,8% cho JEPA, cho cross-modal fusion
  hay cho QWT — can `kaggle_no_jepa`, `kaggle_no_cross`, `kaggle_haar` cung budget.
- Chua kiem `clean_*_tolerance` (giu du lieu sach) tren bo danh gia rieng.
- Chua chay tren split `test`; moi so lieu tren day la `valid`.
- PSNR dat dinh o step 8.000 (37,59) roi giam nhe con 37,49. Muc giam nam trong
  dao dong giua cac lan validation, chua du de ket luan bat dau overfit.

---

## Phan con lai: cac gate chay tren may local

Ban ghi cac gate da chay tren may local (RTX 4060, CUDA 13.0, torch 2.13.0, FP32).

## 1. Du lieu thuc su dung

| Chi so | Gia tri |
| --- | --- |
| Root | `/home/buidinhkhoi/Datasets/tartanair-v2-jepa` (folder view 7 env, anh 640×640) |
| Trajectory | 76 (48 train / 14 valid / 14 test) |
| Sample sau ghep | 30.503 (20.280 / 5.104 / 5.119) |
| IMU rate | 100,0000 Hz (xac minh tu timestamp + `parameter.yaml`) |
| Camera rate | 10,0000 Hz |
| Cua so | L = 128 hang = **1,27 s** |
| `manifest_hash` | `0c21683bb2a1bf45…` |

Chi tiet: [DATA_AUDIT.md](DATA_AUDIT.md).

## 2. Quy mo model

| Thanh phan | Tham so |
| --- | ---: |
| Encoder anh (CNN2D) | 0,75 M |
| Encoder IMU (CNN1D) | 0,25 M |
| Shared gated MLP fusion | 0,33 M |
| Decoder anh | 0,59 M |
| Decoder IMU | 0,19 M |
| Predictor (2×) | 0,13 M |
| **Tong online** | **2,24 M** |
| Teacher EMA (chi khi train) | +1,00 M |

### 2.1 VRAM va throughput — **da do**, khong phai uoc luong

Forward + backward + teacher EMA, FP32, GPU local:

| Batch | Peak VRAM | s/step | sample/s |
| ---: | ---: | ---: | ---: |
| 4 | 478 MB | 0,050 | 80,7 |
| 8 | **898 MB** | 0,091 | **87,6** |
| 16 | 1.730 MB | 0,196 | 81,6 |
| 32 | 3.407 MB | 0,437 | 73,2 |
| 48 | 5.088 MB | 0,679 | 70,7 |

Model nho (2,24 M) nen throughput **dinh o batch 8** roi giam; tang batch chi ton
VRAM ma khong nhanh hon. Batch 8 chi dung ~0,9 GB — thoai mai tren T4 16 GB.

**Nut that co that la DATA LOADING, khong phai GPU:**

| Nguon | sample/s |
| --- | ---: |
| `__getitem__` mot worker (decode PNG + corruption) | 57,3 |
| DataLoader `num_workers=0` | 61,2 |
| DataLoader `num_workers=2` | 106,2 |
| DataLoader `num_workers=4` | **183,8** |
| GPU (batch 8) | 87,6 |

Voi `num_workers=0`, loader (61/s) **cham hon** GPU (87,6/s) — do la ly do
training do duoc ~0,30 s/step thay vi 0,091 s/step. `configs/kaggle_t4.yaml` da
dat `num_workers=4`. Uoc tinh sau khi het nghen: 10.000 step × batch 8 ≈ 15–20 phut
GPU thuan, nen mot phien Kaggle 12h du cho nhieu run ablation.

Peak VRAM khi inference mot trajectory (batch 4): **148 MB**.

## 3. Ket qua gate

| Gate | Trang thai | Bang chung |
| --- | --- | --- |
| G0 data/pairing/split | **PASS** | 12 test; audit 76 trajectory |
| G1 transform | **PASS** | worst rel error 1,09e-07 (nguong 1e-5); gradcheck fp64 |
| G2 forward/backward/EMA | **PASS** | 153/166 tensor doi sau 20 step, moi gia tri huu han |
| G3 overfit | **MOT PHAN** | anh dat nguong, IMU khong — xem muc 4 |
| G4 pilot held-out + JEPA | **PASS** | xem muc 0: ca ba `r` < 1 tren held-out, 2x T4 |
| G5 save/load/resume/export | **PASS** | 15 test; resume tai epoch boundary trung khop **tung bit** |

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q` → **51 passed**.

### 3.1 Determinism — phai bao cao vi anh huong moi phep so sanh

Do bang cach chay HAI lan cung mot cau hinh lien tuc, 16 step, roi so tham so:

| Thiet bi | max \|diff\| giua hai run giong het nhau |
| --- | ---: |
| GPU (CUDA, cuDNN mac dinh) | **9,556e-05** |
| CPU | **0,000e+00** |

GPU **khong deterministic**: cuDNN tu chon thuat toan convolution khac nhau giua
cac lan chay. Vi vay:

- Resume test cua gate G5 chay tren **CPU**, va cho sai khac **dung 0,0** — resume
  tai epoch boundary trung khop tung bit.
- Moi so sanh ablation tren GPU phai tinh den nhieu nen ~1e-4 o muc tham so.
  Chenh lech metric nho hon muc do **khong** duoc doc la khac biet that.

Checkpoint ghi `resume_exact`, `steps_into_epoch` va `data_replay_point`. Dung
giua epoch duoc danh dau `resume_exact=false` kem canh bao khi load — khong hua
hen exact mid-epoch khi chua luu sampler cursor (dac ta muc 20.1).

Bang chung rieng cho QWT (khong chi round-trip): hai cay triet tieu **82,8%**
nang luong nua pho am, tuc la cap Hilbert that. Hai cay giong nhau — vi du Haar
doi ten — se cho ~50% va lam test fail. Xem [QWT_AUDIT.md](QWT_AUDIT.md).

## 4. Gate G3 — overfit 16 sample co dinh, 800 step, LR co dinh 2e-4

Nguong dat truoc: moi nhom giam >= 30% MSE so voi dau vao nhieu.

| Config | image | accel | gyro |
| --- | ---: | ---: | ---: |
| `main_qwt` (`w_imu=1`, `beta=1`) — theo dac ta | +29,57% | +0,04% | −1,24% |
| `main_qwt_smallbeta` (`beta=0.01`, `w_imu=5`) | +29,62% | +0,32% | −0,82% |
| `main_qwt_balanced` (`w_imu=100`) | **+30,13%** | **+9,58%** | −3,87% |

Nhanh anh hoc on dinh va gan/dat nguong trong ca ba. Nhanh IMU khong dat nguong
trong bat ky cau hinh nao. Hai nguyen nhan doc lap da duoc do bang so.

### 4.1 Mat can bang gradient (sua duoc bang weight)

Voi `w_image = w_imu = 1`, do tren batch that:

| Nhanh | Loss | Grad norm vao decoder |
| --- | ---: | ---: |
| `L_image` | 2,874e-02 | 1,042e-01 |
| `L_imu` | 3,985e-05 | 1,186e-03 |

Lech **~88 lan**. Nhieu IMU sau normalize chi ~0,006 rms, ma `SmoothL1(beta=1.0)`
o do van o **vung quadratic** nen dao ham chi `x/beta ≈ 0,006` thay vi 1,0.

Nang `imu_weight` len 100 da cai thien accel tu +0,04% len **+9,58%**, xac nhan
chan doan. Dac ta muc 13 cho phep chinh weight bang pilot va ghi vao config moi —
da ghi thanh `configs/main_qwt_balanced.yaml`.

### 4.2 Nhiem vu gyro gan nhu thoai hoa o profile `mild` (khong sua duoc bang weight)

SNR do tren 64 sample that, so voi **dao dong that** cua tin hieu trong cua so:

| Kenh | std tin hieu | rms nhieu | nhieu/tin hieu | SNR |
| --- | ---: | ---: | ---: | ---: |
| ax | 1,898 | 0,0886 | 4,67% | 26,6 dB |
| ay | 7,707 | 0,0806 | 1,05% | 39,6 dB |
| az | 2,348 | 0,0797 | 3,40% | 29,4 dB |
| gx | 0,323 | 0,00167 | **0,52%** | **45,7 dB** |
| gy | 0,323 | 0,00322 | 1,00% | 40,0 dB |
| gz | 0,266 | 0,00152 | **0,57%** | **44,9 dB** |

Va quan trong hon — do "gap ghenh" giua hai mau lien tiep:

| Kenh | `d(clean)` rms | `d(noise)` rms | ty le |
| --- | ---: | ---: | ---: |
| ax | 0,4496 | 0,1259 | 0,28× |
| gx | 0,0257 | 0,0023 | **0,09×** |
| gz | 0,0216 | 0,0022 | **0,10×** |

Tin hieu gyro sach **da bien thien manh hon nhieu gap 10 lan** giua cac mau lien
tiep. Nhieu khong phai thanh phan "gap ghenh" ma bo loc co the tach ra. Bat ky
phep lam muot nao cung pha tin hieu that nhieu hon la go nhieu — dung nhu ket qua
gyro am o ca ba config. Day la tinh chat cua **bai toan**, khong phai loi cua
kien truc, va khong sua duoc bang loss weight.

### 4.3 Profile `hard` (Stage C) tao ra phan hoc duoc

Ty le nang luong nhieu la **he thong trong cua so** (do lech trung binh — thanh
phan ma model co the uoc luong va tru di):

| Kenh | `mild` (A/B) | `hard` (C: bias + drift + spike) |
| --- | ---: | ---: |
| ax | 6,4% | 17,5% |
| az | 6,8% | **56,0%** |
| gx | 5,9% | **68,5%** |
| gy | 6,2% | 51,9% |
| gz | 4,8% | 51,1% |

O profile `mild`, ~95% nhieu la trang — gan nhu khong the phuc hoi. O `hard`,
hon mot nua nhieu gyro la **do lech he thong trong cua so**, dieu ma model hoan
toan co the hoc tru. Stage C khong chi "kho hon"; no lam nhiem vu IMU **tro nen
hoc duoc**.

Day khong phai de xuat tang nhieu cho metric dep — dac ta muc 18.1 cam dieu do.
Day la ket luan rang bo tham so nhieu IMU mac dinh o muc 5.3 qua nhe so voi dong
luc hoc that cua TartanAir, va bias/drift moi la che do loi IMU dang de khu.

## 5. Khuyen nghi thu tu chay tren Kaggle

1. `check-transform`, `audit-data`, `build-manifest` tren dataset 14 environment.
2. `overfit` voi **`configs/kaggle_balanced.yaml`** (khong phai `kaggle_t4.yaml`)
   — do la cau hinh duy nhat dat nguong 30% o nhanh anh va co tien bo that o accel.
3. Neu muon nhanh IMU thuc su hoc: `overfit` them voi `configs/stage_c.yaml`
   (bias + drift bat). Bao cao ca hai, khong gop.
4. Stage A → Stage B day du, roi `no_jepa.yaml` lam control **cung budget/seed/split**.
5. `no_cross.yaml` va `haar_baseline.yaml` de tach dong gop cua fusion va cua QWT.

## 6. Chua chay / chua chung minh

- ~~Pilot Stage A va Stage B day du tren held-out (G4)~~ — **da chay**, xem muc 0.
- Bat ky so lieu nao ve loi ich cua **JEPA**. `lambda_J` da dat 0,10 trong run o
  muc 0, nhung **chua co control** `kaggle_no_jepa` cung budget de so sanh, nen
  khong the quy phan cai thien nao cho JEPA.
- Loi ich cua **cross-modal fusion**: chua chay `no_cross.yaml`.
- Loi ich cua **QWT** so voi Haar: chua chay `haar_baseline.yaml`.
- Merge overlap cua ca trajectory va metric theo trajectory (muc 18.3) da co ham
  metric nhung **chua chay tren test**.
- Peak VRAM, multi-seed, uncertainty theo trajectory.
- Doi chieu he so QWT voi implementation thu ba doc lap.
