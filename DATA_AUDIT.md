# DATA_AUDIT.md — audit du lieu truoc khi trien khai

Spec muc 2.4. Moi so lieu duoi day duoc **dem tu du lieu that**, khong lay tu
gia dinh. Lenh tai lap:

```bash
python -m qjepa audit-data --root <DATA_ROOT> --window 128 --out outputs/audit
python -m qjepa build-manifest --root <DATA_ROOT> --out outputs/manifest --window 128
```

Ban ghi nay chay tren `/home/buidinhkhoi/Datasets/tartanair-v2-jepa`
(folder view 7 environment, anh nguon 640x640). Dataset Kaggle `tartanair-v2-256`
co 14 environment va anh da resize 256x256 — phai chay lai audit tren do.

## 1. Quy mo thuc te

| Chi so | Gia tri |
| --- | ---: |
| Environment | 7 |
| Trajectory | 76 |
| Tong hang IMU | 314,910 |
| Tong anh | 31,567 |
| Tong thoi luong | 3,148.3 s (0.87 h) |
| Hang IMU / trajectory | min 640 · trung vi 2,675 · max 16,790 |
| Anh / trajectory | min 65 · trung vi 268 · max 1680 |

> Con so "khoang 1.200 hang IMU moi P" trong yeu cau la **uoc luong**. Thuc te
> trung vi la 2,675 hang/trajectory (min 640, max 16,790).
> Khong hard-code N=1200.

## 2. Tan suat — DA XAC MINH, khong con la phong doan

| Chi so | Do duoc | Nguon doc lap |
| --- | ---: | --- |
| IMU rate (trung vi) | 100.0000 Hz | `imu/parameter.yaml`: `imu_fps: 100` |
| Camera rate (trung vi) | 10.0000 Hz | `imu/parameter.yaml`: `img_fps: 10` |
| Do lech dt tuong doi lon nhat | 1.954e-12 | nguong cau hinh 1e-2 |

Timestamp deu gan nhu tuyet doi, nen **128 hang = 127 x 0.01 s = 1,27 s**
(khong phai 1,28 s). Centered window nhin khoang 0,635 s moi phia quanh anh.

Hai frame lien tiep cach 0,1 s = 10 hang IMU, nen hai cua so ke nhau chong lan
`(128-10)/128 = 92,19%`. **Khong duoc coi cac window chong lan la quan sat doc lap.**

## 3. Tinh toan ven

| Kiem tra | Ket qua |
| --- | --- |
| Timestamp tang nghiem ngat o moi trajectory | True |
| Khong NaN/Inf | True |
| So anh khop `len(cam_time)` | True |
| Trajectory khong du 128 hang | 0 |

## 4. Schema va don vi

| Truong | Gia tri |
| --- | --- |
| File IMU | `imu/acc.{npy,txt}`, `imu/gyro.{npy,txt}` — moi file `[N,3]` |
| Timestamp | `imu/imu_time.*` `[N]`, `imu/cam_time.*` `[M]`, don vi **giay** |
| Kenh canonical | `ax, ay, az, gx, gy, gz` (accel roi gyro) |
| Don vi | accel `m/s^2`, gyro `rad/s` |
| Anh | `image_lcam_front/*_lcam_front.png`, RGB |

**Quan trong — dinh dang tren Kaggle:** cell resize trong `kaggle_session.txt`
chi copy `.png` va `.txt`, bo qua `.npy`. Nen dataset da resize **khong co file
`.npy`**. Loader doc `.npy` neu co, nguoc lai `.txt`; hai dinh dang da duoc
doi chieu va **trung khop tuyet doi** (maxdiff 0.0, ca hai float64).

Dung `acc` (specific force, **CO trong luc** — bien do toi ~28 m/s^2) lam target,
khong dung `acc_nograv`. Khong tu bo trong luc, khong vi phan pose tho de tao
target gia.

## 5. Ghep anh-IMU va split

Chinh sach `centered_offline`: voi moi timestamp anh, chon window 128 hang co
tam gan nhat, yeu cau `t[s] <= t_image <= t[s+127]` va center error <= mot dt.

| Ket qua ghep | So luong |
| --- | ---: |
| Sample duoc chap nhan | **30,503** |
| Loai: anh ngoai vung center hop le (bien trajectory) | 1,064 |
| Loai: center error qua lon | 0 |
| Loai: thieu support | 0 |
| Loai: dt khong deu | 0 |
| Loai: khong du hang | 0 |

1,064 anh bi loai deu nam o **hai dau moi
trajectory** (14 anh/trajectory,
tuong ung 64 hang moi phia). Chung bi **loai**, khong pad va khong lap.

| Split | Sample | Trajectory |
| --- | ---: | ---: |
| train | 20,280 | 48 |
| valid | 5,104 | 14 |
| test | 5,119 | 14 |

Split theo **trajectory**, khong theo frame. `Data_easy` va `Data_hard` cua cung
mot P luon nam cung split. Folder view da co san holdout thi giu nguyen.

`manifest_hash = 0c21683bb2a1bf45a770e30f3b1e74ec...` — luu trong checkpoint; `--resume`
tu choi chay neu hash khac.

## 6. Normalization

Tinh tren **209,040 hang IMU clean cua 48 trajectory train**,
moi timestamp dung mot lan — duyet theo trajectory chu khong theo window, vi cac
window chong lan se dem lai cung mot hang nhieu lan.

| Kenh | mean | std |
| --- | ---: | ---: |
| ax | -0.56093 | 8.23858 |
| ay | +0.12107 | 8.03917 |
| az | -8.41800 | 5.57933 |
| gx | +0.00280 | 0.61516 |
| gy | +0.00029 | 0.68192 |
| gz | -0.00228 | 0.63067 |

`az` co mean -8.418 m/s^2 — dung nhu du kien khi accel con trong luc.

std floor: accel 1e-3 m/s^2, gyro 1e-4 rad/s. Moi std do duoc deu lon hon floor
nhieu bac, nen floor khong kich hoat o dataset nay.

## 7. Canh bao da phat hien — mat can bang gradient hai nhanh

Nhieu IMU sau khi normalize RAT nho so voi nhieu anh:

| Dai luong | Gia tri |
| --- | ---: |
| Nhieu accel (vat ly) | 0.0851 m/s^2 rms |
| Nhieu gyro (vat ly) | 0.00242 rad/s rms |
| Nhieu accel (normalized) | 0.0121 rms |
| Nhieu gyro (normalized) | 0.00366 rms |
| MAE nhieu cua anh | 0.0287 |

Vi `|x| ~ 0.006 << beta = 1.0`, SmoothL1 nam sau trong **vung quadratic**, nen
dao ham chi `x/beta ~ 0.006` thay vi 1.0 nhu vung tuyen tinh. Do duoc tren batch
that voi `w_image = w_imu = 1`:

| Nhanh | Loss | Gradient norm vao decoder |
| --- | ---: | ---: |
| `L_image` | 2.874e-02 | **1.042e-01** |
| `L_imu` | 3.985e-05 | **1.186e-03** |

Lech khoang **88 lan**. Day la nguyen nhan so hoc khien nhanh IMU khong hoc
trong gate G3 voi cau hinh mac dinh. Xem `TRAINING_REPORT.md` muc ket qua va hai
config doi chung `main_qwt_balanced.yaml` / `main_qwt_smallbeta.yaml`.

## 8. Chua lam

- Chua audit dataset Kaggle 14 environment (chi moi 7 environment folder view local).
- Chua kiem tra extrinsics camera-IMU; pipeline hien khong dung pose nen chua can,
  nhung khong duoc coi la da xac minh.
- Khong co barometer trong dataset nay.
