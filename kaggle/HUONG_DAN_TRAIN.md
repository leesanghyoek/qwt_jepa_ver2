# Huong dan train qwt_jepa_ver2 tren Kaggle

Viet cho dataset **da xac minh** cua ban:

```
/kaggle/input/datasets/buidinhkhoi/tartanairkhoi/tartanair-v2
14 environment · 166 trajectory (83 easy + 83 hard) · anh 256x256 RGB
IMU 100 Hz · camera 10 Hz · acc.npy CO trong luc · anh khop cam_time
```

Uoc tinh quy mo: ~65.700 anh, loai ~2.300 o bien trajectory → **~63.000 sample**.
Voi batch 8 thi ~6.300 step/epoch, nen **10.000 step ≈ 1,6 epoch**.

**Settings notebook:** Accelerator `GPU T4 x2` · Internet `ON` · Add data:
dataset `tartanairkhoi`.

---

## Cell 0 — Moi truong

```python
import subprocess, torch, sys, os
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip())
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cpu', os.cpu_count())
```

---

## Cell 1 — Lay code

```python
import pathlib, shutil, os, sys

CODE = pathlib.Path('/kaggle/working/code')
if CODE.exists():
    shutil.rmtree(CODE)
!git clone -q https://github.com/leesanghyoek/qwt_jepa_ver2.git {CODE}

os.chdir(CODE)
sys.path.insert(0, str(CODE))
!pip install -q pyyaml
print('cwd =', os.getcwd())
print('qjepa:', (CODE/'qjepa').is_dir())
```

> Phien sau chi can `!git -C /kaggle/working/code pull -q` thay vi clone lai.

---

## Cell 2 — Duong dan va bien dung chung

```python
DATA = '/kaggle/input/datasets/buidinhkhoi/tartanairkhoi/tartanair-v2'
WORK = '/kaggle/working/outputs'
MANIFEST = f'{WORK}/manifest'
CONFIG = 'configs/kaggle_balanced.yaml'      # xem ghi chu ben duoi
!ls {DATA} | head -3
```

**Vi sao `kaggle_balanced.yaml` chu khong phai `kaggle_t4.yaml`:** voi
`imu_weight=1` nhu dac ta goc, gradient vao nhanh IMU nho hon nhanh anh **~88 lan**
va nhanh IMU **khong hoc**. Ban balanced dat `imu_weight=100`. Chi tiet o
`TRAINING_REPORT.md` muc 4.

---

## Cell 3 — Gate G1: transform (chay truoc, ~10 giay)

```python
!python -m qjepa check-transform
```

Phai thay `G1 round-trip: PASS`. Neu FAIL thi **dung lai** — moi so lieu sau do vo nghia.

---

## Cell 4 — Audit du lieu (~1-2 phut)

```python
!python -m qjepa audit-data --root {DATA} --window 128 --out {WORK}/audit
```

Kiem: `imu_rate_hz_median ≈ 100`, `camera_rate_hz_median ≈ 10`,
`all_strictly_increasing: true`, `all_images_match_camera_times: true`,
`trajectories_without_enough_rows: []`.

---

## Cell 5 — Build manifest + normalization (~2-5 phut)

```python
!python -m qjepa build-manifest --root {DATA} --out {MANIFEST} --window 128
```

Ghep anh-IMU theo timestamp, chia split **theo trajectory**, tinh mu/scale tren
rieng train clean. **Luu lai `manifest_hash` in ra** — `--resume` se tu choi neu
hash khac.

> Dataset co san `tartanair-v2-jepa/train/manifest.csv` va `norm_stats.yaml` cua
> project cu. Loader nay **khong dung** chung; no tu build. Khong sao.

---

## Cell 6 — Gate G2 + do toc do that (~2 phut)

```python
import time
t0 = time.time()
!python -m qjepa smoke-train --config {CONFIG} --manifest {MANIFEST} \
    --steps 50 --output-dir {WORK}/smoke
dt = time.time() - t0
print(f'\n~{dt/50:.3f} s/step  ->  10.000 step ~ {dt/50*10000/3600:.1f} gio')
```

Phai thay `G2 finite: True` va phan lon tensor tham so da doi.

**Dung con so `s/step` nay** de chon `--steps` o Cell 8 sao cho ket thuc truoc
gioi han 12h cua phien.

---

## Cell 7 — Gate G3: overfit (~5-10 phut)

```python
!python -m qjepa overfit --config {CONFIG} --manifest {MANIFEST} \
    --samples 16 --steps 800 --output-dir {WORK}/overfit
```

Muc tieu: **ca ba** nhom giam >= 30% MSE so voi input nhieu.

Tren may local voi 7 env, `main_qwt_balanced` dat: anh **+30,1%**, accel **+9,6%**,
gyro **−3,9%**. Nhanh anh dat nguong, nhanh IMU **khong**. Neu ket qua cua ban
tuong tu thi **do la da biet truoc**, khong phai loi cai dat — nguyen nhan o
`TRAINING_REPORT.md` muc 4.2 (nhieu gyro chi bang 0,5% dao dong that).

Neu muon nhanh IMU thuc su hoc, chay them mot lan voi `configs/kaggle_stage_c.yaml`
(bat bias + drift) va **bao cao rieng**, khong gop voi run mild.

---

## Cell 8 — TRAIN (phan chinh)

**Da GPU tu dong.** `--gpus auto` (mac dinh) dung het GPU dang co: T4 x2 → hai
process DDP, may mot GPU → chay thang. Ep mot GPU bang `--gpus 1`.

`batch_size` la batch **moi rank**. De hai cau hinh so sanh duoc, config Kaggle
dat `batch 4 × accum 2`, va khi chay 2 GPU thi accum tu dong giam con 1:

| | batch/rank | accum | rank | **effective batch** |
| --- | ---: | ---: | ---: | ---: |
| 1 GPU | 4 | 2 | 1 | **8** |
| 2 GPU | 4 | 1 | 2 | **8** |

Nho vay run 1 GPU va run 2 GPU **so sanh truc tiep duoc**. Neu ban doi
`gradient_accumulation` thanh so le, code se **canh bao** rang effective batch
doi va hai run khong con so sanh duoc.

```python
import pathlib

PREV = next(iter(sorted(pathlib.Path('/kaggle/input').glob('*/outputs/main/last.pt'))), None)
resume = f'--resume {PREV}' if PREV else ''
STEPS = 10000        # GIAM xuong neu Cell 6 cho thay khong kip 12h

print('resume tu:', PREV or '(phien dau, train tu dau)')
!python -m qjepa train --config {CONFIG} --manifest {MANIFEST} \
    --steps {STEPS} --log-every 100 --gpus auto \
    --output-dir {WORK}/main {resume}
```

Voi 2 GPU se thay dong:

```
[dist] phat hien 2 GPU -> chay lai bang torchrun: ...
[dist] gradient_accumulation 2 -> 1 de giu effective batch khong doi tren 2 GPU
[dist] 2 GPU | batch/rank 4 | accum 1 | effective batch 8
```

Lich hai stage (tu dong theo config):

| Stage | Step | Noi dung |
| --- | --- | --- |
| A | 0 – 2.000 | chi phuc hoi, `lambda_J = 0` |
| B | 2.000 – 10.000 | khoi tao teacher, JEPA ramp 0,01 → 0,10 |

Doc log: `r_image`, `r_acc`, `r_gyro` **< 1** nghia la tot hon input nhieu.
`all<1 True` la dieu kien de luu `best_joint.pt`.

---

## Cell 9 — Danh gia va export

```python
!python -m qjepa evaluate --config {CONFIG} --manifest {MANIFEST} \
    --checkpoint {WORK}/main/last.pt --split valid
!python -m qjepa export --config {CONFIG} --manifest {MANIFEST} \
    --checkpoint {WORK}/main/last.pt
```

---

## Cell 10 — Don dep truoc khi Save Version

```python
import pathlib, shutil
shutil.rmtree('/kaggle/working/code', ignore_errors=True)
shutil.rmtree(f'{WORK}/smoke', ignore_errors=True)
for p in sorted(pathlib.Path(WORK).rglob('*')):
    if p.is_file() and p.stat().st_size > 1e6:
        print(f'{p.stat().st_size/2**20:8.1f} MB  {p.relative_to("/kaggle/working")}')
```

Gioi han commit cua `/kaggle/working` la 20 GB. Checkpoint ~27 MB moi file nen
thoai mai.

---

## Chay tiep phien sau

1. **Save Version → Save & Run All (Commit)**.
2. Tu output tao **New Dataset** (hoac dung truc tiep output cua notebook).
3. Phien sau: **Add data** them output do. Cell 8 tu tim `last.pt` va them `--resume`.

`--resume` khoi phuc model, optimizer, scheduler, teacher EMA, step, stage va RNG.
No **tu choi** chay neu `manifest_hash` hoac image transform khac checkpoint — de
khong vo tinh ghep hai run khac cau hinh vao mot duong metric.

Checkpoint luu **giua epoch** duoc danh dau `resume_exact=false` kem canh bao khi
load: resume se phat lai tu dau epoch, khong phai chinh xac tung bit. Chi checkpoint
o **ranh gioi epoch** moi resume chinh xac.

---

## Sau khi co Stage B: cac run so sanh

Moi run duoi day dung **cung parent, cung budget, cung seed, cung split**:

```python
PARENT = f'{WORK}/main/last.pt'

# Dong gop cua JEPA          (chi khac: stage_b = 0, lambda_J luon 0)
!python -m qjepa train --config configs/kaggle_no_jepa.yaml --manifest {MANIFEST} \
    --init {PARENT}

# Dong gop cua cross-modal fusion   (chi khac: cross_modal = false)
!python -m qjepa train --config configs/kaggle_no_cross.yaml --manifest {MANIFEST} \
    --init {PARENT}

# Dong gop cua QWT                  (chi khac: transform.image = Haar 12 kenh)
!python -m qjepa train --config configs/kaggle_haar.yaml --manifest {MANIFEST} \
    --init {PARENT}
```

> **Dung cac config `configs/kaggle_*.yaml`, khong dung `no_jepa.yaml` /
> `no_cross.yaml` / `haar_baseline.yaml`.** Ba file khong co tien to `kaggle_`
> la ban LOCAL: batch 4, `num_workers=0`, fp32 va **`imu_weight=1`**. Chay chung
> canh run chinh thi khac nhau ca batch, precision lan loss weight — khong con la
> ablation nua. Ban `kaggle_*` chi khac **dung mot bien** dang duoc ablate.

Jacobian regularization (themjacobian v2) — control va treatment tu **cung** parent:

```python
!python -m qjepa train --config configs/kaggle_v2_control.yaml   --manifest {MANIFEST} \
    --init {PARENT}
!python -m qjepa train --config configs/kaggle_v2_treatment.yaml --manifest {MANIFEST} \
    --init {PARENT}

# Probe decoder tren backbone dong bang, CUNG init hash cho ca hai
!python -m qjepa train-probe --config configs/kaggle_v2_probe.yaml --manifest {MANIFEST} \
    --backbone {WORK}/v2_control/last.pt   --output-dir {WORK}/probe_control
!python -m qjepa train-probe --config configs/kaggle_v2_probe.yaml --manifest {MANIFEST} \
    --backbone {WORK}/v2_treatment/last.pt --output-dir {WORK}/probe_treatment
```

---

## Loi hay gap

| Trieu chung | Nguyen nhan |
| --- | --- |
| `manifest_hash khac` khi `--resume` | Manifest build lai khac lan truoc. Dung `--init` cho thi nghiem moi, hoac giu nguyen manifest cu |
| `checkpoint dung transform ... config dung ...` | Dang ghep run QWT voi run Haar. Chay rieng |
| Loss `NaN` | Bao loi ro va dung; kiem `precision` va gradient clip |
| RAM het khi build-manifest | Giam so trajectory, hoac build tung split |
| `MKL_THREADING_LAYER=INTEL is incompatible` | Da xu ly san: code ep `GNU` cho process con cua torchrun |
| Chi thay 1 GPU du co 2 | Kiem `torch.cuda.device_count()` o Cell 0; Kaggle phai chon `GPU T4 x2` |
| `perturbation bi mat` (chi khi bat v2) | Bien do IMU qua lon so voi epsilon o FP32 |

---

## Ba dieu **chua** duoc chung minh

Truoc khi ket luan bat cu dieu gi:

1. **Loi ich cua JEPA** — `lambda_J` chua bao gio > 0 trong mot run co ket qua.
2. **Loi ich cua cross-modal fusion** va **cua QWT so voi Haar** — chua chay ablation.
3. **Loi ich cua Jacobian regularization** — chua co parent Stage B hop le.

Cac run o muc tren la de **tao** bang chung, khong phai xac nhan ket luan da co.
