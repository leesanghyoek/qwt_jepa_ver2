# Huong dan train qwt_jepa_ver2 tren Kaggle

Viet cho dataset **da xac minh** cua ban:

```
/kaggle/input/datasets/buidinhkhoi/tartanairkhoi/tartanair-v2
14 environment · 166 trajectory (83 easy + 83 hard) · anh 256x256 RGB
IMU 100 Hz · camera 10 Hz · acc.npy CO trong luc · anh khop cam_time
```

Uoc tinh quy mo: ~65.700 anh, loai ~2.300 o bien trajectory → **~63.000 sample**.
Voi **effective batch 8** thi ~6.300 step/epoch, nen **10.000 step ≈ 1,6 epoch**.
Con so nay khong doi du chay 1 hay 2 GPU — xem muc "Da GPU" ben duoi.

**Settings notebook:** Accelerator `GPU T4 x2` · Internet `ON` · Add data:
dataset `tartanairkhoi`.

---

## Cell 0 — Moi truong

```python
import subprocess, torch, sys, os
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip())
n = torch.cuda.device_count()
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cpu', os.cpu_count())
print(f'>>> SO GPU = {n}', '-> se dung DDP' if n > 1 else '-> chay mot process')
```

Neu `SO GPU = 1` ma ban muon hai: **Settings → Accelerator → GPU T4 x2**, roi
restart session. Code chay dung voi ca hai truong hop, chi khac toc do.

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
!git -C {CODE} log --oneline -1
!python -m qjepa --help | head -3
```

**Hai dong cuoi quan trong.** Chung cho biet ban dang chay ban code nao va co
nhung lenh gi. Neu mot lenh trong huong dan bao `invalid choice`, nghia la clone
cua ban CU hon huong dan — cap nhat bang:

```python
!git -C /kaggle/working/code pull -q && git -C /kaggle/working/code log --oneline -1
```

Dang giua phien thi `pull` du; khong can chay lai Cell 1 (no xoa va clone lai tu dau).

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
import time, torch
t0 = time.time()
!python -m qjepa smoke-train --config {CONFIG} --manifest {MANIFEST} \
    --steps 50 --output-dir {WORK}/smoke
dt = time.time() - t0
print(f'\n~{dt/50:.3f} s/step  ->  10.000 step ~ {dt/50*10000/3600:.1f} gio')
print(f'(do tren 1 process; voi {torch.cuda.device_count()} GPU thi Cell 8 nhanh hon)')
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

### Da GPU — tu dong, khong phai sua gi

`--gpus auto` la **mac dinh**:

| May co | Hanh vi |
| --- | --- |
| 1 GPU | chay mot process, khong khoi tao DDP |
| 2 GPU (T4 x2) | tu khoi dong lai bang `torchrun`, moi GPU mot process, dong bo bang DDP |

Ep mot GPU: `--gpus 1`. Ep hai: `--gpus 2` (bao loi neu may khong du).

**Effective batch giu nguyen** du chay may GPU. `batch_size` trong config la batch
**moi rank**, va `gradient_accumulation` tu chia cho so GPU:

| | batch/rank | × accum | × rank | = **effective batch** |
| --- | ---: | ---: | ---: | ---: |
| 1 GPU | 4 | 2 | 1 | **8** |
| 2 GPU | 4 | 1 | 2 | **8** |

Nho vay run 1 GPU va run 2 GPU cho cung effective batch, cung lich learning rate
va cung so step. Neu ban doi `gradient_accumulation` thanh so **khong chia het**
cho so GPU, code in `CANH BAO` rang effective batch se doi va hai run khong con
so sanh truc tiep duoc.

> **Ablation phai dung CUNG mot `--gpus`.** Effective batch giong nhau, nhung thu
> tu du lieu thi khac (`DistributedSampler` khac shuffle thuong). Chay control
> tren 1 GPU roi treatment tren 2 GPU la **khong cong bang**.

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

Voi 2 GPU, **ba dong dau** phai la:

```
[dist] phat hien 2 GPU -> chay lai bang torchrun:
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node=2 --standalone ...
  [dist] gradient_accumulation 2 -> 1 de giu effective batch khong doi tren 2 GPU
  [dist] 2 GPU | batch/rank 4 | accum 1 | effective batch 8
```

Dong dau do CLI in truoc khi khoi dong lai; hai dong sau do rank 0 in ben trong
torchrun.

Khong thay ba dong nay nghia la dang chay **mot** GPU — kiem lai Cell 0.

Log train sau do chi do **rank 0** in ra, nen nhin giong het run mot GPU. Do la
dung: hai rank duoc dong bo gradient moi optimizer step nen tham so luon giong
nhau, va chi rank 0 ghi checkpoint de hai process khong dam nhau khi ghi file.

Kiem nhanh 2 GPU co that su duoc dung khong (chay trong luc train o cell khac):

```python
!nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Ca hai dong phai co `utilization` khac 0.

Lich hai stage (tu dong theo config):

| Stage | Step | Noi dung |
| --- | --- | --- |
| A | 0 – 2.000 | chi phuc hoi, `lambda_J = 0` |
| B | 2.000 – 10.000 | khoi tao teacher, JEPA ramp 0,01 → 0,10 |

Doc log: `r_image`, `r_acc`, `r_gyro` **< 1** nghia la tot hon input nhieu.
`all<1 True` la dieu kien de luu `best_joint.pt`.

---

## Cell 8b — Xem model co dang hoc dung khong

```python
from IPython.display import Image, display

!python -m qjepa plot --run-dir {WORK}/main \
    --config {CONFIG} --manifest {MANIFEST} \
    --checkpoint {WORK}/main/last.pt --num-samples 4

display(Image(f'{WORK}/main/training_curves.png'))
display(Image(f'{WORK}/main/samples.png'))
```

Chay duoc **giua chung** khi train con dang chay o cell khac — no chi doc
`train_log.jsonl` va `validation.jsonl`.

### Doc do thi nao truoc

Do thi quan trong nhat la **`r = MSE(output) / MSE(nhieu)`** (o giua hang tren):

| | Y nghia |
| --- | --- |
| `r < 1` | model **co ich** — sai so nho hon khong lam gi |
| `r = 1` | model tra lai gan dung dau vao |
| `r > 1` | model dang **lam hong** du lieu |

Loss giam **khong** du de ket luan. Model co the giam loss ma van lam xau du
lieu so voi dau vao; chi `r` tra loi duoc cau hoi do.

### Bang KIEM TRA (goc duoi phai)

Tu dong danh gia va in ra `OK` / `CANH BAO` / `XAU` / `CHUA DU`:

| Muc | Y nghia khi XAU |
| --- | --- |
| Loss phuc hoi giam | khong hoc duoc; kiem LR, gradient, target |
| Gradient norm | bung no hoac triet tieu |
| Anh/Accel/Gyro tot hon identity | `r >= 1` — chua co ich |
| Ca ba nguon cung tot | chua luu duoc `best_joint.pt` |
| Anh trong [0,1] | nhieu pixel tran ra ngoai truoc khi clamp |
| Khong overfit | validation xau di trong khi train tot len |

> Muc "Loss phuc hoi giam" **khong tinh** so hang JEPA. Khi vao Stage B,
> `loss_total` nhay len la **binh thuong** vi cong them `lambda_J * L_JEPA` —
> khong phai dau hieu hong.

### Panel anh (`samples.png`)

Moi hang mot sample co dinh: `SACH | NHIEU | PHUC HOI | |loi| x5 | sai so accel | sai so gyro`.

Hai cot IMU ve **sai so**, khong ve tin hieu: nhieu IMU nho hon dao dong that
hang tram lan nen ba duong tin hieu chong khit nhau, nhin khong ra gi. Duong xanh
la (sau phuc hoi) **thap hon** duong cam (dau vao) nghia la model dang go bot nhieu.

So sanh hai run:

```python
!python -m qjepa plot --run-dir {WORK}/v2_control   --title "control"
!python -m qjepa plot --run-dir {WORK}/v2_treatment --title "treatment"
```

---

## Cell 8c — Lay anh ra khoi Kaggle

Anh **da nam san** trong `/kaggle/working/...`. Van de chi la tai ve. Bon cach,
xep theo do tin cay:

### Cach 1 — Panel Output ben phai (tin cay nhat)

Thanh ben phai notebook → tab **Output** (hoac **Data → Output**) → duyet toi
`outputs/main/` → bam vao file → nut **Download**. Khong can chay gi them.

Neu khong thay file vua tao, bam nut **refresh** o goc panel do.

### Cach 2 — Link tai ngay trong cell

```python
from IPython.display import FileLink, display
import pathlib

for p in sorted(pathlib.Path(WORK).rglob('*.png')):
    print(f'{p.stat().st_size/1024:7.0f} KB  {p}')
    display(FileLink(str(p)))
```

Bam vao link la tai ve may. Chi hoat dong voi file **trong `/kaggle/working`**.

### Cach 3 — Gop tat ca vao mot file zip

Tien khi co nhieu anh:

```python
import shutil, pathlib
from IPython.display import FileLink, display

shutil.make_archive('/kaggle/working/dothi', 'zip', WORK, '.')
z = pathlib.Path('/kaggle/working/dothi.zip')
print(f'{z.stat().st_size/2**20:.1f} MB')
display(FileLink(str(z)))
```

> Zip ca `WORK` se gom luon checkpoint `.pt` (~27 MB moi file). Muon chi lay anh:
> ```python
> import zipfile, pathlib
> with zipfile.ZipFile('/kaggle/working/dothi.zip', 'w') as z:
>     for p in pathlib.Path(WORK).rglob('*.png'):
>         z.write(p, p.relative_to(WORK))
> ```

### Cach 4 — Save Version roi tai tu Output

**Save Version → Save & Run All (Commit)**. Xong thi vao trang notebook → tab
**Output** → tai tung file hoac tai ca thu muc. Cach nay giu duoc ban ghi lau dai
va la thu ban dung lam input cho phien sau.

### Neu chi muon NHIN ro hon, khong can tai

Do thi mac dinh 110 dpi. Tang len de doc duoc chu nho:

```python
!python -m qjepa plot --run-dir {WORK}/main --dpi 180
```

Hoac phong to ngay trong cell:

```python
from IPython.display import Image, display
display(Image(f'{WORK}/main/training_curves.png', width=1600))
```

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
    --init {PARENT} --gpus auto

# Dong gop cua cross-modal fusion   (chi khac: cross_modal = false)
!python -m qjepa train --config configs/kaggle_no_cross.yaml --manifest {MANIFEST} \
    --init {PARENT} --gpus auto

# Dong gop cua QWT                  (chi khac: transform.image = Haar 12 kenh)
!python -m qjepa train --config configs/kaggle_haar.yaml --manifest {MANIFEST} \
    --init {PARENT} --gpus auto
```

> **Dung cac config `configs/kaggle_*.yaml`, khong dung `no_jepa.yaml` /
> `no_cross.yaml` / `haar_baseline.yaml`.** Ba file khong co tien to `kaggle_`
> la ban LOCAL: batch 4, `num_workers=0`, fp32 va **`imu_weight=1`**. Chay chung
> canh run chinh thi khac nhau ca batch, precision lan loss weight — khong con la
> ablation nua. Ban `kaggle_*` chi khac **dung mot bien** dang duoc ablate.

Jacobian regularization (themjacobian v2) — control va treatment tu **cung** parent:

```python
!python -m qjepa train --config configs/kaggle_v2_control.yaml   --manifest {MANIFEST} \
    --init {PARENT} --gpus auto
!python -m qjepa train --config configs/kaggle_v2_treatment.yaml --manifest {MANIFEST} \
    --init {PARENT} --gpus auto

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
| `invalid choice: 'plot'` (hoac lenh khac) | Clone cu hon huong dan — `git -C /kaggle/working/code pull` |
| `khong thay .../train_log.jsonl` | Chua chay Cell 8, hoac `--run-dir` sai; lenh se liet ke cac run dang co |
| Chi thay 1 GPU du co 2 | Kiem `torch.cuda.device_count()` o Cell 0; Kaggle phai chon `GPU T4 x2` |
| `perturbation bi mat` (chi khi bat v2) | Bien do IMU qua lon so voi epsilon o FP32 |

---

## Ba dieu **chua** duoc chung minh

Truoc khi ket luan bat cu dieu gi:

1. **Loi ich cua JEPA** — `lambda_J` chua bao gio > 0 trong mot run co ket qua.
2. **Loi ich cua cross-modal fusion** va **cua QWT so voi Haar** — chua chay ablation.
3. **Loi ich cua Jacobian regularization** — chua co parent Stage B hop le.

Cac run o muc tren la de **tao** bang chung, khong phai xac nhan ket luan da co.
