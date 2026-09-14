# Stage A, Stage B va phase v2: train cai gi, va bao nhieu step la du

Tai lieu nay dua tren **code dang chay** va **run that cua ban** (Kaggle 2x T4,
14 environment, 10.000 step). Moi so do deu do bang lenh, khong uoc luong.

> **Ba giai doan, khong phai hai.** Stage A va Stage B la hai giai doan cua mot
> run train. Jacobian regularization la **phase v2 rieng**, chay sau do tu mot
> parent checkpoint — xem muc 8. Run cua ban moi chay A va B.

---

## 1. Y niem: khong phai hai model, la mot model voi hai muc tieu

Sai lam de mac: nghi Stage A train "kien truc nay" roi Stage B train "kien truc
khac". **Khong phai.** Chi co **mot** model duy nhat. Hai stage khac nhau o cho:
co **bao nhieu so hang trong loss**.

```
Stage A:  L = 1.0 * L_image + 100.0 * L_imu
Stage B:  L = 1.0 * L_image + 100.0 * L_imu  +  lambda_J * L_JEPA
                                                 ^^^^^^^^^^^^^^^^^
                                                 chi them so hang nay
```

Loss phuc hoi **khong bao gio tat**. Stage B khong thay the Stage A, no **cong
them** mot muc tieu phu voi trong so toi da 0,10.

---

## 2. Module nao nhan gradient — do bang autograd

Gradient norm vao tung module, do tren mot batch that sau khi head da thoat
zero-init:

| Module | Tham so | Buoc 0 | **Stage A** | **Stage B** |
| --- | ---: | ---: | ---: | ---: |
| `image_encoder` (CNN2D) | 753.024 | 0 | 3,18e-02 | **6,64e-02** |
| `imu_encoder` (CNN1D) | 248.832 | 0 | 6,02e-04 | **3,75e-02** |
| `shared_fusion` | 331.264 | 0 | 9,39e-05 | **2,68e-03** |
| `image_decoder` | 586.416 | 7,21e-01 | 6,25e-01 | 6,25e-01 |
| `imu_decoder` | 192.780 | ~0 | 1,60e-02 | 1,60e-02 |
| `image_predictor` | 66.176 | 0 | **0** | 3,78e-02 |
| `imu_predictor` | 66.176 | 0 | **0** | 3,60e-02 |
| `teacher_*` (EMA) | 1.001.856 | khong co gradient — cap nhat bang EMA | | |

### Doc bang nay

**Cot "Buoc 0" la bay.** Luc khoi tao, head he so duoc **zero-init** nen
`delta = 0` va gradient khong truyen nguoc len encoder/fusion. Neu ban kiem tra
wiring bang mot lan backward duy nhat, ban se ket luan sai rang "encoder khong
bao gio train". Sau 3–10 update, head thoat khoi 0 va moi thu chay binh thuong.

**Stage A train GAN NHU TOAN BO model** — encoder, fusion, ca hai decoder. Chi
hai predictor la khong (chung khong co trong loss). Do la
**779.196 / 2.244.668** tham so duoc cap nhat... *neu* chi dem o buoc 0. Sau vai
buoc, con so thuc la **2.112.316** (moi thu tru predictor).

**Stage B lam gradient vao nhanh IMU tang vot.** Cung mot batch:

| | Stage A → Stage B |
| --- | ---: |
| `imu_encoder` | 6,02e-04 → 3,75e-02 — **gap 62 lan** |
| `shared_fusion` | 9,39e-05 → 2,68e-03 — **gap 29 lan** |
| `image_encoder` | 3,18e-02 → 6,64e-02 — gap 2,1 lan |
| hai decoder | khong doi |

Ly do: loss JEPA di **thang** vao `predictor → fused feature → fusion → encoder`,
**khong** phai qua duong decoder. Nen no cap gradient cho phan bieu dien ma loss
phuc hoi (bi nghen o head) cap rat it.

> Day la do tren **mot batch, mot model chua train ky**, voi `lambda_J = 0,1`.
> No cho thay **huong**, khong phai bang chung rang JEPA cai thien chat luong —
> dieu do can run `kaggle_no_jepa` de doi chung.

---

## 3. Stage A lam gi

**Muc tieu:** hoc phuc hoi truc tiep. Anh ra giong anh sach, IMU ra giong IMU sach.

**Duong di:**

```
anh nhieu → QWT → encoder2D ─┐
                             ├→ fusion → decoder → delta he so → iQWT → anh
IMU nhieu → Haar → encoder1D ┘                                    (so voi anh sach)
```

Teacher **chua ton tai**. `lambda_J = 0`. Predictor co trong model nhung khong
nhan gradient.

**Vi sao phai co Stage A truoc:** teacher duoc tao bang `deepcopy` cua encoder
online **tai luc bat dau Stage B**. Neu bat JEPA ngay tu step 0, teacher se la
ban sao cua mot encoder **ngau nhien** — muc tieu latent vo nghia.

---

## 4. Stage B lam gi

**Tai step 2.000** (mac dinh), ba viec xay ra:

1. `initialize_teacher()`: deepcopy hai encoder online → hai teacher, dat
   `requires_grad=False` va `.eval()`.
2. `lambda_J` bat dau ramp `0,01 → 0,10` trong 1.000 update.
3. Sau **moi** optimizer step thanh cong, teacher duoc cap nhat bang EMA:
   `theta_teacher = m * theta_teacher + (1-m) * theta_online`, voi `m` di tu
   0,99 len 0,999 theo lich cosine.

**Duong JEPA:**

```
anh SACH → QWT → teacher encoder (EMA, no_grad) → target latent
                                                      ↓ so sanh
anh nhieu → ... → fusion → predictor ──────────→ latent du doan
```

Predictor hoc doan: *"dac trung trich tu anh HONG nen giong dac trung trich tu
anh SACH"*. Do la mot tin hieu bo sung cho loss pixel.

**Luu y quan trong:** decoder doc **thang** fused feature, **khong** di qua
predictor. Nhanh JEPA la nhanh ben — no nan encoder qua gradient, khong nam tren
duong tao anh. Vi vay `export` bo duoc teacher va predictor ma output khong doi.

---

## 5. Bao nhieu step la dung — tra loi tu run cua ban

Muc cai thien **moi 1.000 step** (`d/1k`, cang am cang tot):

| step | `r_image` | d/1k | `r_acc` | d/1k | `r_gyro` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 | 0,6396 | | 0,9363 | | 1,0047 |
| 2.000 | 0,5585 | −0,0811 | 0,8106 | −0,1257 | 1,0031 |
| 3.000 | 0,5300 | −0,0285 | 0,7461 | −0,0645 | 1,0077 |
| 4.000 | 0,5112 | −0,0188 | 0,6866 | −0,0595 | 1,0023 |
| 5.000 | 0,5059 | −0,0053 | 0,6406 | −0,0460 | 1,0016 |
| **6.000** | 0,4944 | −0,0115 | 0,6101 | −0,0305 | **0,9992** ← ca ba < 1 |
| 7.000 | 0,4966 | +0,0022 | 0,5837 | −0,0264 | 0,9996 |
| **8.000** | **0,4880** | −0,0086 | 0,5705 | −0,0132 | 0,9973 ← `r_image` tot nhat |
| 9.000 | 0,4929 | +0,0049 | 0,5646 | −0,0059 | 0,9966 |
| 10.000 | 0,4925 | −0,0004 | **0,5636** | −0,0010 | **0,9962** |

### Ket luan tung nhanh

**Anh: bao hoa quanh step 5.000–6.000.** Sau do dao dong ±0,005, co buoc con
**tang** (+0,0022 va +0,0049). Day la nhieu giua cac lan validation, khong phai
cai thien. Train them cho anh **khong con loi**.

**Accel: van giam den cuoi, nhung da gan bao hoa.** Muc giam tut tu −0,126
xuong −0,0010 moi 1.000 step. Con giam, nhung toc do da cham 100 lan.

**Gyro: phang suot.** Khong co step nao cuu duoc — nguyen nhan la SNR, khong
phai ngan sach. Xem `TRAINING_REPORT.md` muc 4.2.

### Vi vay

**10.000 step la hop ly cho pilot nay.** Khong phai vi con so dep, ma vi ca ba
nhanh deu da het cai thien dang ke o do.

---

## 6. Mot cai bay quan trong ve cosine

**Khong** the ket luan "train 20.000 step se tot hon" tu bang tren. Ly do:
learning rate theo lich **cosine neo vao `max_optimizer_steps`**.

```
step 8.000 / 10.000  ->  lr = 2.20e-05
step 10.000 / 10.000 ->  lr = 1.00e-06     (gan nhu dung han)
```

Cuoi run LR gan bang 0, nen model **khong the** hoc them du co chay tiep. Muc
"bao hoa" o step 9.000–10.000 mot phan la do LR da tat, khong hoan toan do model
da hoi tu.

**Muon thu ngan sach lon hon thi phai dat `max_optimizer_steps` lon hon tu dau**,
de cosine trai deu tren ca chang duong:

```yaml
train:
  max_optimizer_steps: 20000
  stage_a_optimizer_steps: 4000      # giu ty le 20% nhu cu
  stage_b_optimizer_steps: 16000
```

**Khong** dung `--resume` roi tang `--steps`: checkpoint mang theo scheduler da
o cuoi cosine, nen phan train them se chay voi LR ~1e-6 va gan nhu khong hoc gi.
Dung `--init` de bat dau mot phase moi voi scheduler moi.

---

## 7. Chon ty le A/B nhu the nao

Mac dinh **2.000 / 8.000** (20% / 80%).

| Neu | Thi |
| --- | --- |
| Stage A chua dat `r < 1` o nhanh nao | **Dung lai chan doan**, dung sang Stage B chi vi du step |
| Muon teacher chat luong hon | Tang Stage A (vi du 3.000), vi teacher la ban sao encoder luc do |
| Chay phase v2 tu parent da co | `stage_a = 0`, vao thang Stage B (config `kaggle_v2_*`) |

Trong run cua ban, o step 2.000 (cuoi Stage A) da co `r_image = 0,5585` va
`r_acc = 0,8106` — deu duoi 1. Teacher duoc tao tu mot encoder **da hoc duoc
that**, khong phai encoder ngau nhien. Do la dieu kien dung de vao Stage B.

---

## 8. Jacobian nam o dau? — **KHONG** phai Stage C

Cau hoi hay gap: "sao khong thay Jacobian trong Stage A hay B?"

**Vi no khong nam trong A hay B.** Jacobian regularization
([themjacobian.md](themjacobian.md)) la mot **phase RIENG**, chay **sau** khi
Stage B da xong, tu mot checkpoint parent.

### Toan bo duong di

```
  Stage A  ──→  Stage B  ──→  [ parent checkpoint ]
  0–2.000       2.000–10.000          │
  phuc hoi      + JEPA                │  cung parent, cung budget, cung seed
                                      ├──→ phase v2 CONTROL    (khong Jacobian)
                                      └──→ phase v2 TREATMENT  (co Jacobian)
                                                  │
                                            moi ben freeze backbone
                                                  ↓
                                        train probe decoder de do
                                        bieu dien con giu bao nhieu thong tin
```

### Vi sao phai tach ra, khong gop vao Stage B

1. **Can parent on dinh.** Config yeu cau `parent_stage_required: stable_stage_b`.
   Bat Jacobian tu dau thi khong co moc nao de so sanh.
2. **Phai co doi chung.** Muon biet Jacobian co loi hay khong thi can hai run
   **giong het nhau tru dung no**. Gop vao Stage B thi khong tach duoc dong gop.
3. **No la thi nghiem, khong phai cong thuc chinh.** Mac dinh **TAT**, va
   schema phai la 2 moi bat duoc — config v1 cu chay khong doi.

### Run cua ban khong co Jacobian

| Config | schema | `encoder_sensitivity` | Stage A/B |
| --- | ---: | ---: | ---: |
| **`kaggle_balanced`** (ban da chay) | 1 | **tat** | 2.000 / 8.000 |
| `kaggle_v2_control` | 2 | tat | 0 / 1.000 |
| `kaggle_v2_treatment` | 2 | **bat** | 0 / 1.000 |

Hai config v2 co `stage_a = 0`: chung **vao thang Stage B** vi parent da co
teacher roi, khong can warm-up lai.

### Cach nhan biet Jacobian dang chay

Log se co them cac cot `sens_*`:

```
[B] step 100/1000 loss ... lam_J 0.100 lam_enc 5.00e-05 ... sens_source=image
    sens_gain_normalized=171.97  sens_gain_raw=100.07
```

Log cua ban **khong co** cac cot do — dung nhu ky vong voi `kaggle_balanced`.

### Jacobian lam gi, mot dong

Do do nhay cua **FI/FU** (dac trung dense **truoc** fusion) khi them mot nhieu
loan nho vao dau vao, roi phat neu no qua nhay:

```
L_enc = mean_batch[  mean((h' - h)^2) / mean((x' - x)^2)  ]
        h = LayerNorm khong affine cua FI hoac FU   (CHI de do)
```

Gradient tu so hang nay **chi** toi encoder cua modality bi perturb — khong toi
fusion, decoder, predictor hay teacher (da kiem bang autograd).

Chi tiet: [ENCODER_SENSITIVITY_REPORT.md](ENCODER_SENSITIVITY_REPORT.md) va
[REPRESENTATION_PROBE_REPORT.md](REPRESENTATION_PROBE_REPORT.md).

**Trang thai:** code da xong va qua gate G0–G4, nhung **chua chay** control/treatment
lan nao, nen **chua co bang chung** Jacobian co loi hay khong.

---

## 9. Tom tat mot bang

| | Stage A | Stage B |
| --- | --- | --- |
| Step mac dinh | 0 – 2.000 | 2.000 – 10.000 |
| Loss | `L_image + 100·L_imu` | cong them `lambda_J · L_JEPA` |
| `lambda_J` | 0 | ramp 0,01 → 0,10 |
| Teacher | chua ton tai | deepcopy luc bat dau, roi EMA moi step |
| Predictor nhan gradient | khong | co |
| Encoder / fusion / decoder | **co** (sau vai buoc dau) | co, gradient IMU manh hon nhieu |
| Muc dich | hoc phuc hoi, tao encoder du tot lam teacher | giu phuc hoi, them tin hieu latent noisy→clean |

Va mot bang nua cho phase v2 (khong phai stage):

| | phase v2 control | phase v2 treatment |
| --- | --- | --- |
| Bat dau tu | cung mot parent Stage B | cung parent do |
| `stage_a / stage_b` | 0 / 1.000 | 0 / 1.000 |
| `encoder_sensitivity` | **tat** | **bat** |
| Loss | `L_image + 100·L_imu + λ_J·L_JEPA` | cong them `λ_enc · L_enc` |
| LR | 2e-5 | 2e-5 |
| Sau do | freeze backbone, train probe decoder | y het, cung init hash |
