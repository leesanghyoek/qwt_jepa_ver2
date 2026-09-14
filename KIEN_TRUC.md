# KIEN_TRUC.md — kien truc QWT–JEPA v2, chi tiet

Tai lieu nay mo ta **code dang co** trong `qjepa/`, khong phai danh sach tinh nang
du kien. Moi shape va so tham so trong day deu duoc **do bang forward hook tren
model that**, khong phai tinh tay.

Dac ta goc: [kientruc2.md](kientruc2.md) (v3.0). Ban ghi kiem chung:
[QWT_AUDIT.md](QWT_AUDIT.md), [DATA_AUDIT.md](DATA_AUDIT.md),
[TRAINING_REPORT.md](TRAINING_REPORT.md).

---

## 1. Bai toan

Moi sample gom **mot frame RGB 256×256** va **mot cua so IMU 128 hang × 6 kenh**
quanh timestamp cua frame do. Ca hai deu bi lam hong, model phuc hoi ca hai.

```
I_bad = C_image(I_clean)          # lam mo + nhieu, doc lap
U_bad = C_imu(U_clean)            # nhieu IMU, doc lap
I_hat, U_hat = model(I_bad, U_bad, t_image, t_imu)
```

Hai nguon hong **doc lap nhau**: blur khong sinh tu gyro, spike IMU khong tao
rung camera. Kiem tra bang test tuong quan tham so.

**Khong phai generative model.** Model du doan mot **so hang hieu chinh tren he
so wavelet** roi bien doi nguoc. Khong mam nhieu, khong lay mau, khong diffusion.

**Khong phai I-JEPA goc.** `mask_ratio = 0`, teacher an **anh sach that**. Day la
*noisy-to-clean latent prediction* — hybrid co giam sat, khong phai self-supervised.

---

## 2. So do luong

```
                    ┌─ QWT 2D ──→ [B,48,128,128] ──→ CNN2D encoder ──→ FI [B,128,16,16]
anh nhieu [B,3,256,256]                                     │ skips S0,S1,S2        │
                                                            │                       ▼
                                                            │              shared gated MLP
                                                            │                  fusion
                                                            │                       │
IMU nhieu [B,6,128] ─ Haar 1D ─→ [B,12,64] ─→ CNN1D encoder ─→ FU [B,128,8] ────────┤
                                                    │ skips V0,V1,V2                 │
                                                    │                                │
                     ┌──────────────────────────────┴────────────────────────────────┤
                     ▼                                                               ▼
              decoder IMU ──→ delta_CU [B,12,64]              decoder anh ──→ delta_CI [B,48,128,128]
                     │                                                               │
       CU_bad + delta_CU ──→ iHaar ──→ IMU phuc hoi [B,6,128]      CI_bad + delta_CI ──→ iQWT ──→ anh [B,3,256,256]

     ── nhanh phu, CHI khi train ──────────────────────────────────────────────
     zi ──→ image_predictor ──┐
                              ├──→ loss JEPA ←── teacher EMA (an du lieu SACH)
     zu ──→ imu_predictor ────┘
```

**Diem quan trong:** decoder doc **thang** `zi`/`zu`. Predictor la nhanh ben.
Da xac minh bang AST: `forward()` khong goi teacher va khong goi predictor. Xoa
han predictor roi forward lai cho ket qua lech **0.0**.

---

## 3. Hop dong input/output

| Tensor | Shape | Y nghia |
| --- | --- | --- |
| `image_bad` | `[B,3,256,256]` | RGB float32 trong `[0,1]` |
| `imu_bad_phys` | `[B,6,128]` | don vi SI: accel m/s², gyro rad/s |
| `image_time` | `[B]` | timestamp anh, giay |
| `imu_times` | `[B,128]` | timestamp tung mau IMU, giay |
| → `image_hat` | `[B,3,256,256]` | chinh frame dau vao, da phuc hoi |
| → `imu_hat_phys` | `[B,6,128]` | don vi SI |

Thu tu 6 kenh: `ax, ay, az, gx, gy, gz`. **Tren dia** IMU luu `[128,6]`
(time-major); `collate()` transpose sang `[B,6,128]` cho model. Timestamp luu
rieng, **khong** nhet vao kenh do thu 7.

Duong online chi nhan du lieu **nhieu**. Anh sach / IMU sach / pose GT / severity
that khong bao gio di vao forward.

---

## 4. QWT 2D dual-tree — nhanh anh

File: `qjepa/transforms/qwt.py`. Backend `qwt_dualtree_db4`, 1 level,
boundary `periodic`.

### 4.1 Cau truc

Hai cay filter bank lech nhau **mot mau dau vao**. Sau downsample 2, mot mau dau
vao = nua mau tren luoi he so — dung do tre nua mau ma cap Hilbert doi hoi.

Tich separable cua hai cay theo hai truc cho **bon thanh phan quaternion**:

| component | (cay theo x, cay theo y) | Y nghia |
| --- | --- | --- |
| `real` | (a, a) | khong Hilbert |
| `i` | (b, a) | Hilbert theo truc x |
| `j` | (a, b) | Hilbert theo truc y |
| `k` | (b, b) | Hilbert ca hai truc |

Moi to hop cay cho 4 subband: `(approx, detail_1, detail_2, detail_3) = (LL, LH, HL, HH)`.

```
[B,3,256,256] → [B, 3, 4, 4, 128, 128] → [B, 48, 128, 128]
                     RGB band comp           channel = ((rgb*4)+band)*4 + component
```

48 = 3 RGB × 4 band × 4 component. **RGB duoc bien doi RIENG tung kenh** — khong
dong goi RGB thanh mot quaternion mau.

### 4.2 Bo loc

Daubechies **db4** (8 tap), truc chuan, `sum(h0)=√2`. `h1[n] = (-1)ⁿ·h0[F-1-n]`.

Do bang so: `‖h0‖² = 0.999999999999`, `⟨h0, h0 dich 2⟩ = -1.15e-14`, `⟨h0,h1⟩ = 6.9e-18`.

### 4.3 Tinh nguoc

Moi to hop cay la bien doi **truc giao**, nen synthesis = adjoint cua analysis va
tai tao chinh xac. Bien doi tong hop **du thua 4 lan**; synthesis lay trung binh
bon tai tao.

- Round-trip: relative error **1.09e-07** (nguong spec 1e-5).
- `gradcheck` float64 tren synthesis: PASS.
- Vi du thua, `analysis(synthesis(c)) == c` **khong** dung voi `c` tuy y — dung
  nhu spec §6.2 ghi.

### 4.4 Bang chung day la QWT that

Round-trip **khong** chung minh la QWT — Haar doi ten cung round-trip hoan hao.
Bang chung rieng: dung wavelet 1D tu moi cay, lap `ψ_a + i·ψ_b`, do nang luong
nua pho am.

| Cau truc | Nang luong nua pho DUONG |
| --- | --- |
| Cap Hilbert ly tuong | 100% |
| **Implementation nay** | **82.8%** |
| Hai cay giong nhau (Haar doi ten) | ~50% |

Test `test_qwt_trees_form_hilbert_pair` dat nguong 75% — thay backend bang hai
cay giong nhau se FAIL.

---

## 5. Haar 1D — nhanh IMU

File: `qjepa/transforms/haar.py`. Day la **DWT 1D, khong phai QWT**.

```
A[n] = (u[2n] + u[2n+1]) / √2          u[2n]   = (A[n] + D[n]) / √2
D[n] = (u[2n] - u[2n+1]) / √2          u[2n+1] = (A[n] - D[n]) / √2
```

Pack `[A_ax..A_gz, D_ax..D_gz]`: `[B,6,128] → [B,12,64]`. Round-trip 6.5e-08.

Khong ap QWT 2D len ma tran `thoi gian × sau truc` — truc cam bien khong phai
mot truc khong gian lien tuc.

---

## 6. Hai encoder CNN

File: `qjepa/models/encoders.py`, blocks o `blocks.py`.

```
ConvBlock_d(cin,cout,stride):  Conv{d}d(k=3,stride,pad=1,bias=False) → GroupNorm(8) → SiLU
ResBlock_d(c):                 y = Conv→GN→SiLU→Conv→GN ;  return SiLU(x + y)
Stage = ConvBlock + ResBlock
```

Khong BatchNorm, dropout = 0. Channels `32/64/96/128` deu chia het 8.

### 6.1 Encoder anh (CNN2D) — **do duoc**

| Tang | Thao tac | Output |
| --- | --- | --- |
| input | QWT | `[B,48,128,128]` |
| S0 | Stage(48→32, stride 1) | `[B,32,128,128]` |
| S1 | Stage(32→64, stride 2) | `[B,64,64,64]` |
| S2 | Stage(64→96, stride 2) | `[B,96,32,32]` |
| S3 | Stage(96→128, stride 2) | **`FI [B,128,16,16]`** |

`FI = S3`; skips = `S0, S1, S2`. **Khong temporal mixer** — chi so 3 la RGB, khong phai thoi gian.

### 6.2 Encoder IMU (CNN1D) — **do duoc**

| Tang | Thao tac | Output |
| --- | --- | --- |
| input | Haar 1D | `[B,12,64]` |
| V0 | Stage(12→32, stride 1) | `[B,32,64]` |
| V1 | Stage(32→64, stride 2) | `[B,64,32]` |
| V2 | Stage(64→96, stride 2) | `[B,96,16]` |
| V3 | Stage(96→128, stride 2) | **`FU [B,128,8]`** |

`FU = V3`; skips = `V0, V1, V2`.

> `128` la so **channels**, `8` la chieu **thoi gian** con lai. Model **khong**
> chi phuc hoi 8 hang: decoder upsample ve 64 vi tri he so roi iHaar tra lai
> **du 128 hang**.

---

## 7. Shared gated MLP fusion

File: `qjepa/models/fusion.py`. Khong Transformer, khong attention.

### 7.1 Tao summary rieng

```
gI = LayerNorm₁₂₈( mean_spatial(FI) )                    # [B,128]
u4 = flatten( AdaptiveAvgPool1d(4)(FU) )                  # [B,512]
gU = LayerNorm₁₂₈( Linear(512→128)(u4) )                  # [B,128]
m  = metadata thoi gian                                   # [B,3]
v  = concat(gI, gU, m)                                    # [B,259]
s  = LayerNorm₁₂₈( Linear(256→128)( SiLU( Linear(259→256)(v) ) ) )   # [B,128]
```

**259 = 128 + 128 + 3.**

### 7.2 Metadata thoi gian `m` (3 chieu)

```
T = imu_times[-1] - imu_times[0]                          # phai > 0
m = [ (t_image - (imu_times[0]+imu_times[-1])/2) / T ,
      log(T / 1 giay) ,
      log(median(diff(imu_times)) / 0.01 giay) ]
```

Timestamp loi bi **reject**, khong giau bang epsilon.

### 7.3 Dieu tiet dense feature bang cong

```
gateI = sigmoid( Linear_I(259→128)(v) )                   # [B,128]
gateU = sigmoid( Linear_U(259→128)(v) )

dI = Conv2d₁ₓ₁(128→128)( SiLU( Conv2d₁ₓ₁(256→128)( concat(FI, broadcast(s)) ) ) )
dU = Conv1d₁ₓ₁(128→128)( SiLU( Conv1d₁ₓ₁(256→128)( concat(FU, broadcast(s)) ) ) )

FI_fused = FI + gateI[...,None,None] · dI                 # [B,128,16,16]
FU_fused = FU + gateU[...,None]      · dU                 # [B,128,8]
```

Gate `Linear` khoi tao **weight = 0, bias = -2** → `sigmoid(-2) = 0.1192`. Cong
mo he he luc dau, khong chan cung, van hoc duoc.

> Gate **khong** phai xac suat "cam bien dang tin cay" da calibration. No la
> trong so dac trung hoc tu loss, khong nhan severity ground truth.

### 7.4 Ablation `cross_modal = false`

```
vI = concat(gI, 0, m)        sI = shared_mlp(vI)    gateI = gate_head_I(vI)
vU = concat(0, gU, m)        sU = shared_mlp(vU)    gateU = gate_head_U(vU)
```

Mask **ca summary lan gate**, khong chi zero mot cho. Test xac minh: doi gia tri
IMU khong lam doi output anh, va nguoc lai. So tham so **giu nguyen**.

---

## 8. Hai decoder — du doan delta coefficients

File: `qjepa/models/decoders.py`. Cau truc giong nhau cho 1D va 2D.

Moi buoc: resize ve **size THAT cua skip** → concat skip → Stage(stride 1).

### 8.1 Decoder anh — **do duoc**

| Tang | Channels | Output |
| --- | --- | --- |
| up2 | 128+96 → 96 | `[B,96,32,32]` |
| up1 | 96+64 → 64 | `[B,64,64,64]` |
| up0 | 64+32 → 32 | `[B,32,128,128]` |
| head | Conv2d(32→48, k=3), tuyen tinh | `delta_CI [B,48,128,128]` |

### 8.2 Decoder IMU — **do duoc**

| Tang | Channels | Output |
| --- | --- | --- |
| up2 | 128+96 → 96 | `[B,96,16]` |
| up1 | 96+64 → 64 | `[B,64,32]` |
| up0 | 64+32 → 32 | `[B,32,64]` |
| head | Conv1d(32→12, k=3), tuyen tinh | `delta_CU [B,12,64]` |

Resize: bilinear (2D) / linear (1D), `align_corners=False`.

### 8.3 Cong residual

```
CI_hat = CI_bad + delta_CI   →  image_hat    = iQWT(CI_hat)
CU_hat = CU_bad + delta_CU   →  imu_hat_norm = iHaar(CU_hat)  →  × scale + mu  →  imu_hat_phys
```

**Hai head cuoi zero-init** (weight va bias = 0). Vi vay model moi khoi tao tra
`delta = 0` → output = **dung dau vao** (do duoc: `2.98e-07`). Model hoc dan
*sai khac so voi identity*, khong hoc lai anh tu dau.

Khong ReLU/sigmoid/clamp tren he so hay delta. Khong clamp `image_hat` trong
train loss; chi clamp khi tinh metric va xuat PNG.

> He qua cua zero-init: gradient toi encoder tu reconstruction co the **bang 0 o
> buoc dau**. Phai kiem tra backbone sau 3–10 update, khong ket luan tu backward
> dau tien. Test `test_backbone_moves_after_a_few_updates` lam dieu do.

---

## 9. Nhanh JEPA — chi ton tai khi train

### 9.1 Token hoa

```
image tokens: FI_fused [B,128,16,16] → flatten row-major → [B,256,128]
imu   tokens: FU_fused [B,128,8]     → transpose          → [B,8,128]
```

### 9.2 Predictor

```
P(z) = Linear(256→128)( GELU( Linear(128→256)( LayerNorm₁₂₈(z) ) ) )
```

Hoat dong tren chieu cuoi; so token khong doi. Du doan **latent**, khong phai
pixel hay gia tri gyro.

### 9.3 Teacher EMA

```
TI = teacher_image_encoder( QWT(image_clean) )
TU = teacher_imu_encoder(  Haar(normalize(imu_clean)) )
```

- Teacher = **deepcopy encoder online** tai luc bat dau Stage B (khong phai tu dau).
- `requires_grad_(False)`, `eval()`, forward trong `no_grad()`.
- `model.train()` duoc override de teacher **luon o eval**.
- Teacher **khong** co fusion, decoder hay predictor — moi target giu thong tin
  cua chinh modality do.
- Khong nam trong optimizer; khong chia se storage voi online.

Cap nhat sau **moi optimizer step thanh cong**:

```
θ_teacher = momentum · θ_teacher + (1-momentum) · θ_online
momentum(s) = 0.99 + (0.999-0.99) · 0.5 · (1 - cos(π·p)),   p = clip(s/(N-1), 0, 1)
```

Khong EMA o tung microbatch; fp16 bo step thi khong dem, khong scheduler, khong EMA.

### 9.4 Loss JEPA

```
JI = mean( SmoothL1( LN(P_I(zi_tokens)), stopgrad(LN(TI_tokens)) ) )
JU = mean( SmoothL1( LN(P_U(zu_tokens)), stopgrad(LN(TU_tokens)) ) )
L_JEPA = 0.5 · (JI + JU)
```

`LN` = LayerNorm **khong affine**, eps 1e-5. Mean **rieng tung modality** roi moi
cong — neu khong, 256 token anh se chiem trong so gap 32 lan 8 token IMU.

---

## 10. Loss tong

```
L_total = w_image·L_image + w_imu·L_imu + λ_J·L_JEPA
          [+ λ_edge·L_edge + λ_delta·L_delta + λ_var·L_var]     # mac dinh 0, khong tinh

L_image = mean|image_hat - image_clean|                          # L1 tren [0,1]
L_imu   = 0.5·( SmoothL1(accel_norm) + SmoothL1(gyro_norm) )     # can bang accel/gyro
```

Loss phu chi duoc tinh khi weight > 0 — tranh `0 × NaN` va compute thua.

---

## 11. Lich huan luyen — **vua hoc vua khoi phuc**

| | Stage A (step 0–2000) | Stage B (step 2000–10000) |
| --- | --- | --- |
| Loss khoi phuc | **bat** | **van bat** |
| Loss JEPA | tat (λ_J = 0) | bat, ramp 0.01 → 0.10 |
| Teacher | chua ton tai | khoi tao o dau Stage B, cap nhat EMA |

**Khong phai pretrain → finetune.** Ca hai stage deu khoi phuc; Stage B chi *cong
them* mot loss phu. λ_max = 0.10 nen JEPA dong gop ~10% trong so.

```
λ_J(j) = 0.01 + (0.10 - 0.01) · min( j / (R-1), 1 ),    R = 1000 update
```

Optimizer: AdamW, LR 2e-4, betas (0.9, 0.999), weight decay 1e-4 (**khong** ap
cho bias va affine cua norm). Warmup 5% roi cosine ve 1e-6, tinh theo optimizer
update **thanh cong**. Grad clip global norm 1.0 sau `unscale_`.

---

## 12. Bang shape day du (B = batch)

| Tensor | Shape |
| --- | --- |
| Anh tren dia | `[256,256,3]` RGB |
| IMU mot window tren dia | `[128,6]` |
| `image_bad` vao model | `[B,3,256,256]` |
| He so QWT `CI` | `[B,48,128,128]` |
| S0 / S1 / S2 / **FI** | `[B,32,128,128]` / `[B,64,64,64]` / `[B,96,32,32]` / **`[B,128,16,16]`** |
| `imu_bad` normalized | `[B,6,128]` |
| He so Haar `CU` | `[B,12,64]` |
| V0 / V1 / V2 / **FU** | `[B,32,64]` / `[B,64,32]` / `[B,96,16]` / **`[B,128,8]`** |
| `gI`, `gU`, `s` | moi cai `[B,128]` |
| metadata `m` | `[B,3]` |
| vao shared MLP `v` | `[B,259]` |
| `FI_fused` / `FU_fused` | `[B,128,16,16]` / `[B,128,8]` |
| token JEPA anh / IMU | `[B,256,128]` / `[B,8,128]` |
| `delta_CI` / `delta_CU` | `[B,48,128,128]` / `[B,12,64]` |
| `image_hat` / `imu_hat_phys` | `[B,3,256,256]` / `[B,6,128]` |

---

## 13. So tham so — **dem tu model that**

| Module | Params |
| --- | ---: |
| `image_encoder` (CNN2D) | 753.024 |
| `imu_encoder` (CNN1D) | 248.832 |
| `shared_fusion` | 331.264 |
| `image_decoder` | 586.416 |
| `imu_decoder` | 192.780 |
| `image_predictor` | 66.176 |
| `imu_predictor` | 66.176 |
| **Tong online** | **2.244.668** |
| teacher (chi khi train) | +1.001.856 |
| **Sau export** (bo teacher + predictor) | **2.112.316** |

Transform khong co tham so hoc — bo loc db4 luu lam **buffer** (16 gia tri), di
theo checkpoint, khong nam trong optimizer. `normalizer` giu 19 buffer
(mu, scale, floor, fitted).

---

## 14. Nhung gi **khong** co trong model

Ghi ro de khong bi hieu nham:

| Khong co | Ghi chu |
| --- | --- |
| Transformer / attention | fusion la MLP co cong; spec §0.6 khong bat buoc attention |
| Frame truoc/sau | dung **mot** anh; khong co truc K, khong temporal mixer |
| Optical flow, pose supervision | pose chi dung de audit du lieu |
| Pretrained backbone | train tu dau; khong ap ImageNet normalization |
| Masking kieu I-JEPA | `mask_ratio = 0` |
| Sinh anh / lay mau / diffusion | chi du doan delta he so |
| Barometer, velocity | khong co trong dataset nay |
| Bien doi toa do camera–IMU hoc duoc | khong co |
| Tich phan quan tinh trong forward | khong co |

---

## 14b. Migration v2 (themjacobian.md)

Mac dinh **TAT**; bat bang `migration_schema_version: 2`.

| Bo sung | Vi tri | Ghi chu |
| --- | --- | --- |
| `encode_image_dense` / `encode_imu_dense_normalized` | `joint_jepa.py` | tra dung FI/FU ma `forward` dung, **cung instance** |
| `encode_and_fuse` | `joint_jepa.py` | ZI/ZU cho probe decoder |
| `freeze_backbone` | `joint_jepa.py` | `requires_grad_(False)` + `.eval()` |
| FD sensitivity | `training/encoder_sensitivity.py` | do tai FI/FU **truoc fusion** |
| Probe decoder | `models/representation_probe.py` | 0,61 M, he so **tuyet doi**, chi doc ZI/ZU |

Loss moi:

```
L_total = L_old + lambda_encoder * multiplier * L_enc_a
L_enc_a = mean_batch( mean_nonbatch((h'-h)^2) / mean_nonbatch((x'-x)^2) )
h = LayerNorm khong affine theo channel cua FI/FU      # CHI de do
```

Gradient tu `L_enc_sens` chi toi **encoder cua source bi perturb** — khong toi
fusion, decoder, predictor, teacher hay encoder con lai (da kiem bang autograd).

## 15. Gioi han da biet

- **Mot anh don khong quan sat truc tiep chuyen dong giua cac frame**, nen loi ich
  cua fusion anh–IMU cho khu nhieu **chua duoc bao dam**. Phai do bang
  `configs/no_cross.yaml`.
- `centered_offline` nhin truoc ~0,635 s quanh anh → **khong phai pipeline causal
  thoi gian thuc**. Khong goi day la real-time.
- Cac window IMU ke nhau chong lan ~92% → khong coi so window la so quan sat doc lap.
- Voi `w_imu = 1` nhu dac ta, gradient vao nhanh IMU nho hon nhanh anh **~88 lan**
  va nhanh IMU khong hoc. Xem TRAINING_REPORT §4.
- Chua co bang chung ve loi ich cua **JEPA**, cua **cross-modal fusion**, hay cua
  **QWT so voi Haar** — `λ_J` chua bao gio > 0 trong mot run co ket qua.
- Chua co bang chung ve loi ich cua **Jacobian regularization** (v2): khong co
  parent Stage B hop le nen control/treatment chua chay (G5 = NOT_RUN).
