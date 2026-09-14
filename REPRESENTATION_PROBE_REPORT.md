# REPRESENTATION_PROBE_REPORT.md

Decoder danh gia bieu dien tren backbone dong bang, theo
[themjacobian.md](themjacobian.md) muc 12, 13.

> **Trang thai: IMPLEMENTED, chua co ket qua so sanh.** Chua co cap backbone
> control/treatment hop le de train hai probe. So lieu duoi day chung minh **cai
> dat dung**, khong phai ket luan ve chat luong bieu dien.

## 1. Kien truc (muc 12.2, 12.3) — shape do duoc

Nhanh anh, tu `ZI [B,128,16,16]`:

| Buoc | Phep toan | Output |
| --- | --- | --- |
| P2 | Upsample 32×32 → ConvBlock2D(128,96) → ResBlock2D(96) | `[B,96,32,32]` |
| P1 | Upsample 64×64 → ConvBlock2D(96,64) → ResBlock2D(64) | `[B,64,64,64]` |
| P0 | Upsample 128×128 → ConvBlock2D(64,32) → ResBlock2D(32) | `[B,32,128,128]` |
| Head | Conv2D(32,48,k=3), tuyen tinh | `[B,48,128,128]` |
| Synthesis | inverse QWT cua he so **tuyet doi** | `[B,3,256,256]` |

Nhanh IMU, tu `ZU [B,128,8]`:

| Buoc | Phep toan | Output |
| --- | --- | --- |
| P2 | Upsample 16 → ConvBlock1D(128,96) → ResBlock1D(96) | `[B,96,16]` |
| P1 | Upsample 32 → ConvBlock1D(96,64) → ResBlock1D(64) | `[B,64,32]` |
| P0 | Upsample 64 → ConvBlock1D(64,32) → ResBlock1D(32) | `[B,32,64]` |
| Head | Conv1D(32,12,k=3), tuyen tinh | `[B,12,64]` |
| Synthesis | inverse Haar cua he so tuyet doi | `[B,6,128]` normalized |

Tham so: **0,61 M**, optimizer va checkpoint **rieng** khoi backbone.

Lich size suy tu **metadata kien truc tinh** (`Hc/4`, `Hc/2`, `Hc`), khong doc
shape tu skip tensor theo sample.

## 2. Bang chung khong ro ri thong tin (muc 12.1, G3)

| Rang buoc | Cach kiem | Ket qua |
| --- | --- | --- |
| Chi nhan ZI/ZU | doc `forward.__code__.co_varnames` | `('self','zi','zu')` |
| Khong anh/IMU goc | doi input goc, giu nguyen latent | output **khong doi** |
| Khong skip encoder | grep than ham forward | khong co `image_skips`/`imu_skips` |
| Khong residual he so nhieu | zero-latent cho output khac | khong tai hien anh goc |
| Khong teacher/predictor | grep than ham forward | khong xuat hien |
| He so **tuyet doi** | head khong zero-init | `|W| > 0` |
| Permute batch | permute latent | output permute tuong ung |
| Latent hang so | ba sample cung latent | ba output **giong het nhau** |

Config validator tu choi `encoder_skips=true`, `input_coefficient_residual=true`,
`teacher_features=true`, `predictor_features=true`, `coefficients=residual` va
backbone chua frozen.

## 3. Bang chung backbone that su dong bang (muc 13)

`backbone_fingerprint()` hash **toan bo parameters + buffers**. Sau mot optimizer
step cua probe:

```
backbone hash 0ea4fd387bb0bef8 | probe init hash 0dd7a6320fd57c70
probe params 0.61M | backbone dong bang: True
...
probe xong: {'steps': 40, ..., 'backbone_unchanged': True}
```

- `test_backbone_unchanged_after_probe_optimizer_step`: fingerprint **khong doi**,
  `p.grad is None` o moi tham so backbone, va probe **co** gradient khac 0.
- `freeze_backbone()` dat `requires_grad_(False)` va `.eval()` cho encoder, fusion,
  normalizer va ca hai transform.
- Vong train goi `self.backbone.eval()` moi epoch — backbone khong bao gio ve train mode.
- Optimizer chi chua `probe.parameters()`.

## 4. Initialization chung cho control/treatment

`test_shared_initialization_hash_for_control_and_treatment`: cung
`initialization_seed` cho **cung** `state_hash`; seed khac cho hash khac. Hash
duoc luu trong checkpoint probe cung `backbone_hash` va `frozen_fingerprint`.

Head dung Kaiming (linear) + bias 0. **Khong** zero-init (day la decoder he so
tuyet doi, khong phai correction head) va **khong** copy trong so tu decoder
phuc hoi da train.

## 5. Chay thu (smoke, KHONG phai ket qua)

40 update tren backbone `outputs/ovf_balanced/last.pt`:

```
[probe] step  1/2000 loss 68.57796 img 0.68573 imu 0.67892
[probe] step 40/2000 loss 42.13588 img 0.39140 imu 0.41744
  [probe val 40] psnr 6.956 ssim 0.0259 accel_rmse 5.1613 gyro_rmse 0.41993
```

Loss giam, gradient huu han, backbone khong doi. **Chat luong khong co y nghia**:

1. 40/2000 update — mới 2% ngan sach.
2. Backbone la run **overfit tren 16 sample**, khong phai Stage B that.
3. `imu_weight=100` ke thua tu config balanced khien loss IMU chiem gan het tong.

## 6. Gioi han dien giai (muc 12.4)

Decoder nay la **probe phi tuyen co nang luc huu han**. No do thong tin **giai ma
duoc theo kien truc va ngan sach da chon**, khong do toan bo thong tin toan hoc
trong latent.

No co the phuc hoi mem hon decoder chinh vi **khong co skip/residual** — dieu do
khong tu chung minh encoder vo dung. Neu probe kem ma decoder chinh tot, ket luan
dung la: **duong bypass (skip + residual he so) dang gop phan lon vao chat luong**,
chua du bang chung rang bieu dien cuoi giu du thong tin.

## 7. Chua chay

- C-probe va T-probe tu hai backbone khac nhau, cung init/budget — **NOT_RUN**.
- Ngan sach day du 2.000 update.
- Learning curve de nhan dien undertraining.
- So sanh `last` cung budget va `best_joint_validation` theo cung rule.
