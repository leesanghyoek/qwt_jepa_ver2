# ENCODER_SENSITIVITY_REPORT.md

Bao cao regularization Jacobian tai encoder theo [themjacobian.md](themjacobian.md)
muc 6, 7, 9, 11.

> **Trang thai: IMPLEMENTED, chua co ket qua chat luong.** Khong co parent Stage B
> hop le nen phase control/treatment **chua chay** — xem
> [MIGRATION_V2_AUDIT.md](MIGRATION_V2_AUDIT.md) muc 4. Moi so lieu duoi day la
> **do bang so tren batch that** de kiem chung cai dat, khong phai bang chung
> Jacobian regularization co loi.

## 1. Cong thuc da cai dat

Tap tai **FI/FU dense cuoi hai encoder online, TRUOC fusion**:

```
e_a[n] = mean_nonbatch( (x'_a[n] - x_a[n])^2 )      # nang luong delta THUC, core units
d_a[n] = mean_nonbatch( (h'_a[n] - h_a[n])^2 )      # h = LayerNorm khong affine cua f
g_a[n] = d_a[n] / e_a[n]
L_enc_a = mean_batch( g_a )
L_total = L_old + lambda_encoder * multiplier * L_enc_a
```

- Perturb **truoc wavelet**, khong perturb tung he so QWT.
- LayerNorm **chi de DO**; FI/FU goc van vao fusion khong doi (`feature_to_fusion: raw_unchanged`).
- Mot source moi update thanh cong, luan phien anh/IMU; moi microbatch trong cung
  accumulation group dung **cung** source.
- `lambda(s) = weight_max * min((s+1)/ramp_updates, 1)`, `weight_max=1e-4`, `ramp=200`.

Day la **normalized-latent directional finite-difference sensitivity**, khong phai
Frobenius/spectral norm hay Lipschitz bound cua Jacobian that.

## 2. Do duoc tren batch that (8 update dau, treatment)

| step | source | lambda_enc | gain normalized | gain raw | E_in | clipped |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | image | 5.00e-07 | 171.97 | 100.07 | 1.424e-05 | 8.04% |
| 2 | imu | 1.00e-06 | 5.23 | 2.75 | 1.000e-04 | 0% |
| 3 | image | 1.50e-06 | 163.44 | 99.85 | 1.514e-05 | 1.74% |
| 4 | imu | 2.00e-06 | 8.17 | 4.03 | 1.000e-04 | 0% |
| 5 | image | 2.50e-06 | 141.96 | 85.89 | 1.526e-05 | 0.86% |
| 6 | imu | 3.00e-06 | 7.93 | 3.75 | 1.000e-04 | 0% |
| 7 | image | 3.50e-06 | 146.39 | 90.05 | 1.460e-05 | 5.34% |
| 8 | imu | 4.00e-06 | 7.84 | 4.13 | 1.000e-04 | 0% |

Kiem chung tu bang nay:

- **Luan phien source dung**: chan = anh, le = IMU.
- **Ramp dung**: tang tuyen tinh `weight_max/ramp = 5e-7` moi update.
- **E_in anh** ≈ 1,5e-05 so voi nominal `(1/255)² = 1,538e-05` — **thap hon** vi
  pixel o bien bi clamp; day la delta **thuc**, dung nhu muc 6.3 yeu cau.
- **E_in IMU** = 1,000e-04 = `0,01²` chinh xac (IMU khong clamp).
- **clipped fraction** khac 0 o anh, bang 0 o IMU — dung ky vong.

> **Hai cot gain khong cung don vi.** Anh do theo cuong do `[0,1]`, IMU theo do
> lech chuan train. Gain anh ~150 va gain IMU ~6 **khong** noi encoder anh "nhay
> gap 25 lan"; hai toa do khac nhau (muc 6.4).

> **Raw gain chi la diagnostic.** Raw gain giam co the do co scale feature chu
> khong phai robust hon. Khong cong vao loss, khong so truc tiep voi normalized
> gain (muc 8.6).

## 3. Canh bao so hoc da phat hien

Test `test_large_imu_magnitude_loses_perturbation_in_fp32`: o bien do IMU lon
(vi du 50 m/s²), FP32 khong giu du `epsilon = 0,01` — delta thuc do duoc la
9,9966e-05 thay vi 1,0000e-04.

Code **do delta thuc** nen ty so van dung, nhung dung gia dinh `E_in` luon bang
`epsilon²`. Neu bien do IMU tang them, `minimum_input_energy = 1e-12` se raise
loi ro rang thay vi am tham bao gain = 0.

## 4. Gradient routing — da kiem bang autograd

| Module | Gradient tu `L_enc_sens`? | Test |
| --- | --- | --- |
| Encoder cua source bi perturb | **Co** | `test_gradient_reaches_only_the_perturbed_encoder` |
| Encoder con lai | Khong | cung test |
| Fusion / predictor / decoder chinh | Khong | cung test |
| Teacher | Khong (`grad is None`) | `test_teacher_gets_no_gradient_from_sensitivity` |

Nhanh base **khong** bi detach — `test_base_feature_is_not_detached` so gradient
co/khong detach va yeu cau chung phai khac nhau.

Autograd khop **central finite difference** o float64 tren toy encoder
`f_theta(x) = A(theta)x` chay qua dung duong LN cua loss that
(`test_autograd_matches_finite_difference_on_toy_encoder`, atol 1e-6).

## 5. Kiem soat collapse (muc 8)

`qjepa/evaluation/representation.py::latent_statistics` bao cao:

1. `raw_rms` va `channel_variance_mean` — phat hien co scale.
2. `std_across_samples_*` — std qua **cac sample khac nhau tai cung vi tri/channel**,
   khong phai std giua cac pixel cua mot sample (do se tao variance gia).
3. `effective_rank` tu SVD entropy tren pooled feature, kem `max_rank_possible`.
4. `cosine_similarity_mean` giua cac sample.

Test `test_constant_feature_gives_zero_loss_but_is_detectable` chung minh diem
mau chot cua muc 8: feature hang so cho `L_enc = 0` **tuyet doi**, nhung
`std_across_samples_mean = 0.0` bat duoc ngay. **Loss nho khong chung minh bieu
dien con thong tin.**

## 6. Chua chay / chua chung minh

- Phase control va treatment tu cung parent Stage B — **NOT_RUN**, khong co parent hop le.
- Moi ket luan ve "latent on dinh hon", "bieu dien de giai ma hon" hay "he thong
  phuc hoi tot hon" — **chua co bang chung**.
- Alpha sweep 0,5/1/2 va bank 64 sample: code co trong `diagnose-sensitivity`,
  chua chay tren checkpoint co y nghia.
- Chua do overhead thoi gian/bo nho cua extra encoder forward tren run dai.
