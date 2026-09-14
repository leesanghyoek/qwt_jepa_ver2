# CONFIGURATION.md — tra cuu cau hinh

Schema strict: moi field khong khai bao deu bi reject, o **moi cap long nhau**.
Xem [qjepa/config.py](qjepa/config.py). Sau khi load, `resolved_config.yaml`
duoc ghi vao `output_dir` voi day du mac dinh da resolve.

## Rang buoc duoc kiem tra khi `validate()`

| Rang buoc | Ly do |
| --- | --- |
| `schema_version == 3` | tranh chay config cua ban dac ta khac |
| `imu_window_samples == 128` | L co dinh theo yeu cau (dac ta muc 0.8) |
| `imu_window_samples % 16 == 0` | ba lan downsample + Haar |
| `image_size` chan | wavelet mot level |
| `encoder_channels[-1] == embedding_dim` | fusion nhan dung D |
| moi channel chia het `groupnorm_groups` | GroupNorm |
| `transform.image in {qwt, dwt_haar_baseline}` | khong backend la |
| `allow_silent_fallback == false` | khong tu fallback QWT -> Haar (muc 6.3) |
| `allow_imu_padding == false` | khong pad hang gia (muc 3.3) |
| `stage_a + stage_b == max_optimizer_steps` | budget nhat quan |

## Nhom field chinh

### `data`

| Field | Mac dinh | Ghi chu |
| --- | --- | --- |
| `imu_window_samples` | 128 | **co dinh**; 128 hang = 1,27 s o 100 Hz |
| `image_size` | `[256,256]` | nguon khac se resize canh ngan + center crop |
| `window_mode` | `centered_offline` | nhin truoc ~0,635 s — khong phai causal |
| `max_center_error_in_imu_dt` | 1.0 | loai sample thay vi ghep cuong ep |
| `max_relative_dt_deviation` | 0.01 | window phai gan deu |
| `split_ratio` | `[0.8,0.1,0.1]` | theo **trajectory**; bo qua neu folder view da co split |
| `imu_std_floor` | accel 1e-3, gyro 1e-4 | on dinh so hoc, khong phai muc nhieu |

### `transform`

| Field | Mac dinh | Ghi chu |
| --- | --- | --- |
| `image` | `qwt` | 48 kenh he so; `dwt_haar_baseline` cho 12 kenh |
| `image_levels` | 1 | chi ho tro mot level |
| `imu` | `dwt_haar_1d` | 12 kenh he so |

### `model`

| Field | Mac dinh | Ghi chu |
| --- | --- | --- |
| `encoder_channels` | `[32,64,96,128]` | chia het 8 cho GroupNorm |
| `embedding_dim` | 128 | D |
| `imu_summary_bins` | 4 | pool FU dai 8 thanh 4 bin |
| `fusion_hidden_dim` | 256 | shared MLP `259 -> 256 -> 128` |
| `cross_modal` | true | false = mask ca summary lan gate (muc 9.3) |
| `gate_bias_init` | -2.0 | gate khoi tao `sigmoid(-2) = 0,119` |
| `zero_init_coefficient_heads` | true | model khoi tao ~ identity |

### `loss`

| Field | Mac dinh | Ghi chu |
| --- | --- | --- |
| `image_weight` | 1.0 | L1 tren `[0,1]` |
| `imu_weight` | 1.0 | **xem canh bao ben duoi** |
| `smooth_l1_beta` | 1.0 | ap dung cho loss IMU va JEPA |
| `jepa_start_weight` → `jepa_max_weight` | 0.01 → 0.10 | ramp tuyen tinh |
| `jepa_ramp_updates` | 1000 | tinh tu step dau cua Stage B |
| `edge_weight`, `imu_delta_weight`, `variance_weight` | 0.0 | chi tinh khi > 0 |

> **Canh bao `imu_weight`.** Voi `w_imu = 1` va `beta = 1`, gradient vao nhanh
> IMU nho hon nhanh anh khoang **88 lan** (do duoc, xem DATA_AUDIT muc 7), va
> nhanh IMU khong hoc trong gate G3. Hai config doi chung da chuan bi san:
> `main_qwt_balanced.yaml` (`imu_weight=100`) va `main_qwt_smallbeta.yaml`
> (`smooth_l1_beta=0.01`, `imu_weight=5`).

### `train`

| Field | Mac dinh | Ghi chu |
| --- | --- | --- |
| `learning_rate` | 2e-4 | AdamW; warmup 5% roi cosine ve `minimum_lr` |
| `batch_size` / `gradient_accumulation` | 4 / 2 | effective 8; Kaggle dung 8 / 1 |
| `precision` | `fp32` | `fp16` tren T4; transform LUON chay fp32 |
| `grad_clip_norm` | 1.0 | sau `unscale_` khi dung AMP |
| `stage_a_optimizer_steps` | 2000 | `lambda_J = 0` |
| `stage_b_optimizer_steps` | 8000 | teacher khoi tao o day, JEPA ramp |
| `teacher_momentum_start/end` | 0.99 / 0.999 | lich cosine theo so update JEPA |
| `num_workers` | 0 | tang sau khi xac nhan tai lap duoc |

`scheduler` va EMA chi buoc khi optimizer update **thanh cong** — fp16 bo step
thi khong dem, khong scheduler, khong EMA.

### `evaluation`

| Field | Mac dinh | Ghi chu |
| --- | --- | --- |
| `image_metric_clamp` | true | clamp `[0,1]` khi tinh metric; van bao `raw_mae` |
| `test_realizations` | `[0,1,2]` | robustness, khong chon realization dep nhat |
| `clean_*_tolerance` | 1/255, 0.02, 0.001 | nguong giu du lieu sach |

## Checkpoint

| File | Y nghia |
| --- | --- |
| `last.pt` | trang thai day du de `--resume` |
| `best_image.pt` / `best_imu.pt` | tot nhat theo tung nguon — **khong** dam bao hon identity |
| `best_joint.pt` | chi luu khi **ca ba** ty so `r_image`, `r_acc`, `r_gyro` < 1 |
| `inference.pt` | bo teacher va predictor |

`--resume` tu choi chay neu `manifest_hash` hoac image transform khac checkpoint.
Dung `--init` de bat dau thi nghiem moi tu trong so cu.
