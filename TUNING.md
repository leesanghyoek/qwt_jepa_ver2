# Vi sao cac thong so train duoc dat nhu vay

Moi thay doi duoi day **suy tu run that** cua ban (Kaggle 2x T4, 14 environment,
10.000 step, 21,6 phut), khong phai phong doan.

## Bang thay doi

| Tham so | Cu | **Moi** | Ly do |
| --- | ---: | ---: | --- |
| `max_optimizer_steps` | 10.000 | **30.000** | accel van con giam khi het ngan sach |
| `stage_a_optimizer_steps` | 2.000 | **4.000** | teacher tot hon truoc khi bat JEPA |
| `stage_b_optimizer_steps` | 8.000 | **26.000** | Stage B moi la cho accel cai thien manh |
| `minimum_lr` | 1e-6 | **1e-5** | duoi cosine cu gan nhu khong hoc gi |
| `validation_batches` | 32 | **128** | 128 mau qua it, gay nhieu do |
| `checkpoint_every_updates` | 500 | **1.000** | bot ghi dia, khong mat gi |
| `jepa_ramp_updates` | 1.000 | **2.000** | ty le voi Stage B dai gap 3 |
| `learning_rate` | 2e-4 | **2e-4** (giu) | da chay on dinh |
| `batch_size` x `accum` | 4 x 2 | **4 x 2** (giu) | giu effective batch 8 |

Uoc tinh: **~65 phut** train + ~4 phut validation. Van chi dung **9%** gioi han
phien 12h.

---

## 1. Vi sao tang len 30.000 step

Muc cai thien moi 1.000 step o run cu:

| | 1k→2k | 5k→6k | 9k→10k |
| --- | ---: | ---: | ---: |
| `r_image` | −0,0811 | −0,0115 | −0,0004 |
| `r_acc` | −0,1257 | −0,0305 | **−0,0010** |

Anh **da bao hoa** tu khoang step 5.000–6.000. Nhung **accel van dang giam** khi
ngan sach het — chua cham day.

Va ban moi dung **3%** thoi gian phien (21,6 phut / 12 gio). Khong co ly do gi
de dung som.

> 30.000 step = **4,7 epoch** tren 50.686 sample train.

## 2. Vi sao `minimum_lr` tu 1e-6 len 1e-5

Day la diem de bo sot. Lich cosine neo vao `max_optimizer_steps`, nen o run cu:

```
step  8.000/10.000 -> lr 2.20e-05
step 10.000/10.000 -> lr 1.00e-06     = 0,5% dinh
```

Khoang 1.000 step cuoi chay voi LR gan nhu bang 0 — **khong hoc gi**. Nghia la
mot phan "bao hoa" ta thay o cuoi **la do lich LR tat**, khong hoan toan do model
da hoi tu.

Voi `minimum_lr = 1e-5`, cuoi run LR con **5% dinh**, du de tiep tuc tinh chinh:

| step | lr moi | % dinh |
| ---: | ---: | ---: |
| 1.499 | 2,00e-04 | 100% (het warmup) |
| 15.000 | 1,13e-04 | 56% |
| 29.999 | **1,00e-05** | **5%** |

## 3. Vi sao Stage A tu 2.000 len 4.000

Teacher duoc tao bang `deepcopy` encoder online **tai luc bat dau Stage B**.
Teacher tot hon thi muc tieu latent tot hon.

O run cu, cuoi Stage A (step 2.000) da co `r_image = 0,5585` va `r_acc = 0,8106`
— encoder **da hoc duoc that**, du dieu kien lam teacher. Nhung voi ngan sach
30.000, danh 4.000 cho Stage A chi la **13%**, van con 26.000 cho Stage B.

Giu ty le nghieng ve Stage B vi do la noi accel cai thien manh nhat: Stage A dua
`r_acc` tu ~1,00 xuong 0,81; Stage B dua tiep xuong 0,56.

## 4. Vi sao `validation_batches` tu 32 len 128

Day la sua **phep do**, khong phai sua model.

Validation cu lay `32 batch x 4 = 128 mau`, tren tap valid ~5.100 mau — tuc chi
**2,5%**. Voi mau nho nhu vay, dao dong ta thay o `r_image`:

```
step 6.000: 0,4944
step 7.000: 0,4966   (+0,0022  — XAU di?)
step 8.000: 0,4880
step 9.000: 0,4929   (+0,0049  — XAU di?)
```

**rat co the chi la nhieu do**, khong phai model that su xau di. Voi 128 batch
(512 mau = 10% tap valid), duong cong se on dinh hon va viec chon
`best_joint.pt` dang tin hon.

Chi phi: ~8 giay moi lan do, 30 lan = **4 phut**.

## 5. Nhung gi **khong** doi, va vi sao

| Giu nguyen | Ly do |
| --- | --- |
| `learning_rate = 2e-4` | grad norm 0,03–0,17 suot run, khong lan nao cham nguong clip 1,0. On dinh. |
| `batch 4 x accum 2` | giu effective batch 8 de run 1 GPU va 2 GPU so sanh duoc |
| `imu_weight = 100` | da chung minh: accel giam 43,6%. Voi weight 1 thi nhanh IMU khong hoc. |
| `grad_clip_norm = 1.0` | chua bao gio kich hoat, giu lam luoi an toan |
| `warmup_fraction = 0.05` | 5% x 30.000 = 1.500 step, hop ly |

## 6. Phase v2 giu nguyen 1.000 update

`kaggle_v2_control` va `kaggle_v2_treatment` **khong** tang ngan sach. Dac ta
[themjacobian.md](themjacobian.md) muc 11 chot phase v2 la **1.000 successful
updates, LR 2e-5**. Doi ngan sach se lam ket qua khong con so sanh duoc voi
thiet ke thi nghiem.

## 7. Dieu nay **khong** bao dam ket qua tot hon

Nhung thay doi tren la **suy luan tu mot run, mot seed**. Chung hop ly nhung
chua duoc kiem chung:

- Accel co the bao hoa ngay sau 10.000 step, va 20.000 step them khong doi lai gi.
- Stage A dai hon co the **khong** cho teacher tot hon dang ke.
- `minimum_lr` cao hon co the lam cuoi run kem on dinh hon mot chut.

Cach duy nhat de biet: chay, roi so **duong cong validation** voi run 10.000
step cu (`training_curves.png`). Giu lai checkpoint cu de doi chieu.
