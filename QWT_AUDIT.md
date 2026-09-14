# QWT_AUDIT.md — kiem chung bien doi wavelet

Spec muc 6.2 yeu cau ghi ro filter coefficients, sampling, boundary handling,
reconstruction scaling va code revision. Tai lieu nay la ban ghi do.

## 1. Backend dang dung

| Truong | Gia tri |
| --- | --- |
| Backend id | `qwt_dualtree_db4` |
| Revision | `1.0.0` |
| File | [qjepa/transforms/qwt.py](qjepa/transforms/qwt.py) |
| Levels | 1 (cau hinh chinh) |
| Boundary mode | `periodic` (circular extension) |
| Scale convention | `orthonormal_db4_mean_of_four_trees` |
| Coefficient channels | 48 = 3 RGB x 4 band x 4 component |

## 2. Filter bank

Bo loc scaling la **Daubechies db4** (4 vanishing moments, 8 taps), chuan hoa
`sum(h0) = sqrt(2)`. Nguon: I. Daubechies, *Ten Lectures on Wavelets*, SIAM
1992, Table 6.1; trung voi `pywt.Wavelet('db4').dec_lo` dao nguoc.

```
h0 = [ 0.230377813308855230,  0.714846570552541500,
       0.630880767929590400, -0.027983769416983850,
      -0.187034811718881140,  0.030841381835986965,
       0.032883011666982945, -0.010597401784997278]

h1[n] = (-1)^n * h0[F-1-n]        # QMF truc giao
```

Tinh truc chuan duoc kiem tra **bang so** trong test, khong tin vao hang so
chep tay:

| Kiem tra | Gia tri do duoc |
| --- | --- |
| `sum(h0)` | 1.414213562373 (= sqrt(2)) |
| `\|\|h0\|\|^2` | 0.999999999999 |
| `<h0, h0 dich 2>` | -1.15e-14 |
| `<h0, h1>` | 6.94e-18 |

## 3. Cau truc dual-tree va y nghia quaternion

Hai cay lech nhau MOT mau dau vao. Sau downsample 2, mot mau dau vao tuong ung
nua mau tren luoi he so — dung do tre nua mau ma cap Hilbert doi hoi. Day la
cau truc level-1 tieu chuan cua dual-tree (Selesnick, Baraniuk, Kingsbury,
*The Dual-Tree Complex Wavelet Transform*, IEEE SPM 22(6), 2005, muc "first
stage").

Tich separable cua hai cay theo hai truc cho bon thanh phan quaternion, theo
nghia cua Chan, Choi, Baraniuk, *Coherent Multiscale Image Processing Using
Dual-Tree Quaternion Wavelets*, IEEE TIP 17(7), 2008:

| component | (tree theo x, tree theo y) | Y nghia |
| --- | --- | --- |
| `real` | (a, a) | khong Hilbert |
| `i` | (b, a) | Hilbert theo truc x |
| `j` | (a, b) | Hilbert theo truc y |
| `k` | (b, b) | Hilbert theo ca hai truc |

Moi to hop cay cho bon subband, xep theo `band_order`:

```
(approx, detail_1, detail_2, detail_3) = (LL, LH, HL, HH)
LH = thap theo x, cao theo y      HL = cao theo x, thap theo y
```

Packing: `channel = ((rgb * 4) + band) * 4 + component`.

## 4. Bang chung day LA cap Hilbert, khong phai Haar doi ten

Round-trip chinh xac KHONG chung minh mot bien doi la QWT — mot Haar doi ten
cung round-trip hoan hao. Bang chung rieng: dung mot wavelet 1D tu moi cay,
lap tin hieu giai tich `psi_a + i*psi_b` va do phan nang luong o nua pho am.

| Cau truc | Ty le nang luong o nua pho DUONG |
| --- | --- |
| Cap Hilbert ly tuong | 100% |
| **Implementation nay** | **82.8%** |
| Hai cay giong nhau (vi du Haar doi ten) | ~50% |

82.8% la muc dac trung cua dual-tree o **level 1**, noi tinh giai tich yeu nhat
va se tot len o cac level sau. Test `test_qwt_trees_form_hilbert_pair` dat
nguong 75% va se FAIL neu ai do thay backend bang hai cay giong nhau.

## 5. Tinh nguoc

Moi to hop cay la mot bien doi **truc giao** (db4 truc chuan + extension tuan
hoan), nen synthesis cua no dung bang adjoint cua analysis va tai tao chinh xac.
Bien doi tong hop du thua 4 lan; synthesis lay trung binh bon tai tao.

Vi du thua, `analysis(synthesis(c)) == c` KHONG dung voi c tuy y — dung nhu spec
muc 6.2 ghi. Gate bat buoc la round-trip tren **du lieu dau vao** va gradient
cua synthesis, ca hai deu PASS.

## 6. Ket qua gate G1 (`python -m qjepa check-transform`)

| Transform | Case | Channels | Relative error | Max abs error |
| --- | --- | ---: | ---: | ---: |
| qwt | random | 48 | 1.011e-07 | 2.384e-07 |
| qwt | zeros | 48 | n/a | 0.000e+00 |
| qwt | constant | 48 | 1.825e-08 | 2.980e-08 |
| qwt | rgb_distinct | 48 | 1.086e-07 | 1.192e-07 |
| dwt_haar_baseline | random | 12 | 9.616e-08 | 2.384e-07 |
| dwt_haar_1d | random | 12 | 6.488e-08 | 3.576e-07 |

**PASS** — worst relative error 1.086e-07, nguong spec la 1e-5.

Kiem tra khac da PASS:

- `gradcheck` float64 tren synthesis.
- Gradient chay toi 100% he so, huu han.
- Ba kenh RGB doc lap: nhieu loan kenh R doi he so R, kenh G va B doi **dung 0.0**.
- Reject kich thuoc le thay vi tu pad.
- Reject layout sai revision thay vi `strict=False`.

## 7. Nhung dieu KHONG duoc ket luan tu tai lieu nay

- Chua co doi chieu he so voi mot implementation QWT doc lap thu ba tren cung
  input. Muc 4 chung minh cau truc dual-tree bang tinh giai tich, chua phai
  doi chieu so hoc voi [repository QUAVE/QWT](https://github.com/ispamm/QWT).
- Level 1 co tinh giai tich yeu nhat trong ho dual-tree; khong suy rong ket qua
  sang cau hinh nhieu level.
- Round-trip va gradient la tinh chat SO HOC. Chung khong chung minh QWT giup
  chat luong phuc hoi tot hon Haar — dieu do phai do bang run
  `configs/haar_baseline.yaml` cung budget/seed/split.
