# =============================================================================
#  KIEM TRA DATASET KAGGLE cho qwt_jepa_ver2  -- copy nguyen cell nay va chay
#  Tu chua: KHONG can clone code truoc. Output gioi han ~60 dong.
# =============================================================================
import os, pathlib, json, collections
import numpy as np

IMG_DIR = "image_lcam_front"
NEED = ["acc", "gyro", "imu_time", "cam_time"]
report = {}

# ---- 1. Tim data root -------------------------------------------------------
def find_roots():
    out = []
    for base in sorted(pathlib.Path("/kaggle/input").glob("*")):
        for cand in [base] + sorted(p for p in base.glob("*") if p.is_dir()):
            if any(cand.glob(f"*/Data_*/P*/{IMG_DIR}")):
                out.append(cand)
    return out

roots = find_roots()
print("=" * 68)
print("1. DATA ROOT")
if not roots:
    print("  !! KHONG TIM THAY. Cac thu muc trong /kaggle/input:")
    for p in sorted(pathlib.Path("/kaggle/input").glob("*")):
        print("     ", p.name, "->", [c.name for c in sorted(p.glob("*"))[:6]])
    raise SystemExit("Dung lai: chua Add Input dataset, hoac cau truc khac du kien.")
ROOT = roots[0]
print(f"  {ROOT}")
if len(roots) > 1:
    print(f"  (con {len(roots)-1} root khac: {[str(r) for r in roots[1:]]})")
report["root"] = str(ROOT)

# ---- 2. Environment / trajectory -------------------------------------------
trajs = sorted(ROOT.glob(f"*/Data_*/P*/{IMG_DIR}"))
trajs = [t.parent for t in trajs]
envs = sorted({t.parents[1].name for t in trajs})
diffs = collections.Counter(t.parent.name for t in trajs)
print("\n2. QUY MO")
print(f"  environment : {len(envs)}")
print(f"  trajectory  : {len(trajs)}")
print(f"  difficulty  : {dict(diffs)}")
print(f"  danh sach env: {envs}")
report.update(num_env=len(envs), num_traj=len(trajs), envs=envs)

# ---- 3. Dinh dang file IMU: .npy hay .txt? ---------------------------------
print("\n3. DINH DANG FILE IMU  (loader doc .npy truoc, khong co thi .txt)")
fmt = collections.Counter()
missing = []
for t in trajs:
    for name in NEED:
        if (t / "imu" / f"{name}.npy").is_file():   fmt[f"{name}.npy"] += 1
        elif (t / "imu" / f"{name}.txt").is_file(): fmt[f"{name}.txt"] += 1
        else: missing.append(f"{t.relative_to(ROOT)}/imu/{name}.*")
for k in sorted(fmt): print(f"  {k:16} {fmt[k]:4d}/{len(trajs)} trajectory")
print(f"  THIEU: {len(missing)}" + (f"  vi du: {missing[:2]}" if missing else ""))
report["imu_formats"] = dict(fmt); report["missing_count"] = len(missing)

# ---- 4. Doc thu 3 trajectory ------------------------------------------------
def load(d, name):
    npy, txt = d / f"{name}.npy", d / f"{name}.txt"
    if npy.is_file(): return np.load(npy).astype(np.float64)
    return np.loadtxt(txt, dtype=np.float64)

print("\n4. DOC THU 3 TRAJECTORY")
print(f"  {'trajectory':34} {'imu':>6} {'anh':>5} {'camT':>5} {'imu_Hz':>7} {'cam_Hz':>7} {'khop':>5}")
samples, ok_all = [], True
for t in trajs[:: max(1, len(trajs) // 3)][:3]:
    d = t / "imu"
    try:
        acc, gyro = load(d, "acc"), load(d, "gyro")
        it, ct = load(d, "imu_time"), load(d, "cam_time")
        imgs = sorted((t / IMG_DIR).glob("*.png"))
        fi = 1 / np.median(np.diff(it)) if len(it) > 1 else float("nan")
        fc = 1 / np.median(np.diff(ct)) if len(ct) > 1 else float("nan")
        match = len(imgs) == len(ct)
        ok_all &= match and acc.shape == gyro.shape and len(it) == len(acc)
        print(f"  {str(t.relative_to(ROOT)):34} {len(acc):6d} {len(imgs):5d} {len(ct):5d} "
              f"{fi:7.2f} {fc:7.2f} {str(match):>5}")
        samples.append(dict(traj=str(t.relative_to(ROOT)), imu_rows=len(acc),
                            images=len(imgs), imu_hz=round(fi, 3), cam_hz=round(fc, 3),
                            acc_shape=list(acc.shape), gyro_shape=list(gyro.shape),
                            images_match_camtime=bool(match)))
    except Exception as e:
        ok_all = False
        print(f"  {str(t.relative_to(ROOT)):34} LOI: {type(e).__name__}: {e}")
report["samples"] = samples

# ---- 5. Kich thuoc anh ------------------------------------------------------
print("\n5. KICH THUOC ANH")
try:
    from PIL import Image
    sizes, modes = collections.Counter(), collections.Counter()
    for t in trajs[:: max(1, len(trajs) // 5)][:5]:
        for p in sorted((t / IMG_DIR).glob("*.png"))[:2]:
            with Image.open(p) as im:
                sizes[im.size] += 1; modes[im.mode] += 1
    for s, n in sizes.items(): print(f"  {s[0]}x{s[1]}: {n} anh mau")
    print(f"  mode: {dict(modes)}")
    report["image_sizes"] = {f"{k[0]}x{k[1]}": v for k, v in sizes.items()}
    if list(sizes) != [(256, 256)]:
        print("  !! CHU Y: khong phai 256x256 -> loader se resize+center-crop")
except Exception as e:
    print("  bo qua:", e)

# ---- 6. Don vi / gia tri IMU ------------------------------------------------
print("\n6. GIA TRI IMU (mot trajectory)")
try:
    d = trajs[0] / "imu"
    acc, gyro = load(d, "acc"), load(d, "gyro")
    print(f"  acc  min={acc.min(0).round(2).tolist()} max={acc.max(0).round(2).tolist()}")
    print(f"  gyro min={gyro.min(0).round(3).tolist()} max={gyro.max(0).round(3).tolist()}")
    g = abs(acc.mean(0)).max()
    print(f"  |mean acc| lon nhat = {g:.2f} m/s^2 -> "
          f"{'CO trong luc (dung acc.npy, KHONG phai acc_nograv)' if g > 5 else 'NGHI da bo trong luc - kiem tra lai!'}")
    report["acc_range"] = [acc.min(0).tolist(), acc.max(0).tolist()]
    report["gyro_range"] = [gyro.min(0).tolist(), gyro.max(0).tolist()]
    p = d / "parameter.yaml"
    if p.is_file(): print("  parameter.yaml:", p.read_text().strip().replace("\n", " | "))
except Exception as e:
    print("  loi:", e)

# ---- 7. File phu tro --------------------------------------------------------
print("\n7. FILE PHU TRO (khong bat buoc voi loader moi)")
for rel in ["tartanair-v2-jepa/train/manifest.csv", "norm_stats.yaml"]:
    for base in [ROOT, ROOT.parent]:
        f = base / rel
        if f.is_file(): print(f"  co: {f}"); break
    else: print(f"  khong co: {rel}  (loader tu build-manifest, khong sao)")

# ---- 8. Ket luan ------------------------------------------------------------
print("\n" + "=" * 68)
print("KET LUAN:", "SAN SANG" if (ok_all and not missing) else "CO VAN DE - xem muc tren")
print(f"Lenh tiep theo:\n  python -m qjepa build-manifest --root {ROOT} --out /kaggle/working/outputs/manifest --window 128")
print("=" * 68)
print("\n----- COPY TU DAY GUI LAI -----")
print(json.dumps(report, indent=1)[:2500])
