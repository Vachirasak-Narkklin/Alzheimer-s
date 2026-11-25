"""ล็อกการเตรียมภาพ (preprocessing / Target Shape) ให้ทุกโมเดลใช้ร่วมกัน (2D / 2.5D / 3D)

ไฟล์ล็อกทั้งสามของโปรเจ็ค (วางไว้ที่เดียว เช่น 00_shared แล้วให้ทุกโมเดล import / อ่านจากที่นี่)
  split.json            -> ใครอยู่ train / val / test
  augmentation_lock.py  -> มุมหมุนของแต่ละคนในแต่ละ epoch
  processing_lock.py    -> (ไฟล์นี้) ภาพเริ่มต้นที่ทุกโมเดลได้เหมือนกัน + วิธีตัดอินพุตของแต่ละโมเดล

ขั้นตอนที่ล็อก (ตรงกับที่ 3D ใช้อยู่แล้ว)
  1. โหลด *_masked_gfc.hdr (ถ้ามี 4 มิติ ใช้ช่องแรก)
  2. ตัดพื้นดำรอบสมอง (bounding box ของ voxel > 0, margin = 0)  -> 137 x 171 x 145
  3. clip ที่ percentile 1-99 แล้ว scale เป็น 0-1
  4. resize ทั้ง volume เป็น TARGET_SHAPE ด้วย linear (order=1)  -> 128 x 128 x 96
  ทุกโมเดลได้ volume นี้ "ตัวเดียวกัน" (เก็บเป็น cache พร้อม checksum)

หลังล็อกแล้ว แต่ละโมเดลทำของตัวเองด้วย make_input():
  "3D"   -> (1, 128, 128, 96) ทั้ง volume
  "2D"   -> (1, H, W)  axial slice กลาง (slice 48)
  "2.5D" -> (3, H, W)  [sagittal 64, coronal 64, axial 48]
  H = W = BASE_PLANE (128) หรือ out_size ที่ขอ (เช่น 224 สำหรับ ResNet18 pretrained)

การหมุน (augmentation) ทำบน volume ที่ล็อกแล้ว รอบแกน z (axes = (0, 1)) ด้วย linear + mode="nearest"
  3D   หมุนทั้ง volume
  2D   หมุน axial slice กลาง (เท่ากับ slice กลางของ volume ที่หมุนแล้ว)
  2.5D หมุน volume ก่อน แล้วตัดสามระนาบจาก volume ที่หมุนแล้ว (ไม่หมุนทีละระนาบ)
  มุมมาจาก augmentation_lock.angle_for(seed, epoch, subject_id, rotate_deg=15)
"""
import glob
import hashlib
import json
import os

import numpy as np
from scipy.ndimage import map_coordinates, zoom
from scipy.ndimage import rotate as _scipy_rotate

# ---------------------------------------------------------------- ค่าที่ล็อก
PREPROCESS_VERSION = "v1"
TARGET_SHAPE = (128, 128, 96)          # (x, y, z) ของ volume ที่ล็อก
CLIP_PERCENTILES = (1, 99)
RESIZE_ORDER = 1                        # linear
CROP_MARGIN = 0
BASE_PLANE = 128                        # ขนาดด้านของภาพ 2D / 2.5D ถ้าไม่ระบุ out_size
CENTER = tuple(s // 2 for s in TARGET_SHAPE)   # (64, 64, 48) = (sagittal, coronal, axial)

DATA_ROOT = os.environ.get("OASIS_ROOT", "/home/jovyan/shared/donut/Alzheimer/OAS1")
PATH_PATTERN = "{root}/OAS1_{sid}_MR1/PROCESSED/MPRAGE/T88_111/*_masked_gfc.hdr"
CACHE_NAME = "locked_volumes_{x}x{y}x{z}.npy".format(x=TARGET_SHAPE[0], y=TARGET_SHAPE[1], z=TARGET_SHAPE[2])
META_NAME = "locked_volumes_meta.json"


def params_dict():
    return {"version": PREPROCESS_VERSION, "target_shape": list(TARGET_SHAPE), "clip_percentiles": list(CLIP_PERCENTILES),
            "resize_order": RESIZE_ORDER, "crop_margin": CROP_MARGIN, "dtype": "float32"}


def preprocess_hash():
    """hash ของ 'ค่าที่ล็อก' (ไม่ขึ้นกับเครื่อง) ทุกโมเดลต้องได้ค่าเดียวกัน"""
    return hashlib.sha256(json.dumps(params_dict(), sort_keys=True).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- ขั้นที่ 1-4: เตรียม volume
def prep_volume(volume):
    """ตัดพื้นดำ -> clip percentile 1-99 + scale 0-1 -> resize เป็น TARGET_SHAPE (float32)"""
    v = np.asarray(volume)
    if v.ndim == 4:
        v = v[:, :, :, 0]
    if v.ndim != 3:
        raise ValueError(f"ต้องเป็น volume 3 มิติ แต่ได้ shape {v.shape}")
    nz = np.argwhere(v > 0)
    if nz.size == 0:
        raise ValueError("volume ว่าง ไม่มี voxel ที่ > 0")
    lo = np.maximum(nz.min(axis=0) - CROP_MARGIN, 0)
    hi = np.minimum(nz.max(axis=0) + CROP_MARGIN, np.array(v.shape) - 1)
    v = v[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    p_lo, p_hi = np.percentile(v, CLIP_PERCENTILES)
    v = np.clip(v, p_lo, p_hi)
    v = (v - p_lo) / (p_hi - p_lo + 1e-8)
    v = zoom(v, [t / s for t, s in zip(TARGET_SHAPE, v.shape)], order=RESIZE_ORDER)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def subject_path(subject_id, data_root=None):
    sid = str(subject_id).zfill(4)
    pattern = PATH_PATTERN.format(root=data_root or DATA_ROOT, sid=sid)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพของคน {sid}: {pattern}")
    return files[0]


def load_volume(subject_id, data_root=None):
    import nibabel as nib
    return prep_volume(nib.load(subject_path(subject_id, data_root)).get_fdata())


def volume_checksum(v):
    """checksum ของภาพ (ปัดเป็น 16 บิตก่อน กันความต่างจุดทศนิยมเล็กๆ ข้ามเครื่อง)"""
    q = np.rint(np.asarray(v, dtype=np.float64) * 65535).astype(np.uint16)
    return hashlib.sha256(q.tobytes()).hexdigest()[:10]


# ---------------------------------------------------------------- cache กลาง (ทุกโมเดลโหลดไฟล์เดียวกัน)
def build_cache(subject_ids, cache_dir=".", data_root=None, overwrite=False, verbose=True):
    """ประมวลผลภาพของทุกคนครั้งเดียว เก็บเป็น .npy + meta (checksum รายคน) ที่ cache_dir"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path, meta_path = os.path.join(cache_dir, CACHE_NAME), os.path.join(cache_dir, META_NAME)
    if os.path.exists(cache_path) and not overwrite:
        raise FileExistsError(f"มี cache แล้ว: {cache_path} (ใช้ overwrite=True ถ้าต้องการสร้างใหม่)")
    ids = sorted({str(s).zfill(4) for s in subject_ids})
    arr = np.zeros((len(ids),) + TARGET_SHAPE, dtype=np.float32)
    sums = {}
    for k, sid in enumerate(ids):
        arr[k] = load_volume(sid, data_root)
        sums[sid] = volume_checksum(arr[k])
        if verbose and ((k + 1) % 20 == 0 or k + 1 == len(ids)):
            print(f"  ประมวลผลแล้ว {k + 1}/{len(ids)}")
    import scipy
    np.save(cache_path, arr)
    meta = {"params": params_dict(), "preprocess_hash": preprocess_hash(), "ids": ids, "checksums": sums,
            "numpy": np.__version__, "scipy": scipy.__version__}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return cache_path


def load_locked(subject_ids, cache_dir=".", check=True, verbose=True):
    """คืน array (N, 128, 128, 96) เรียงตาม subject_ids ที่ขอ (ทุกโมเดลเรียกฟังก์ชันนี้)

    check=True ตรวจ: preprocess_hash ตรงกับค่าที่ล็อก, shape ถูก, checksum รายคนตรงกับ meta
    """
    cache_path, meta_path = os.path.join(cache_dir, CACHE_NAME), os.path.join(cache_dir, META_NAME)
    want = [str(s).zfill(4) for s in subject_ids]
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"ไม่พบ cache: {cache_path}\nสร้างก่อนด้วย build_cache(ids ทั้งหมดจาก split.json) แล้วใช้ไฟล์เดียวกันทุกโมเดล")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta["preprocess_hash"] != preprocess_hash():
        raise ValueError(f"cache สร้างด้วยค่าเก่า (hash {meta['preprocess_hash']} != {preprocess_hash()}) ต้องสร้างใหม่")
    index = {sid: k for k, sid in enumerate(meta["ids"])}
    missing = [s for s in want if s not in index]
    if missing:
        raise KeyError(f"ไม่มีใน cache {len(missing)} คน เช่น {missing[:5]} -> สร้าง cache ใหม่จากรายชื่อทั้งหมดใน split.json")
    full = np.load(cache_path, mmap_mode="r")
    if tuple(full.shape[1:]) != TARGET_SHAPE:
        raise ValueError(f"shape ใน cache {full.shape[1:]} ไม่ตรงกับที่ล็อก {TARGET_SHAPE}")
    out = np.stack([np.asarray(full[index[s]]) for s in want]).astype(np.float32)
    if check:
        bad = [s for s, v in zip(want, out) if volume_checksum(v) != meta["checksums"][s]]
        if bad:
            raise ValueError(f"checksum ไม่ตรงกับ meta {len(bad)} คน เช่น {bad[:5]} (ไฟล์ cache เสียหรือถูกแก้)")
    if verbose:
        print(f"[processing_lock] โหลด {len(want)} volume | shape {out.shape[1:]} | hash {preprocess_hash()}")
    return out


def locked_fingerprint(arr, ids):
    """ลายนิ้วมือชุดภาพ (พิมพ์ในทุกโน้ตบุ๊ก ต้องตรงกันทุกโมเดล ถ้าโหลดรายชื่อเดียวกัน)"""
    h = hashlib.sha256()
    for sid, v in zip(ids, arr):
        h.update(str(sid).zfill(4).encode())
        h.update(volume_checksum(v).encode())
    return h.hexdigest()[:10]


# ---------------------------------------------------------------- การหมุน (รอบแกน z) บน volume ที่ล็อก
def rotate_volume(v, angle):
    """หมุนทั้ง volume รอบแกน z (axes=(0,1)), linear, mode nearest, ขนาดเท่าเดิม"""
    if angle == 0:
        return v
    return _scipy_rotate(v, angle, axes=(0, 1), reshape=False, order=1, mode="nearest")


def _src(X, Y, c, s, mx, my):
    return c * (X - mx) + s * (Y - my) + mx, -s * (X - mx) + c * (Y - my) + my


def _sample(v, x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return map_coordinates(v, [x, y, z], order=1, mode="nearest")


def planes_of_rotated(v, angle=0.0):
    """ตัด (sagittal, coronal, axial) กลางจาก volume ที่ 'หมุนรอบแกน z แล้ว' โดยไม่ต้องสร้าง volume ที่หมุนทั้งก้อน

    ผลเท่ากับ rotate_volume(v, angle) แล้วตัด v[cx,:,:], v[:,cy,:], v[:,:,cz] (ต่างกันระดับ ~1e-15) แต่เร็วกว่ามาก
    """
    nx, ny, nz = v.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    if angle == 0:
        return v[cx, :, :], v[:, cy, :], v[:, :, cz]
    a = np.deg2rad(angle)
    c, s, mx, my = np.cos(a), np.sin(a), (nx - 1) / 2, (ny - 1) / 2
    X, Y, Z = np.arange(nx)[:, None], np.arange(ny)[None, :], np.arange(nz)[None, :]
    sx, sy = _src(np.float64(cx), Y.T, c, s, mx, my)
    sag = _sample(v, sx, sy, Z)                          # (ny, nz)
    sx, sy = _src(X, np.float64(cy), c, s, mx, my)
    cor = _sample(v, sx, sy, Z)                          # (nx, nz)
    sx, sy = _src(X, Y, c, s, mx, my)
    axi = _sample(v, sx, sy, np.float64(cz))             # (nx, ny)
    return sag, cor, axi


def _to_square(img, size):
    if img.shape == (size, size):
        return img
    return zoom(img, (size / img.shape[0], size / img.shape[1]), order=1)


# ---------------------------------------------------------------- อินพุตของแต่ละโมเดล (ส่วนที่ต่างกันได้)
def make_input(volume, mode, angle=0.0, out_size=None):
    """สร้างอินพุตของโมเดลจาก volume ที่ล็อก (128,128,96) พร้อมมุมหมุน (องศา, 0 = ไม่หมุน)

    mode "3D"   -> (1, 128, 128, 96)
    mode "2D"   -> (1, S, S)   axial slice กลาง
    mode "2.5D" -> (3, S, S)   [sagittal, coronal, axial]
    S = out_size หรือ BASE_PLANE (ขยายตอนเข้าโมเดลได้เลย เช่น out_size=224)
    """
    v = np.asarray(volume)
    if tuple(v.shape) != TARGET_SHAPE:
        raise ValueError(f"volume ต้องเป็น {TARGET_SHAPE} (ที่ล็อกแล้ว) แต่ได้ {v.shape}")
    size = out_size or BASE_PLANE
    if mode == "3D":
        if out_size is not None:
            raise ValueError("3D ใช้ขนาดที่ล็อก ไม่รับ out_size (ถ้าจะปรับ ให้ปรับในโมเดล)")
        return rotate_volume(v, angle)[None].astype(np.float32)
    sag, cor, axi = planes_of_rotated(v, angle)
    if mode == "2D":
        out = np.stack([_to_square(axi, size)])
    elif mode == "2.5D":
        out = np.stack([_to_square(sag, size), _to_square(cor, size), _to_square(axi, size)])
    else:
        raise ValueError(f"mode ต้องเป็น '3D', '2D' หรือ '2.5D' ไม่ใช่ {mode!r}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def describe():
    return (f"processing_lock {PREPROCESS_VERSION} | hash {preprocess_hash()} | target {TARGET_SHAPE} | "
            f"clip p{CLIP_PERCENTILES[0]}-{CLIP_PERCENTILES[1]} | order {RESIZE_ORDER} | "
            f"center (sag,cor,axi) = {CENTER}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="สร้าง cache volume ที่ล็อก จากรายชื่อใน split.json")
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--cache-dir", default=".")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    with open(a.split) as f:
        ids = list(json.load(f)["labels"].keys())
    print(describe())
    print("สร้าง cache:", build_cache(ids, a.cache_dir, a.data_root, a.overwrite))
