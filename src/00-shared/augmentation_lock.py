"""ล็อก augmentation ให้ทุกโมเดลใช้ร่วมกัน (2D / 2.5D / 3D)

ค่าที่ล็อก (ได้จากการทดลองทีละตัวบน val, ResNet18 2D/2.5D/3D, 3 seed):
  - หมุน  ±10 องศา          (ROTATE_DEG)
  - gamma d = 0.2 (γ 0.8-1.2) (GAMMA_D)
  - noise  ไม่ใช้ (std = 0)   (NOISE_STD) ทดลอง Gaussian แล้วไม่ช่วยอย่างชัดเจน

ค่าสุ่มทั้งหมดถูกกำหนดจาก (seed, epoch, subject_id) เท่านั้น ไม่ขึ้นกับโมเดล ลำดับ shuffle หรือตัวสุ่มส่วนกลาง
วางไฟล์นี้ไว้ที่เดียว (เช่น 00_shared) แล้วให้ทุกโมเดล import ไปใช้ จะได้ค่าเดียวกันเป๊ะ

LOCK_MODE:
  "per_epoch"   (แบบ A) คนเดิมได้มุมต่างกันในแต่ละ epoch แต่เหมือนกันทุกโมเดล
  "per_subject" (แบบ B) คนเดิมได้มุมเดียวตลอดการเทรน และเหมือนกันทุกโมเดล
"""
import numpy as np

# ---- ค่าที่ล็อก ----
ROTATE_DEG = 10
GAMMA_D = 0.2
NOISE_STD = 0.0
NOISE_TYPE = "gaussian"


def angle_for(seed, epoch, subject_id, rotate_deg=ROTATE_DEG, mode="per_epoch"):
    if mode == "per_epoch":
        key = [int(seed), int(epoch), int(subject_id)]
    elif mode == "per_subject":
        key = [int(seed), int(subject_id)]
    else:
        raise ValueError(f"mode ต้องเป็น 'per_epoch' หรือ 'per_subject' ไม่ใช่ {mode!r}")
    return float(np.random.default_rng(key).uniform(-rotate_deg, rotate_deg))


def augment_intensity(x, seed, epoch, sid, gamma_d=GAMMA_D, noise_std=NOISE_STD, noise_type=NOISE_TYPE):
    """ปรับความสว่าง (gamma) และเติม noise หลังหมุนแล้ว x ต้องอยู่ใน [0, 1]
    gamma: x ** γ, γ สุ่มใน (1 - d, 1 + d) ค่าเดียวต่อคนต่อ epoch
    noise: เติมเฉพาะในเนื้อสมอง (x > 0) แล้วบีบกลับเป็น [0, 1]
    """
    if gamma_d == 0 and noise_std == 0:
        return x
    x = np.clip(x, 0, 1)
    if gamma_d > 0:
        g = np.random.default_rng([int(seed), int(epoch), int(sid), 4]).uniform(1 - gamma_d, 1 + gamma_d)
        x = x ** g
    if noise_std > 0:
        rng = np.random.default_rng([int(seed), int(epoch), int(sid), 6])
        brain = x > 0
        n1 = rng.normal(0, noise_std, x.shape)
        if noise_type == "rician":
            n2 = rng.normal(0, noise_std, x.shape)
            noisy = np.sqrt((x + n1) ** 2 + n2 ** 2)
        else:
            noisy = x + n1
        x = np.where(brain, np.clip(noisy, 0, 1), x)
    return x.astype(np.float32)
