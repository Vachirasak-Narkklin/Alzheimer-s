import os
import sys
import json
import time
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# โฟลเดอร์ที่มี split.json / processing_lock.py / augmentation_lock.py (แก้ให้ตรงกับเครื่อง)
SHARED_DIR = "/home/jovyan/Alzheimer's/src/00-shared"
CACHE_DIR = os.path.join(SHARED_DIR, "locked_cache")
sys.path.insert(0, SHARED_DIR)
import processing_lock as PL
import augmentation_lock as AL

print(PL.describe())

# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

with open(os.path.join(SHARED_DIR, "split.json")) as f:
    split = json.load(f)

labels = split["labels"]
train_ids = split["train"]
val_ids = split["val"]
test_ids = split["test"]

print("Train      :", len(train_ids))
print("Validation :", len(val_ids))
print("Test       :", len(test_ids), "(ไฟล์นี้ไม่ใช้ test จนกว่าจะถึงเซลล์สุดท้าย)")

# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

MODE = "3D"            # ไฟล์นี้ใช้ 3D เท่านั้น
PRETRAINED = False       # 3D สร้างจากศูนย์ ไม่มี pretrained

BATCH_SIZE = 8
EPOCHS = 200
SEEDS = [1334, 42, 2026, 7, 2002]     # ล็อก 5 seed เท่ากันทุกโมเดล
NUM_WORKERS = 4          # ถ้า error เรื่อง shared memory ให้ใส่ 0
CHECKPOINT_EVERY = 10      # เซฟจุดต่อทุกกี่ epoch (ไว้ให้เริ่มต่อได้ถ้าหลุดกลางทาง)

# augmentation ที่ล็อกแล้ว (ใช้เหมือนกันทุกโมเดล)
LOCKED_AUG = {"ROTATE_DEG": 10, "GAMMA_D": 0.2, "NOISE_STD": 0.0, "P_ROTATE": 1.0, "P_GAMMA": 1.0}
NOISE_TYPE = "gaussian"

import itertools

BASE = {"STEM_KERNEL": 7, "OPT": "adam", "LR": 0.001, "WD": 0.0, "DROPOUT": 0.0, "LS": 0.0}

def mk(**kw):
    return {**LOCKED_AUG, **BASE, **kw}

# ---------- แกนของ grid ----------
GRID_LR = [0.001, 0.01, 0.0001]
GRID_OPT_WD = [("adam", 0.0), ("adam", 0.0001), ("adam", 0.001), ("adam", 0.01),
               ("adamw", 0.0001), ("adamw", 0.001), ("adamw", 0.01)]      # Adam กับ AdamW ที่ wd = 0 เหมือนกัน จึงใส่ชุดเดียว
GRID_DROPOUT = [0.0, 0.1, 0.3, 0.5]
GRID_LS = [0.0, 0.05, 0.1, 0.15, 0.2]

def grid_name(lr, opt, wd, do, ls):
    if (lr, opt, wd, do, ls) == (0.001, "adam", 0.0, 0.0, 0.0):
        return "stem_7"                                   # ชุดที่ทุกแกนเป็นค่าเริ่มต้น = ที่รันไปแล้ว
    o = "wd0" if wd == 0 else f"{opt}{wd}"
    return f"lr{lr}_{o}_do{do}_ls{ls}"

EXPERIMENTS = {}
for lr, (opt, wd), do, ls in itertools.product(GRID_LR, GRID_OPT_WD, GRID_DROPOUT, GRID_LS):
    EXPERIMENTS[grid_name(lr, opt, wd, do, ls)] = mk(LR=lr, OPT=opt, WD=wd, DROPOUT=do, LS=ls)
print("จำนวนชุดทั้ง grid:", len(EXPERIMENTS))

# ---------- ชุดที่จะรัน: top-5 จาก grid + control ----------
TOP5 = [
    "lr0.001_adamw0.0001_do0.5_ls0.0",
    "lr0.001_adamw0.0001_do0.0_ls0.15",
    "lr0.01_adam0.001_do0.0_ls0.15",
    "lr0.01_adamw0.001_do0.0_ls0.2",
    "lr0.01_adamw0.01_do0.5_ls0.05"
]

LR_ONLY_REF = None     # ไม่ใช้ตัวเทียบ LR-only

RUN_LIST = ["stem_7"] + TOP5
for n in RUN_LIST:
    assert n in EXPERIMENTS, f"ไม่มีชื่อนี้ใน grid: {n}"
todo = [n for n in RUN_LIST for s in SEEDS if not os.path.exists(os.path.join("runs_resnet50_3D", n, "results", f"seed{s}.json"))]
print(f"RUN_LIST {len(RUN_LIST)} ชุด | ยังไม่มีผล {len(todo)} รัน | ประมาณ {len(todo) * 17.5 / 60:.1f} ชั่วโมง")

EXP_DIR = "runs_resnet50_3D"                 # โฟลเดอร์เดียวกับไฟล์ ResNet50 3D เดิม (ใช้ผล stem_7 ซ้ำ ต้องเป็นโฟลเดอร์เดียวกัน)
REUSE_FROM = {}                             # ไม่คัดลอกผลจากไฟล์อื่น
os.makedirs(EXP_DIR, exist_ok=True)


# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ---------- ดูความคืบหน้า: อ่านจากไฟล์ progress ของแต่ละการทดลอง (รันเซลล์นี้ซ้ำได้ตลอด แม้เทรนอยู่) ----------
def show_progress():
    rows = []
    for name in RUN_LIST:
        for seed in SEEDS:
            p = os.path.join(EXP_DIR, name, "progress", f"seed{seed}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                rows.append({"การทดลอง": name, "seed": seed, "สถานะ": d["status"],
                             "epoch": f"{d['epoch']}/{d['epochs']}", "best_epoch": d["best_epoch"],
                             "best combo": round(d["best_combo"], 4), "val AUC ล่าสุด": round(d["last_val_auc"], 4),
                             "นาทีที่ใช้": d["minutes"], "อัปเดตล่าสุด": d["updated"]})
            else:
                rows.append({"การทดลอง": name, "seed": seed, "สถานะ": "ยังไม่เริ่ม", "epoch": "-", "best_epoch": "-",
                             "best combo": "-", "val AUC ล่าสุด": "-", "นาทีที่ใช้": "-", "อัปเดตล่าสุด": "-"})
    table = pd.DataFrame(rows)
    done = int((table["สถานะ"] == "เสร็จแล้ว").sum())
    print(f"เสร็จแล้ว {done} / {len(table)} รัน")
    print(table["สถานะ"].value_counts().to_string())
    notdone = table[table["สถานะ"] != "เสร็จแล้ว"]
    display(notdone.head(15))                         # ที่ยังไม่เสร็จ (กำลังรัน / ยังไม่เริ่ม) แสดง 15 แถวแรก
    display(table[table["สถานะ"] == "เสร็จแล้ว"].tail(5))   # ที่เพิ่งเสร็จล่าสุด

pass

# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

try:
    PL.load_locked(val_ids, CACHE_DIR, verbose=False)
except FileNotFoundError:
    print("ยังไม่มี cache กำลังสร้าง (ครั้งแรกใช้เวลาสักพัก)")
    PL.build_cache(train_ids + val_ids + test_ids, cache_dir=CACHE_DIR)

if MODE == "3D":
    make_kw = {}
else:
    make_kw = {"out_size": 128}


def augment_intensity(x, seed, epoch, sid, gamma_d, noise_std, noise_type):
    """Gamma และ Noise (ทำหลังหมุนและตัด slice แล้ว) ค่าสุ่มผูกกับ (seed, epoch, คน) ทำซ้ำได้ทุกรอบ"""
    if gamma_d == 0 and noise_std == 0:
        return x                                    # ไม่ทำอะไร: ภาพคงเดิมทุกค่า

    x = np.clip(x, 0, 1)                            # ภาพ normalize เป็น [0, 1] แล้ว

    # ---- Gamma: x ยกกำลัง g โดย g สุ่มใน (1 - d, 1 + d) ค่าเดียวต่อคนต่อ epoch ----
    if gamma_d > 0:
        g = np.random.default_rng([seed, epoch, sid, 4]).uniform(1 - gamma_d, 1 + gamma_d)
        x = x ** g                                  # 0 ยังเป็น 0, 1 ยังเป็น 1

    # ---- Noise: เติมเฉพาะในเนื้อสมอง (voxel ที่ > 0) พื้นหลังที่ตัดทิ้งแล้วไม่เติม ----
    if noise_std > 0:
        rng = np.random.default_rng([seed, epoch, sid, 6])
        brain = x > 0
        n1 = rng.normal(0, noise_std, x.shape)
        if noise_type == "rician":
            n2 = rng.normal(0, noise_std, x.shape)
            noisy = np.sqrt((x + n1) ** 2 + n2 ** 2)    # Rician: ขนาดของค่าเชิงซ้อนที่มี noise
        else:
            noisy = x + n1                              # Gaussian
        x = np.where(brain, np.clip(noisy, 0, 1), x)

    return x.astype(np.float32)


class MRIDataset(Dataset):
    # train=True: ใช้ augmentation ตามค่าที่ตั้งไว้ในตัว dataset / train=False: ไม่ทำอะไรกับภาพ
    def __init__(self, ids, train):
        self.ids = list(ids)
        self.volumes = PL.load_locked(self.ids, CACHE_DIR, verbose=False)
        self.y = [labels[s] for s in self.ids]
        self.train = train

        # ค่า augmentation (ตั้งใหม่ทุกครั้งก่อนเทรนแต่ละการทดลอง)
        self.seed = 1334
        self.epoch = 1
        self.rotate_deg = 0
        self.gamma_d = 0.0
        self.noise_std = 0.0
        self.noise_type = "gaussian"
        self.p_rotate = 1.0       # ความน่าจะเป็นที่คนนี้ใน epoch นี้จะถูกหมุน (1.0 = ทุกครั้ง)
        self.p_gamma = 1.0        # ความน่าจะเป็นที่จะถูกปรับ gamma (สุ่มแยกจากการหมุน)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = int(self.ids[i])

        # "โยนเหรียญ" ว่าครั้งนี้จะทำ augmentation ไหม ผูกกับ (seed, epoch, คน) ทำซ้ำได้ ไม่ขึ้นกับโมเดล
        do_rotate = do_gamma = True
        if self.train and self.p_rotate < 1:
            do_rotate = np.random.default_rng([self.seed, self.epoch, sid, 8]).random() < self.p_rotate
        if self.train and self.p_gamma < 1:
            do_gamma = np.random.default_rng([self.seed, self.epoch, sid, 9]).random() < self.p_gamma

        angle = 0.0
        if self.train and self.rotate_deg > 0 and do_rotate:
            angle = AL.angle_for(self.seed, self.epoch, sid, self.rotate_deg)

        x = PL.make_input(self.volumes[i], MODE, angle, **make_kw)
        x = np.ascontiguousarray(x, dtype=np.float32)

        if self.train:
            x = augment_intensity(x, self.seed, self.epoch, sid, self.gamma_d if do_gamma else 0.0, self.noise_std, self.noise_type)

        return torch.from_numpy(x), self.y[i]


pass  # (ไม่โหลดชุด train)
val_data = MRIDataset(val_ids, train=False)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

IN_SHAPE = tuple(val_data[0][0].shape)
print("อินพุตต่อคน:", IN_SHAPE)

# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

def make_norm(kind, ch, groups):
    if kind == "group":
        return nn.GroupNorm(groups, ch)
    return nn.BatchNorm3d(ch)


class Bottleneck3D(nn.Module):
    # บล็อก Bottleneck ของ ResNet50: Conv1x1-BN-ReLU → Conv3x3-BN-ReLU → Conv1x1-BN แล้วบวกค่าเดิมกลับ (residual) ก่อน ReLU
    # channel ออก = width x 4, stride 2 (ถ้ามี) อยู่ที่ conv 3x3x3
    expansion = 4

    def __init__(self, in_ch, width, stride=1, norm="batch", groups=8):
        super().__init__()
        out_ch = width * self.expansion
        self.conv1 = nn.Conv3d(in_ch, width, kernel_size=1, bias=False)
        self.bn1 = make_norm(norm, width, groups)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = make_norm(norm, width, groups)
        self.conv3 = nn.Conv3d(width, out_ch, kernel_size=1, bias=False)
        self.bn3 = make_norm(norm, out_ch, groups)

        # ถ้าขนาด/จำนวนช่องเปลี่ยน ต้องปรับทางลัด (shortcut) ให้ตรงกันก่อนบวก
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                make_norm(norm, out_ch, groups))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = torch.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return torch.relu(y + self.shortcut(x))


class ResNet50_3D(nn.Module):
    # ResNet50 แบบ 3 มิติ สร้างจากศูนย์ (ไม่มี pretrained)
    def __init__(self, in_ch=1, n_classes=2, dropout=0.0, kernel=7, norm="batch", groups=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=kernel, stride=2, padding=kernel // 2, bias=False),
            make_norm(norm, 64, groups),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        def stage(in_ch, width, n_blocks, stride):
            blocks = [Bottleneck3D(in_ch, width, stride, norm, groups)]
            blocks += [Bottleneck3D(width * 4, width, 1, norm, groups) for _ in range(n_blocks - 1)]
            return nn.Sequential(*blocks)

        self.layer1 = stage(64, 64, 3, 1)
        self.layer2 = stage(256, 128, 4, 2)
        self.layer3 = stage(512, 256, 6, 2)
        self.layer4 = stage(1024, 512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, n_classes)

        # ตั้งค่าเริ่มต้นแบบ ResNet ต้นฉบับ (kaiming normal, BN weight 1 / bias 0)
        for mod in self.modules():
            if isinstance(mod, nn.Conv3d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def make_model(stem_kernel, dropout=0.0):
    return ResNet50_3D(in_ch=1, n_classes=2, dropout=dropout, kernel=stem_kernel)
