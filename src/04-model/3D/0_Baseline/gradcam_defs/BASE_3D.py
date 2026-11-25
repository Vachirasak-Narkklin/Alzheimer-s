import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

with open(os.path.join(SHARED_DIR, "split.json")) as f:
    split = json.load(f)

labels = split["labels"]          # รหัสคน -> 0 (Normal) หรือ 1 (AD)
train_ids = split["train"]
val_ids = split["val"]
test_ids = split["test"]

print("Train      :", len(train_ids))
print("Validation :", len(val_ids))
print("Test       :", len(test_ids), "(ไม่ใช้จนกว่าจะถึงเซลล์สุดท้าย)")

# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

MODE = "3D"

# การเทรน
LR = 0.001
WEIGHT_DECAY = 0.0
DROPOUT = 0.0
LABEL_SMOOTHING = 0.0
BATCH_SIZE = 8
EPOCHS = 200
SEEDS = [1334, 42, 2026, 7, 2002]     # ล็อก 5 seed เท่ากันทุกโมเดล
NUM_WORKERS = 4          # ถ้า error เรื่อง shared memory ให้ใส่ 0
CHECKPOINT_EVERY = 10     # เซฟจุดต่อทุกกี่ epoch (เผื่อหลุดกลางทาง รันซ้ำจะเริ่มต่อให้)

# Augmentation
ROTATE_DEG = 10       # หมุนสุ่มใน ±10 องศา
GAMMA_D = 0.2         # gamma สุ่มใน (1 - 0.2, 1 + 0.2)

# ผลลัพธ์เก็บที่นี่ (โฟลเดอร์ย่อยตามชื่อสูตร) ถ้ามีผลเดิมอยู่แล้วจะถูกนำมาใช้ ไม่เทรนซ้ำ
EXP_DIR = "runs_baseline_3D"
NAME = "bn"
RUN_DIR = os.path.join(EXP_DIR, NAME)
for sub in ("results", "models", "figures"):
    os.makedirs(os.path.join(RUN_DIR, sub), exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

try:
    PL.load_locked(val_ids, CACHE_DIR, verbose=False)
except FileNotFoundError:
    print("ยังไม่มี cache กำลังสร้าง (ครั้งแรกใช้เวลาสักพัก)")
    PL.build_cache(train_ids + val_ids + test_ids, cache_dir=CACHE_DIR)


MAKE_KW = {}      # 3D ใช้ volume เต็ม ไม่ต้องกำหนดขนาดสไลซ์


def adjust_gamma(x, seed, epoch, sid, d):
    # ยกกำลังภาพด้วย gamma ที่สุ่มใน (1-d, 1+d) หนึ่งค่าต่อคนต่อ epoch (0 ยังเป็น 0, 1 ยังเป็น 1)
    g = np.random.default_rng([seed, epoch, sid, 4]).uniform(1 - d, 1 + d)
    return (np.clip(x, 0, 1) ** g).astype(np.float32)


class MRIDataset(Dataset):
    # train=True: หมุน + gamma / train=False: ไม่ทำอะไรกับภาพ
    def __init__(self, ids, train):
        self.ids = list(ids)
        self.volumes = PL.load_locked(self.ids, CACHE_DIR, verbose=False)
        self.y = [labels[s] for s in self.ids]
        self.train = train
        self.seed = 1334      # ตั้งใหม่ก่อนเทรนแต่ละ seed
        self.epoch = 1        # ตั้งใหม่ทุก epoch

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = int(self.ids[i])
        angle = 0.0
        if self.train:
            angle = AL.angle_for(self.seed, self.epoch, sid, ROTATE_DEG)
        x = PL.make_input(self.volumes[i], MODE, angle, **MAKE_KW)
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.train:
            x = adjust_gamma(x, self.seed, self.epoch, sid, GAMMA_D)
        return torch.from_numpy(x), self.y[i]


pass  # (ไม่โหลดชุด train)
val_data = MRIDataset(val_ids, train=False)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

IN_SHAPE = tuple(val_data[0][0].shape)
print("อินพุตต่อคน:", IN_SHAPE)

# ---------------- (เซลล์ถัดไปจากโน้ตบุ๊กเดิม) ----------------

class SimpleCNN(nn.Module):
    # Conv 3 ชั้น (kernel 3x3x3 + BatchNorm) -> Global Average Pooling -> Linear 2 คลาส (Normal / AD)
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=False),     # ชั้น 1: 1 -> 8 channel  (128x128x96 -> 64x64x48 หลัง MaxPool)
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1, bias=False),    # ชั้น 2: 8 -> 16 channel  (64x64x48 -> 32x32x24)
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),   # ชั้น 3: 16 -> 32 channel (32x32x24 -> 16x16x12)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)                  # เฉลี่ยทั้งก้อน 16x16x12 เหลือ 1 ค่าต่อ channel -> 32 ค่า
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 2)                          # 32 ฟีเจอร์ -> 2 คลาส

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(self.gap(x))
        return self.fc(x)


# ดูรูปร่างข้อมูลในแต่ละชั้น และจำนวน parameter
_m = SimpleCNN().eval()
_x = torch.zeros(1, *IN_SHAPE)
print("input   ", tuple(_x.shape[1:]))
with torch.no_grad():
    k = 0
    for layer in _m.features:
        _x = layer(_x)
        if isinstance(layer, nn.MaxPool3d):
            k += 1
            print(f"layer {k}  ", tuple(_x.shape[1:]))
print("จำนวน parameter:", f"{sum(q.numel() for q in _m.parameters()):,}")
del _m, _x
