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
import timm
# โฟลเดอร์ที่มี split.json / processing_lock.py / augmentation_lock.py (แก้ให้ตรงกับเครื่อง)
SHARED_DIR = "/home/jovyan/Alzheimer_Project/src/00-shared"
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

MODE = "2D"            # "3D" / "2.5D" / "2D"
import itertools

BATCH_SIZE = 8
EPOCHS = 200
SEEDS = [1334, 42, 2026, 7, 2002]     # ล็อก 5 seed เท่ากันทุกโมเดล
NUM_WORKERS = 4          # ถ้า error เรื่อง shared memory ให้ใส่ 0
CHECKPOINT_EVERY = 10      # เซฟจุดต่อทุกกี่ epoch (ไว้ให้เริ่มต่อได้ถ้าหลุดกลางทาง)

# augmentation ที่ล็อกแล้ว (ใช้เหมือนกันทุกโมเดล): หมุน ±10, gamma d = 0.2, ไม่มี noise, ทำกับทุกภาพทุก epoch
LOCKED_AUG = {"ROTATE_DEG": 10, "GAMMA_D": 0.2, "NOISE_STD": 0.0, "P_ROTATE": 1.0, "P_GAMMA": 1.0}
NOISE_TYPE = "gaussian"     # ไม่ได้ใช้ (noise ปิด) แต่โค้ดเดียวกับไฟล์อื่นต้องมี

# โครงสร้างที่เลือกแล้ว: pretrained + stem kernel 3 + full fine-tuning (เทรนทุกชั้น)
BASE = {"PRETRAINED": True, "STEM_KERNEL": 3, "FT": "full",
        "OPT": "adamw", "LR": 0.001, "WD": 0.0, "DROPOUT": 0.0, "LS": 0.0}

def mk(**kw):
    return {**LOCKED_AUG, **BASE, **kw}

# ---------- แกนของ grid ----------
GRID_LR = [0.001, 0.01, 0.0001]
GRID_OPT_WD = [("adamw", 0.0), ("adam", 0.0001), ("adam", 0.001), ("adam", 0.01),
               ("adamw", 0.0001), ("adamw", 0.001), ("adamw", 0.01)]      # Adam กับ AdamW ที่ wd = 0 เหมือนกัน จึงใส่ชุดเดียว
GRID_DROPOUT = [0.0, 0.1, 0.3, 0.5]
GRID_LS = [0.0, 0.05, 0.1, 0.15, 0.2]

def grid_name(lr, opt, wd, do, ls):
    if (lr, opt, wd, do, ls) == (0.001, "adamw", 0.0, 0.0, 0.0):
        return "control"                                   # ชุดที่ทุกแกนเป็นค่าเริ่มต้น = ที่รันไปแล้ว
    o = "wd0" if wd == 0 else f"{opt}{wd}"
    return f"lr{lr}_{o}_do{do}_ls{ls}"

EXPERIMENTS = {}
for lr, (opt, wd), do, ls in itertools.product(GRID_LR, GRID_OPT_WD, GRID_DROPOUT, GRID_LS):
    EXPERIMENTS[grid_name(lr, opt, wd, do, ls)] = mk(LR=lr, OPT=opt, WD=wd, DROPOUT=do, LS=ls)
print("จำนวนชุดทั้ง grid:", len(EXPERIMENTS))

# ---------- ชุดที่จะรัน: top-5 จาก grid + control ----------
TOP5 = [
    "lr0.01_wd0_do0.0_ls0.0",
    "lr0.01_adamw0.0001_do0.3_ls0.05",
    "lr0.01_adamw0.0001_do0.1_ls0.05",
    "lr0.01_adamw0.0001_do0.1_ls0.2",
    "lr0.01_adamw0.001_do0.0_ls0.1"
]
LR_ONLY_REF = None     # ไม่ใช้ตัวเทียบ LR-only

RUN_LIST = ["control"] + TOP5
for n in RUN_LIST:
    assert n in EXPERIMENTS, f"ไม่มีชื่อนี้ใน grid: {n}"

EXP_DIR = "runs_grid50_2D"                 # ผลทุกชุดเก็บที่นี่ (แยกโฟลเดอร์ตามชื่อ)
# control ตั้งค่าเหมือน runs_resnet50_2D/pre_stem_3 ทุกอย่าง: คัดลอกผลมาใช้ ไม่เทรนซ้ำ
REUSE_FROM = {"control": "runs_resnet50_2D/pre_stem_3"}
todo = [n for n in RUN_LIST for s in SEEDS if not os.path.exists(os.path.join(EXP_DIR, n, "results", f"seed{s}.json"))]
print(f"RUN_LIST {len(RUN_LIST)} ชุด | ยังไม่มีผล {len(todo)} รัน | ประมาณ {len(todo) * 4.0 / 60:.1f} ชั่วโมง (ถ้ารันละ ~4 นาที)")
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

class ResNet50Timm(nn.Module):
    # ResNet50 จาก timm: pretrained ImageNet (ถ้า pretrained = True) หรือสร้างจากศูนย์ แล้วเทรนต่อทั้งโมเดล
    def __init__(self, in_ch=1, n_classes=2, dropout=0.0, kernel=7, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=pretrained, in_chans=in_ch, num_classes=0)
        if kernel != 7:
            resize_stem(self.backbone, in_ch, kernel, pretrained)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.backbone(x)))


def resize_stem(backbone, in_ch, kernel, from_pretrained):
    # stem: ResNet50 ต้นฉบับใช้ kernel 7 stride 2 ถ้าเปลี่ยนขนาด kernel:
    #   pretrained -> ย่อ weight pretrained เดิมมาเป็นค่าเริ่มต้น (bilinear)
    #   สร้างจากศูนย์ -> ตั้งค่าเริ่มต้นแบบ ResNet50 ต้นฉบับ (kaiming normal) ไม่ย่อ weight สุ่ม
    old = backbone.conv1
    new = nn.Conv2d(in_ch, 64, kernel_size=kernel, stride=2, padding=kernel // 2, bias=False)
    if from_pretrained:
        with torch.no_grad():
            new.weight.copy_(F.interpolate(old.weight, size=(kernel, kernel), mode="bilinear", align_corners=False))
    else:
        nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
    backbone.conv1 = new


def make_model(p):
    # p = ค่าตั้งของการทดลอง (EXPERIMENTS[ชื่อ])
    in_ch = 1
    return ResNet50Timm(in_ch=in_ch, n_classes=2, dropout=p["DROPOUT"], kernel=p["STEM_KERNEL"], pretrained=p["PRETRAINED"])


# ---------- วิธี fine-tune: เลือกว่าชั้นไหนเทรน / freeze ในแต่ละ epoch ----------
STAGE_ORDER_DEEP_FIRST = ["head", "layer4", "layer3", "layer2", "layer1", "stem"]    # ลำดับการปลดของ gradual (ลึกสุดก่อน)


def stage_modules(model):
    # ชั้นของ ResNet50 (timm) -> โมดูลที่อยู่ในชั้นนั้น
    b = model.backbone
    return {"stem": [b.conv1, b.bn1], "layer1": [b.layer1], "layer2": [b.layer2],
            "layer3": [b.layer3], "layer4": [b.layer4], "head": [model.fc]}


def trainable_stages(p, epoch):
    # ชั้นที่เทรนใน epoch นี้ (epoch เริ่มนับที่ 1)
    ft = p.get("FT", "full")
    if ft in ("full", "layerwise"):
        return set(STAGE_ORDER_DEEP_FIRST)
    if ft == "probe":
        return {"head"}
    if ft == "freeze_early":
        return {"layer3", "layer4", "head"}
    if ft == "gradual":
        n = 1 + (epoch - 1) // p["UNFREEZE_EVERY"]
        return set(STAGE_ORDER_DEEP_FIRST[:n])
    raise ValueError(ft)


def apply_trainable(model, trainable):
    # ชั้นที่ freeze: ไม่อัปเดตน้ำหนัก และ BatchNorm อยู่โหมด eval (ไม่อัปเดตค่าสถิติ) / ชั้นที่เทรน: โหมด train ปกติ
    for name, mods in stage_modules(model).items():
        on = name in trainable
        for m in mods:
            for q in m.parameters():
                q.requires_grad_(on)
            m.train(on)


def make_optimizer(model, p):
    # Adam: weight decay แบบ L2 รวมใน gradient / AdamW: weight decay แยกจาก gradient (ที่ wd = 0 สองตัวนี้เหมือนกัน)
    opt_cls = optim.AdamW if p["OPT"] == "adamw" else optim.Adam
    return opt_cls(model.parameters(), lr=p["LR"], weight_decay=p["WD"])
