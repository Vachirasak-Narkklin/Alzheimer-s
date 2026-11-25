# วางเป็นเซลล์เดียวในโน้ตบุ๊กทุกโมเดล แทนเซลล์ train_test_split เดิม
# ต้องมี df ที่มีคอลัมน์ subject_id และ label อยู่ก่อน
import json

SPLIT_PATH = "split.json"   # ชี้ไปที่ไฟล์กลาง (ทุกโมเดลใช้ไฟล์เดียวกัน)

# 1) อ่านรายชื่อจากไฟล์
with open(SPLIT_PATH) as f:
    split = json.load(f)

train_ids = set(split["train"])
val_ids = set(split["val"])
test_ids = set(split["test"])
file_labels = split["labels"]          # {"0001": 0, "0003": 1, ...}

# 2) ตรวจว่าไม่มีใครอยู่ซ้ำกันระหว่างชุด (ทีละคู่)
assert len(train_ids & val_ids) == 0, "มีคนอยู่ทั้ง train และ val"
assert len(train_ids & test_ids) == 0, "มีคนอยู่ทั้ง train และ test"
assert len(val_ids & test_ids) == 0, "มีคนอยู่ทั้ง val และ test"

# 3) ตรวจว่าคนใน df กับคนในไฟล์เป็นกลุ่มเดียวกันเป๊ะ
ids_in_file = train_ids | val_ids | test_ids
ids_in_df = set(df["subject_id"])
only_in_df = ids_in_df - ids_in_file       # มีใน df แต่ไม่มีในไฟล์
only_in_file = ids_in_file - ids_in_df     # มีในไฟล์ แต่ไม่มีใน df
assert not only_in_df and not only_in_file, (
    f"รายชื่อไม่ตรงกัน | มีใน df แต่ไม่มีในไฟล์: {sorted(only_in_df)[:10]} "
    f"| มีในไฟล์แต่ไม่มีใน df: {sorted(only_in_file)[:10]}"
)

# 4) ตรวจว่า label ใน df ตรงกับที่เก็บไว้ในไฟล์
df_labels = df.groupby("subject_id")["label"].first()
wrong = [s for s in ids_in_file if int(df_labels[s]) != int(file_labels[s])]
assert not wrong, f"label ไม่ตรงกับไฟล์ {len(wrong)} คน เช่น {wrong[:10]}"

# 5) แบ่ง df ตามรายชื่อ
train_df = df[df["subject_id"].isin(train_ids)]
val_df = df[df["subject_id"].isin(val_ids)]
test_df = df[df["subject_id"].isin(test_ids)]

# 6) แสดงสรุป
print("โหลด", SPLIT_PATH)
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    n_ad = int((part["label"] == 1).sum())
    n_normal = len(part) - n_ad
    print(f"  {name:5s} รวม={len(part)}  ปกติ={n_normal}  อัลไซเมอร์={n_ad}  ({100 * n_ad / len(part):.1f}% อัลไซเมอร์)")
