# กราฟต่อ epoch ของแต่ละรัน (อ่านจากไฟล์ผลที่บันทึกไว้ ไม่ต้องเทรนซ้ำ)
def plot_run(model_name, seed):
    r = json.load(open(os.path.join(OUT_DIR, f"{model_name}_seed{seed}.json")))
    h = r["history"]
    epochs = range(1, len(h["train_loss"]) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, h["train_loss"], label="train_loss")
    plt.plot(epochs, h["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss {model_name} (seed {seed})")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, h["val_auc"])
    plt.xlabel("epoch")
    plt.ylabel("val_auc")
    plt.title("Validation AUC")

    plt.subplot(1, 3, 3)
    plt.plot(epochs, h["val_f1"])
    plt.xlabel("epoch")
    plt.ylabel("val_f1")
    plt.title("Validation F1")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"runcurves_{model_name}_seed{seed}.png"), dpi=150)
    plt.show()


PLOT_SCREENING_TOO = False      # True = วาดกราฟของทุกรันที่คัดกรอง (seed 1334) ด้วย

try:
    plot_models = [CHOSEN]      # ค่าที่เลือก
except NameError:
    # ไฟล์ที่ไม่มีการเลือก (เช่น ResNet18) วาดทุกโมเดลที่มีผลอยู่
    plot_models = sorted({os.path.basename(f).split("_seed")[0] for f in glob.glob(os.path.join(OUT_DIR, "*_seed*.json"))})

for name in plot_models:
    for s in SEEDS:
        if os.path.exists(os.path.join(OUT_DIR, f"{name}_seed{s}.json")):
            plot_run(name, s)

if PLOT_SCREENING_TOO:
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*_seed1334.json"))):
        name = os.path.basename(f).split("_seed")[0]
        if name not in plot_models:
            plot_run(name, 1334)
