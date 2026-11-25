# การวิเคราะห์สภาวะอัลไซเมอร์จากภาพถ่ายคลื่นแม่เหล็กไฟฟ้าด้วยการเรียนรู้เชิงลึก

โครงงานปริญญานิพนธ์: จำแนกผู้เป็น / ไม่เป็นโรคอัลไซเมอร์ (2 กลุ่ม) จากภาพ MRI สมอง 3 มิติของชุดข้อมูล OASIS-1
ด้วยโมเดล 9 กลุ่ม (Baseline CNN, ResNet-18, ResNet-50 × 2D, 2.5D, 3D) ปล่อยให้โมเดลเรียนรู้ตำแหน่งสำคัญเองจากสมองทั้งก้อน
แล้วใช้ Grad-CAM ตรวจว่าโมเดลมองบริเวณไหน

- ป้ายกำกับจากค่า CDR: `CDR = 0` ปกติ, `CDR >= 0.5` อัลไซเมอร์, ไม่มีค่า CDR ตัดทิ้ง
- ข้อมูล 234 คน แบ่งระดับบุคคล: ฝึก 163 / ตรวจสอบ 35 / ทดสอบ 36 (`src/00-shared/split.json`)
- ล็อกค่าเตรียมภาพและการเพิ่มความหลากหลายของข้อมูลร่วมกันทุกกลุ่ม เลือกโมเดลด้วยชุดตรวจสอบ และเปิดชุดทดสอบครั้งเดียว

## ข้อมูล (ไม่รวมใน repository นี้)

ภาพ MRI **ไม่ได้อยู่ใน repository** ติดเงื่อนไขการใช้ข้อมูลของ OASIS ให้ดาวน์โหลดเองจากโครงการ OASIS
หลังลงทะเบียนและยอมรับเงื่อนไข: <https://sites.wustl.edu/oasisbrains/> (ชุด OASIS-1: Cross-sectional MRI Data)

อ้างอิง: Marcus, D. S., et al. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in
young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498-1507.

ตั้งตำแหน่งข้อมูลด้วยตัวแปรสภาพแวดล้อม `OASIS_ROOT` (ค่าเริ่มต้นอยู่ใน `src/00-shared/processing_lock.py`)
และแก้ `SHARED_DIR` ในเซลล์ตั้งค่าของแต่ละโน้ตบุ๊กให้ตรงกับตำแหน่งจริงบนเครื่อง

## โครงสร้างโฟลเดอร์

```text
Alzheimer's/
|-- Readme.txt                      (ไฟล์นี้)
`-- src/
    |-- 00-shared/                  ไฟล์ล็อกกลางที่ทุกโมเดลใช้ร่วมกัน
    |-- 01-data-review/             ตรวจและทำความเข้าใจข้อมูล
    |-- 02-data-leakage/            สร้างและตรวจการแบ่งชุดข้อมูล (กันข้อมูลรั่ว)
    |-- 03-augment/                 ทดลองเพิ่มความหลากหลายของข้อมูล (ResNet-18)
    |-- 04-model/                   โมเดลทั้ง 9 กลุ่ม แยกตามรูปแบบข้อมูล
        |-- 2D/    (Baseline, Resnet18, Resnet50)
        |-- 25D/   (Baseline, Resnet18, Resnet50)
        `-- 3D/    (0_Baseline, 1_Resnet18, 2_ResNet50, runcurves_cell.py)
    `-- results/                    ผลการทดลอง (json / csv) ของแต่ละกลุ่ม ดูหัวข้อ "ผลลัพธ์"
```

## รายละเอียดไฟล์

```text
[src/00-shared]  ไฟล์ล็อกกลาง
  split.json                     รายชื่อผู้เข้าร่วมในชุดฝึก / ตรวจสอบ / ทดสอบ (ใช้ทุกโมเดล)
  split_reference.json           ไฟล์อ้างอิงของการแบ่งข้อมูล ใช้เทียบตรวจ
  processing_lock.py             ล็อกการเตรียมภาพ (ขนาดภาพ 128 x 128 x 96, การตัดค่า
                                 ความเข้ม, การย่อภาพ, การสร้างอินพุต 2D / 2.5D / 3D จากปริมาตรที่ล็อก)
  augmentation_lock.py           ล็อกการเพิ่มความหลากหลายของข้อมูล:
                                 หมุน +-10 องศา, gamma d = 0.2, ไม่ใช้ noise,
                                 ไม่พลิกซ้ายขวา; ค่าสุ่มกำหนดจาก (seed, epoch, subject_id)
  load_locked_split_cell.py      เซลล์โค้ดอ่าน split.json ที่วางในโน้ตบุ๊กของทุกโมเดล
  03_Slice_Position_Check.ipynb  ตรวจตำแหน่งภาพตัด (x, y, z) เทียบกับฮิปโปแคมปัส
                                 บนปริมาตรที่ล็อกแล้ว
  04_Test_Processing_Lock_3D.ipynb  ทดสอบ processing_lock.py กับ OASIS-1 จริง
  processing_lock_test_report.json  รายงานผลการทดสอบข้างต้น
  fig_processing_lock_check.png     รูปประกอบผลการทดสอบข้างต้น

[src/01-data-review]  ตรวจข้อมูล
  00_Review_Data.ipynb           ตรวจตารางข้อมูล ตัดคนที่ไม่มีค่า CDR ออก ตั้ง label
  01_TargetShape_Experiment.ipynb  ทดลองเลือกขนาดภาพที่ใช้ (target shape)
  A_Check_OASIS_Hippocampus.ipynb  ตรวจข้อมูล OASIS-1 เทียบแผนที่สมองก่อนทำ Grad-CAM
  A_check_output/                ผลของไฟล์ข้างบน
      hippocampus_slice_check.csv   ตารางตรวจภาพตัดเทียบฮิปโปแคมปัส
      slice_vs_hippocampus.png      รูปภาพตัดเทียบฮิปโปแคมปัส
      qc_overlay.png                รูปตรวจการซ้อนทับ
  Check_Demographics_Split.ipynb ตรวจอายุและเพศของแต่ละชุดข้อมูล
  age_distribution_by_split.png  รูปการกระจายอายุตามชุดข้อมูล

[src/02-data-leakage]  การแบ่งชุดข้อมูลและการตรวจข้อมูลรั่ว
  create_and_test_split_json.ipynb     สร้าง split.json และทดสอบ
  preload_split_by_seed.ipynb          โหลดข้อมูลล่วงหน้าตามการแบ่งของแต่ละ seed
  preload_locked_split_by_seed.ipynb   เช่นเดียวกัน แต่ใช้การแบ่งที่ล็อกแล้ว
  preload_split_check.ipynb            ตรวจความถูกต้องของการแบ่ง
  preload_split_check_3modes.ipynb     ตรวจการแบ่งทั้ง 3 รูปแบบข้อมูล (2D / 2.5D / 3D)
  split_pairwise_seed_check.ipynb      ตรวจซ้อนทับของรายชื่อระหว่าง seed เป็นคู่ๆ

[src/03-augment]  ทดลองการเพิ่มความหลากหลายของข้อมูล (ใช้ ResNet-18 ทดสอบ)
  Aug_Rotation_ResNet18_{2D,25D,3D}.ipynb   ทดลองระดับการหมุน
  Aug_Gamma_ResNet18_{2D,25D,3D}.ipynb      ทดลองระดับ gamma
  Aug_Noise_ResNet18_{2D,25D,3D}.ipynb      ทดลองการเติม noise
  Aug_Prob_ResNet18_{2D,25D,3D}.ipynb       ทดลองความน่าจะเป็นของการเพิ่มความหลากหลาย
  ผลของโฟลเดอร์นี้นำไปกำหนดค่าใน 00-shared/augmentation_lock.py

[src/04-model]  โมเดลทั้ง 9 กลุ่ม
  ไฟล์หลักของแต่ละกลุ่ม:
  2D/Baseline/   Baseline_CNN3Layer_2D.ipynb
  2D/Resnet18/   Resnet18_2D.ipynb
  2D/Resnet50/   Resnet50_2D.ipynb
  25D/Baseline/  Baseline_CNN3Layer_25D.ipynb
  25D/Resnet18/  Resnet18_25D.ipynb
  25D/Resnet50/  Resnet50_25D.ipynb
  3D/0_Baseline/ Baseline_CNN3Layer_3D.ipynb
                 + Baseline_CNN3Layer_3D_BatchSize.ipynb   (ทดลองขนาดชุดย่อย)
                 + Baseline_CNN3Layer_3D_Width_3seed.ipynb (ทดลองความกว้าง รัน 3 seed)
  3D/1_Resnet18/ Resnet18_3D.ipynb
  3D/2_ResNet50/ Resnet50_3D.ipynb
  3D/runcurves_cell.py   เซลล์โค้ดวาดกราฟต่อ epoch ของแต่ละรัน (อ่านจากไฟล์ผลที่บันทึกไว้)

  ไฟล์ Baseline_CNN_{2D,25D,3D}.ipynb (ในโฟลเดอร์ Baseline ของแต่ละรูปแบบข้อมูล)
  เป็นการรันเพิ่มเติมของ Baseline โดยใช้โครงสร้างและจำนวนพารามิเตอร์เดียวกับ
  Baseline_CNN3Layer_* ไฟล์หลักที่ใช้อ้างอิงคือ Baseline_CNN3Layer_*

  ไฟล์เสริมที่อยู่ในโฟลเดอร์โมเดล:
  Check_Criteria_Baseline.ipynb   ตรวจเกณฑ์ตัดสินเลือกชุดค่า: ใช้วิธีเดียวกับ Baseline
                                  แต่ margin มาจากความแปรปรวนของโมเดลนั้นเอง
                                  (มีอยู่ในโฟลเดอร์ ResNet และ 3D/0_Baseline)
  Grad-CAM_All_9_Models.ipynb     สร้างแผนภาพความร้อน Grad-CAM ของทั้ง 9 กลุ่มและตรวจ
                                  ตำแหน่งเทียบฮิปโปแคมปัส (Harvard-Oxford atlas)
  gradcam_defs/*.py               นิยามโมเดล 9 กลุ่ม (BASE_/R18_/R50_ x 2D/25D/3D) ที่
                                  Grad-CAM โหลดน้ำหนักเข้าไป เนื้อหาเหมือนกันทุกสำเนา
                                  (อยู่ใน 3D/0_Baseline, 3D/1_Resnet18, 3D/2_ResNet50)
  หมายเหตุ: Grad-CAM_All_9_Models.ipynb มีสำเนาในหลายโฟลเดอร์ เพราะรันแยกตามกลุ่ม
  เนื้อหาโค้ดใช้ร่วมกัน ผลลัพธ์ที่ตกค้างในไฟล์จึงต่างกันตามกลุ่มที่รัน
```

## ลำดับการรัน

1. `01-data-review` ตรวจข้อมูล ตั้ง label
2. `02-data-leakage` สร้าง / ตรวจ `split.json`
3. `00-shared` ตรวจ `processing_lock.py` และตำแหน่งภาพตัด
4. `03-augment` ทดลองการเพิ่มความหลากหลายของข้อมูล (ได้ค่าใน `augmentation_lock.py`)
5. `04-model` เทรน / รันซ้ำแต่ละกลุ่ม
6. `04-model/.../Grad-CAM_All_9_Models.ipynb` อธิบายผลของโมเดล

## ผลลัพธ์

โฟลเดอร์ `src/results/` เก็บผลของแต่ละรัน (ไฟล์ `seed*.json`: เมตริกและประวัติต่อ epoch) และตารางสรุป (`*.csv`)
ไฟล์น้ำหนักโมเดล (`.pt`) และผลดิบของ grid ทั้งหมดไม่รวมใน repository เนื่องจากมีขนาดใหญ่

```text
src/results/
|-- Resnet18_2D, Resnet18_25D, Resnet50_2D, Resnet50_25D, Resnet50_3D
|     <ชื่อชุดค่า>/seed{1334,42,2026,7,2002}.json   ชุดค่า top-5 จาก grid + control (รัน 5 seed)
|     summary_*.csv                                  ตารางสรุปทุกชุดค่าและ top-5
|     test_*.csv                                     ผลบนชุดทดสอบของชุดที่เลือก
|-- Resnet18_3D   (ชุด stem_5 และชุดที่เทียบ รวม 10 seed)
`-- baseline/
      runs_baseline_{2D,25D,3D}/                     ผลการพัฒนา Baseline CNN ทุกชุดค่า (3 seed)
      runs_baseline_{2D,25D}_ep300/                  รันเพิ่มเติมที่ 300 epoch
```

## ข้อควรระวัง

- ห้ามแก้ `split.json`, `processing_lock.py`, `augmentation_lock.py` หลังเริ่มทดลอง เพราะทุกกลุ่มใช้ร่วมกัน
- ชุดทดสอบเปิดครั้งเดียวหลังล็อกค่าจากชุดตรวจสอบ
- 2D / 2.5D ใช้น้ำหนักตั้งต้น ImageNet ส่วน 3D เริ่มจากศูนย์ จึงเทียบข้ามรูปแบบข้อมูลได้ระดับประมาณ
