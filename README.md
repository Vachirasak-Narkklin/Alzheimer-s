# Alzheimer's Disease Analysis from Brain MRI Using Deep Learning

Senior thesis project (Computer Science, Silpakorn University). We classify **Alzheimer's / not Alzheimer's**
(binary) from 3D brain MRI of the **OASIS-1** dataset. Nine model groups are compared
(Baseline CNN, ResNet-18, ResNet-50 x 2D, 2.5D, 3D). Instead of cropping the hippocampus, the whole brain is given
to the model, and Grad-CAM is used to check which regions the model relies on.

- Labels from CDR: `CDR = 0` -> normal, `CDR >= 0.5` -> Alzheimer's, subjects without a CDR value are discarded.
- 234 subjects, split at subject level: train 163 / validation 35 / test 36 (`src/00-shared/split.json`).
- Preprocessing and augmentation are locked and shared by all groups; models are selected on the validation set;
  the test set is opened once.

## Data

The MRI data are **not distributed with this repository**. Please download OASIS-1 (Cross-sectional MRI Data)
yourself from the OASIS project after registering and accepting its terms: <https://sites.wustl.edu/oasisbrains/>

Citation: Marcus, D. S., et al. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in
young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498-1507.
https://doi.org/10.1162/jocn.2007.19.9.1498

Acknowledgment required by OASIS: "Data were provided [in part] by OASIS-1: Cross-Sectional:
Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; NIH grants P50 AG05681,
P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382"

Set the data location with the `OASIS_ROOT` environment variable (default is in `src/00-shared/processing_lock.py`)
and edit `SHARED_DIR` in the setup cell of each notebook to match your machine.

## Project setup

| Item | Value |
|---|---|
| Input volume | 128 x 128 x 96 |
| 2D input | one axial slice (slice 48) |
| 2.5D input | three mid-slices: axial 48, coronal 64, sagittal 64 |
| Augmentation (train only) | rotation +-10 deg, gamma d = 0.2; no flip, no noise, no scaling |
| Training | 200 epochs, batch size 8, no early stopping |
| Seeds | 1334, 42, 2026, 7, 2002 (ResNet and test); 3 seeds for Baseline development |
| Checkpoint selection | epoch with the highest (AUC + F1) / 2 on the validation set |
| Evaluation | AUC, F1, accuracy, sensitivity, specificity, threshold 0.5, test set opened once |

## Repository structure

```text
Alzheimer's/
|-- README.md
|-- requirements.txt
`-- src/
    |-- 00-shared/       locked files shared by every model (split, preprocessing, augmentation)
    |-- 01-data-review/  data inspection and label creation
    |-- 02-data-leakage/ split creation and leakage checks
    |-- 03-augment/      augmentation sweeps (ResNet-18, 2D / 2.5D / 3D)
    |-- 04-model/        the 9 model groups
    |   |-- 2D/    (Baseline, Resnet18, Resnet50)
    |   |-- 25D/   (Baseline, Resnet18, Resnet50)
    |   `-- 3D/    (0_Baseline, 1_Resnet18, 2_ResNet50, runcurves_cell.py)
    `-- results/         run results (json / csv)
```

### File details

```text
[src/00-shared]
  split.json                        subject lists for train / validation / test (used by every model)
  split_reference.json              reference copy used to verify the split
  processing_lock.py                locked preprocessing (128 x 128 x 96, intensity clipping, resizing,
                                    construction of 2D / 2.5D / 3D inputs)
  augmentation_lock.py              locked augmentation: rotation +-10 deg, gamma d = 0.2, no noise, no
                                    left-right flip; randomness is derived from (seed, epoch, subject_id)
  load_locked_split_cell.py         code cell that loads split.json, pasted into every model notebook
  03_Slice_Position_Check.ipynb     checks slice positions (x, y, z) against the hippocampus
  04_Test_Processing_Lock_3D.ipynb  tests processing_lock.py on real OASIS-1 data
  processing_lock_test_report.json, fig_processing_lock_check.png   outputs of the notebook above

[src/01-data-review]
  00_Review_Data.ipynb              inspects the clinical table, drops subjects without CDR, builds labels
  01_TargetShape_Experiment.ipynb   experiment for choosing the target shape
  A_Check_OASIS_Hippocampus.ipynb   checks OASIS-1 against a brain atlas before Grad-CAM
  Check_Demographics_Split.ipynb    age and sex of each split
  A_check_output/, age_distribution_by_split.png   outputs

[src/02-data-leakage]
  create_and_test_split_json.ipynb, preload_split_by_seed.ipynb, preload_locked_split_by_seed.ipynb,
  preload_split_check.ipynb, preload_split_check_3modes.ipynb, split_pairwise_seed_check.ipynb
                                    create the split and verify there is no subject overlap

[src/03-augment]
  Aug_{Rotation,Gamma,Noise,Prob}_ResNet18_{2D,25D,3D}.ipynb
                                    augmentation sweeps; results define augmentation_lock.py

[src/04-model]
  2D/Baseline/   Baseline_CNN3Layer_2D.ipynb         (main file)
  2D/Resnet18/   Resnet18_2D.ipynb
  2D/Resnet50/   Resnet50_2D.ipynb
  25D/Baseline/  Baseline_CNN3Layer_25D.ipynb        (main file)
  25D/Resnet18/  Resnet18_25D.ipynb
  25D/Resnet50/  Resnet50_25D.ipynb
  3D/0_Baseline/ Baseline_CNN3Layer_3D.ipynb (+ _BatchSize, _Width_3seed experiments)
  3D/1_Resnet18/ Resnet18_3D.ipynb
  3D/2_ResNet50/ Resnet50_3D.ipynb
  Baseline_CNN_{2D,25D,3D}.ipynb    additional Baseline runs with the same structure and parameter count
  Check_Criteria_Baseline.ipynb     checks the selection criterion (margin from the model's own variance)
  Grad-CAM_All_9_Models.ipynb       Grad-CAM of all 9 groups and overlap with the hippocampus
                                    (Harvard-Oxford atlas); copies exist in several folders because it was run per group
  gradcam_defs/*.py                 model definitions loaded by Grad-CAM (identical copies)
  3D/runcurves_cell.py              plots per-epoch curves from saved result files
```

## Run order

1. `01-data-review` - inspect the data and create labels
2. `02-data-leakage` - create and verify `split.json`
3. `00-shared` - check `processing_lock.py` and slice positions
4. `03-augment` - augmentation sweeps (give the values in `augmentation_lock.py`)
5. `04-model` - train / repeat each group
6. `04-model/.../Grad-CAM_All_9_Models.ipynb` - explain the models

## Results folder

`src/results/` holds the per-run results (`seed*.json`: metrics and per-epoch history) and summary tables (`*.csv`).
Model weights (`.pt`) and the raw outputs of the full grids are not included because of their size.

```text
src/results/
|-- Resnet18_2D, Resnet18_25D, Resnet50_2D, Resnet50_25D, Resnet50_3D
|     <config>/seed{1334,42,2026,7,2002}.json   top-5 grid configs + control (5 seeds)
|     summary_*.csv, test_*.csv                  summaries and test results of the selected config
|-- Resnet18_3D
`-- baseline/
      runs_baseline_{2D,25D,3D}/                 Baseline CNN development runs (3 seeds)
      runs_baseline_{2D,25D}_ep300/              additional 300-epoch runs
```

## Final results

Test set (36 subjects: 15 Alzheimer's / 21 normal), mean +- SD over 5 seeds, threshold 0.5,
test set opened once after the configurations were locked on the validation set.

| Model | AUC | F1 | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Baseline CNN 2D | 0.759 +- 0.059 | 0.651 +- 0.053 | 0.683 +- 0.032 | 0.720 +- 0.137 | 0.657 +- 0.098 |
| Baseline CNN 2.5D | 0.782 +- 0.044 | 0.704 +- 0.030 | 0.728 +- 0.023 | 0.787 +- 0.119 | 0.686 +- 0.110 |
| Baseline CNN 3D | 0.766 +- 0.021 | 0.658 +- 0.023 | 0.672 +- 0.012 | 0.760 +- 0.089 | 0.610 +- 0.078 |
| ResNet-18 2D | **0.855 +- 0.022** | 0.707 +- 0.028 | 0.789 +- 0.015 | 0.613 +- 0.056 | 0.914 +- 0.040 |
| ResNet-18 2.5D | 0.779 +- 0.039 | 0.679 +- 0.034 | 0.711 +- 0.037 | 0.733 +- 0.082 | 0.695 +- 0.093 |
| ResNet-18 3D | 0.789 +- 0.078 | 0.653 +- 0.128 | 0.706 +- 0.087 | 0.693 +- 0.203 | 0.714 +- 0.067 |
| ResNet-50 2D | 0.766 +- 0.041 | 0.686 +- 0.029 | 0.717 +- 0.023 | 0.747 +- 0.087 | 0.695 +- 0.080 |
| ResNet-50 2.5D | 0.816 +- 0.042 | **0.716 +- 0.057** | 0.733 +- 0.025 | 0.827 +- 0.154 | 0.667 +- 0.083 |
| ResNet-50 3D | 0.817 +- 0.030 | 0.699 +- 0.067 | 0.739 +- 0.032 | 0.747 +- 0.152 | 0.733 +- 0.072 |

Summary

- ResNet-18 2D has the highest AUC (0.855); ResNet-50 2.5D has the highest F1 (0.716).
- ResNet-18 2D has high specificity (0.914) but low sensitivity (0.613) because the threshold was fixed at 0.5 and not tuned.
- With only 36 test subjects, most differences between models are within the variation across seeds, so we do not
  claim that one model or one input type is clearly better. A simple Baseline CNN is already competitive (AUC 0.76-0.78).
- Resources (parameters / peak GPU memory / minutes per run): Baseline 16,722-64,642 params, 0.03-1.56 GB, 3-10 min;
  ResNet-18 11.2-33.1 M, 0.22-2.22 GB, 2-11 min; ResNet-50 23.5-46.2 M, 0.57-4.56 GB, 4-17 min.
  3D models cost the most memory and time.

Grad-CAM (exploratory)

- 3D models: ResNet-50 3D and Baseline 3D put more attention on the hippocampus region for Alzheimer's cases than for
  normal cases (hippocampus focus ratio 1.48 vs 0.98 for ResNet-50 3D); ResNet-18 3D shows no clear difference.
- 2D models use one axial slice that does not contain the hippocampus; the attention of ResNet-18 2D is higher around
  the ventricles for Alzheimer's cases (ventricular enlargement is a known sign of brain atrophy).
- 2.5D models rely mostly on the coronal slice (about 45-47% for ResNet) which is the plane that shows the hippocampus.
- Grad-CAM shows where the model looks, not why the disease occurs; it is not clinical proof.

## Possible improvements

- Tune the decision threshold (or use class-aware metrics) to balance sensitivity and specificity.
- Use more data: combine OASIS-2 and ADNI, or use k-fold cross-validation, to reduce the variance of a 36-subject test set.
- Report confidence intervals from per-subject probabilities (save predictions per subject).
- Use 3D pretrained weights (for example MedicalNet) so that 3D models are not trained from scratch while 2D / 2.5D use ImageNet weights.
- Use more slices or multiple slice positions for 2D / 2.5D, and add age and sex as covariates or matched controls.
- Adjust multiple comparisons in the Grad-CAM analysis and validate regions with a clinician.

## Limitations

- Small dataset: 234 subjects, 36 in the test set (15 Alzheimer's). One fixed split; the differences between models are
  often smaller than the variation across seeds. Per-subject probabilities were not stored, so bootstrap confidence
  intervals could not be computed from the saved test files.
- Single dataset (OASIS-1), cross-sectional, one acquisition setup; no external validation on OASIS-2 or ADNI.
- Binary labels only (CDR 0 vs >= 0.5); severity stages and mild cases are not separated. CDR is a clinical rating,
  not a pathological confirmation.
- Model groups are not perfectly comparable: 2D / 2.5D start from ImageNet weights while 3D starts from scratch;
  Baseline structures differ across input types (for example BatchNorm only in 3D); first-layer kernels differ across groups.
- Hyperparameters were selected on the validation set of the same split (35 subjects), so some optimism is possible.
- Age and sex were not matched or controlled; the model may partly use age-related atrophy rather than disease-specific signal.
- Grad-CAM p-values are exploratory (many tests, not corrected); atlas overlap is approximate (atlas space differs from
  OASIS space by a few mm, and atrophied brains change the tissue inside a fixed region); the deepest layer is too coarse
  to localize.
- This is a research prototype, not a diagnostic tool, and must not be used for clinical decisions.

## Notes

- Do not edit `split.json`, `processing_lock.py` or `augmentation_lock.py` after experiments start; every group shares them.
- The test set is opened once, after the configurations are locked on the validation set.
- 2D / 2.5D use ImageNet pretrained weights while 3D starts from scratch, so cross-input comparisons are approximate.
