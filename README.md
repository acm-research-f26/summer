![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)
# Fibroid Cavity-Contact Classification from MRI Segmentation Masks

## 📌 Project Summary

This project extends MRI-based uterine fibroid segmentation into a downstream clinical classification task: identifying whether an individual fibroid touches or distorts the uterine cavity.

The work is motivated by recent research on automated uterine myoma segmentation, especially [Deep Learning-Based Automated Segmentation of Uterine Myomas](https://arxiv.org/abs/2508.11010), which uses nnU-Netv2 to segment the uterine wall, uterine cavity, myomas, and nabothian cysts from MRI. That paper focuses on producing accurate segmentation masks. This project asks the next question: once those masks exist, can they be converted into a fibroid-level clinical signal?

## 🎯 Motivation

Uterine fibroids are common benign tumors that can cause heavy bleeding, pelvic pain, and fertility-related complications. A clinically important distinction is whether a fibroid affects the uterine cavity, because cavity involvement can influence treatment planning and eligibility for less invasive procedures.

Segmentation models can locate anatomy, but they do not automatically explain what the anatomy means for a patient. A mask may show the cavity and fibroids, but a clinician still has to interpret whether a particular fibroid is close enough to matter.

This project is motivated by that gap. It turns segmentation masks into interpretable fibroid-level features and uses those features to support cavity-contact classification.

## 🧩 Novelty

**Downstream clinical bridge:** The project adds a classification step after segmentation, turning pixel-level masks into fibroid-level predictions.

**Individual fibroid analysis:** Instead of treating all fibroid tissue as one region, the pipeline separates connected fibroid components and analyzes each fibroid independently.

**Leakage-aware modeling:** Features that directly define the label, such as boundary-contact counts and overlap ratios, are excluded from model training and kept only for auditing.

**Interpretable feature design:** The model uses geometric measurements such as volume, distance, and aspect ratio rather than raw image pixels.

**Patient-level evaluation:** Grouped cross-validation prevents fibroids from the same patient from appearing in both training and testing folds.

## 🧠 Methodology

**Dataset:** The project uses the public [Uterine Myoma MRI Dataset (UMD)](https://figshare.com/articles/dataset/UMD_zip/23541312), which contains sagittal T2-weighted pelvic MRI scans from 300 patients with pixel-level annotation masks.

The segmentation labels are:

```text
0 = background
1 = uterine wall
2 = uterine cavity
3 = myoma/fibroid
4 = nabothian cyst
```

**Paper basis:** The project is based on the segmentation task described in [Deep Learning-Based Automated Segmentation of Uterine Myomas](https://arxiv.org/abs/2508.11010). That work applies nnU-Netv2 to segment uterine MRI structures. This implementation builds on that idea by adding a downstream fibroid-level classification layer.

**Architecture:** The referenced segmentation work uses [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). This project does not retrain nnU-Net. Instead, it assumes segmentation masks are available and implements the post-processing, feature extraction, labeling, and classification pipeline.

**Feature extraction:** Each 3D segmentation mask is processed with connected components to isolate individual fibroids. For each fibroid, the pipeline computes:

```text
volume_voxels
volume_mm3
centroid_to_cavity_dist_mm
aspect_ratio
```

**Label derivation:** A binary `cavity_touching` label is derived from fibroid-cavity boundary contact.

**Leakage control:** Direct-contact measurements are not used as predictors:

```text
boundary_contact_count
boundary_contact_ratio
overlap_count
overlap_ratio
min_distance_to_cavity_mm
```

These fields are retained for auditing and label derivation only.

**Models:** The project trains and evaluates:

```text
Logistic Regression
Random Forest
XGBoost
```

**Evaluation:** Models are evaluated with grouped patient-level cross-validation.

**Metrics and outputs:** The pipeline generates ROC AUC scores, ROC curves, per-fold metrics, feature summaries, class-balance reports, leakage audits, trained model files, and segmentation preview images.

## 🌍 Impact

This project shows how medical image segmentation outputs can be converted into a more clinically meaningful form. Instead of stopping at anatomical masks, the pipeline produces fibroid-level predictions that relate directly to treatment-planning questions.

## Data Setup

Download the UMD dataset from Figshare:

```text
https://figshare.com/articles/dataset/UMD_zip/23541312
```

After downloading `UMD.zip`, unzip it into:

```text
data/UMD/
```

Expected structure:

```text
data/UMD/UMD/UMD_221129_001/UMD_221129_001_seg.nii.gz
data/UMD/UMD/UMD_221129_001/UMD_221129_001_t2.nii.gz
...
```

The current pipeline reads the `*_seg.nii.gz` files.

## How to Run

Set up the environment:

```bash
make setup
```

Run the full pipeline on synthetic data:

```bash
make demo
```

Run the pipeline on the real UMD dataset:

```bash
make extract
make train
make report
make preview
```

Run tests:

```bash
make test
```

Optional SHAP interpretation can be enabled with:

```bash
.venv/bin/pip install -e ".[interpretability]"
```

If SHAP is not installed, the pipeline records that status and continues normally.

## Limitations

The `cavity_touching` label is derived from mask geometry, not from an independent radiologist-assigned clinical judgment.

The pipeline assumes the segmentation masks are accurate. Errors in the masks may affect the extracted features and labels.

The current model uses a small set of geometric predictors. More spatial features may improve performance if they are designed without reintroducing label leakage.

The pipeline has only been run on the UMD dataset and has not been validated on an external, independently annotated uterine MRI dataset.

## Additional Sources

- [UMD dataset on Figshare](https://figshare.com/articles/dataset/UMD_zip/23541312)
- [Large-scale uterine myoma MRI dataset covering all FIGO types with pixel-level annotations](https://www.nature.com/articles/s41597-024-03170-x)
- [Deep Learning-Based Automated Segmentation of Uterine Myomas](https://arxiv.org/abs/2508.11010)
- [nnU-Net official repository](https://github.com/MIC-DKFZ/nnUNet)
