# SleepGuard — Pre-Sleep HRV Features for Nocturnal Hypoglycemia Prediction

Research project investigating whether heart rate variability (HRV) features
extracted from a pre-sleep window can improve prediction of nocturnal
hypoglycemia in Type 1 Diabetes, beyond a baseline using glucose, insulin,
and meal data alone.

## Research Question

Most existing work on hypoglycemia and wearable signals focuses on
**detection** — identifying a low blood glucose event while it is happening,
using real-time heart rate or galvanic skin response. This project asks a
different question:

> Can HRV features measured **before sleep onset** predict whether a
> nocturnal hypoglycemic event will occur at all, hours later?

This is a prediction problem from a pre-event window rather than a
real-time detection problem. The motivation is grounded in autonomic
nervous system physiology: counter-regulatory hormone responses tied to
glucose dysregulation are known to affect HRV, but no existing study has
extracted and evaluated HRV features specifically from the OhioT1DM dataset
for this purpose. This gap was explicitly identified as future work in
prior nocturnal hypoglycemia research.

## Dataset

**OhioT1DM 2018 cohort** — 6 patients (IDs: 540, 544, 552, 567, 584, 596)
with continuous glucose monitoring, insulin (basal + bolus), meal logs, and
heart rate from a Basis Peak fitness band recorded at 5-minute intervals.

The 2018 cohort is the only one with heart rate data — the 2020 cohort used
a different sensor (Empatica Embrace) without an equivalent HR stream.

Access requires a request to the dataset maintainers (Ohio State University /
Dr. Razvan Bunescu). This repository ships with a **synthetic data
generator** that mirrors the exact XML schema of the real dataset, so the
full pipeline can be built, tested, and demonstrated before real data
access is granted.

## Method

For each patient-night:

1. **Label** — night is marked hypoglycemic if any glucose reading falls
   below 70 mg/dL between 11:00 PM and 7:00 AM.
2. **Baseline features** — extracted from the 2-hour pre-sleep window
   (9:00–11:00 PM): glucose mean/std/last/slope, total bolus insulin,
   mean basal rate, total carbs.
3. **HRV features** — extracted from the same pre-sleep window using the
   5-minute heart rate stream converted to pseudo-RR intervals
   (`RR_ms = 60000 / HR_bpm`): RMSSD, SDNN, pNN50, mean HR, HR std.
4. **Models** — two Random Forest classifiers trained with
   Leave-One-Patient-Out cross-validation:
   - **Model A**: baseline features only
   - **Model B**: baseline + HRV features
5. **Evaluation** — AUROC, F1, precision, recall, per-patient breakdown,
   and feature importances.

## Project Structure

```
sleepguard/
├── src/
│   ├── generate_synthetic_data.py   # builds synthetic OhioT1DM-format XML
│   ├── data_loader.py               # parses XML into DataFrames
│   ├── feature_engineering.py       # labeling + feature extraction
│   ├── model.py                     # LOPO-CV training & evaluation
│   ├── visualize.py                 # generates result PNGs
│   └── main.py                      # pipeline entry point
├── data/
│   └── synthetic/                   # auto-generated synthetic dataset
├── results/                         # output metrics, CSVs, PNGs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run on synthetic data (auto-generated on first run):

```bash
cd src
python main.py
```

Run on real OhioT1DM data once access is granted:

```bash
python main.py --data /path/to/OhioT1DM/2018/train
```

All metrics print directly to the terminal. Result CSVs and PNGs are saved
to `results/` and overwritten on every run.

## Current Status

First implementation complete and validated end-to-end on synthetic data.
Real dataset access requested from Ohio State (pending approval). Once
real data is available, results will be regenerated and reported with
statistical significance testing across the leave-one-patient-out folds.

## Known Limitations

- **Small sample size**: 6 patients, ~250 patient-nights total. Results are
  a pilot study, not a generalizable claim.
- **HRV is derived, not measured directly**: the Basis Peak band records
  5-minute aggregated heart rate, not raw RR intervals. HRV features are
  computed from pseudo-RR intervals derived from this aggregated signal,
  consistent with prior literature working with the same constraint.
- **No statistical significance testing yet** on the baseline vs. HRV
  model comparison — planned for the next iteration.
- **Synthetic data results are illustrative only** and do not represent a
  real biological finding; they exist to validate that the pipeline runs
  correctly end-to-end.
