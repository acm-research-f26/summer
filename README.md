![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# SleepGuard Predicting Nocturnal Hypoglycemia in Type 1 Diabetes Using Pre-Sleep HRV Features

## 📌 Project Summary
SleepGuard is a research pipeline that investigates whether heart rate
variability (HRV) features extracted from a pre-sleep window can predict
nocturnal hypoglycemia in patients with Type 1 Diabetes. Using the
OhioT1DM 2018 dataset, the project builds two classifiers — one using
only glucose, insulin, and meal data, and one that adds pre-sleep HRV
features — to test whether autonomic nervous system signals measured
before sleep onset carry predictive information about a hypoglycemic
event that won't occur for hours.

## 🎯 Motivation
Nocturnal hypoglycemia is one of the most dangerous complications of
Type 1 Diabetes because it occurs while patients are asleep and unable
to respond to early symptoms. Most existing wearable-sensor research
focuses on detecting hypoglycemia while it is already happening, using
real-time heart rate or skin response signals. This leaves a gap almost
no work asks whether signals available before sleep can predict whether
a nocturnal low will happen at all, which would give patients and
caregivers actionable warning time rather than a real-time alert during
an event already in progress.

## 🧩 Novelty

 Pre-sleep prediction window, not real-time detection Frames the
  problem as predicting a nocturnal hypoglycemic event hours in advance
  from a pre-sleep feature window, rather than detecting an event as it
  is occurring.
 HRV features on OhioT1DM No existing study has extracted and
  evaluated HRV-derived features from the OhioT1DM dataset for
  nocturnal hypoglycemia prediction, despite a 2024 paper explicitly
  identifying the HRV–hypoglycemia relationship as open future work.
 Pseudo-HRV from aggregated wearable data Demonstrates a method for
  deriving RMSSD, SDNN, and pNN50 from 5-minute aggregated heart rate
  (Basis Peak band), rather than requiring raw RR-interval data that
  most consumer wearables don't expose.

## 🧠 Methodology

1. Dataset Uses the [OhioT1DM 2018 Dataset](httpsmarthealth.cs.ohio.eduOhioT1DM-dataset.html)
   — 6 Type 1 Diabetes patients with continuous glucose monitoring,
   insulin (basal + bolus), meal logs, and Basis Peak heart rate data
   at 5-minute resolution. ~250+ patient-nights total.
2. Architecture Random Forest classifier (scikit-learn), trained in
   two configurations for direct comparison
    Model A (Baseline) glucose meanstdlastslope, total bolus
     insulin, mean basal rate, total carbs — all from the 2-hour
     pre-sleep window.
    Model B (Baseline + HRV) all baseline features plus RMSSD,
     SDNN, pNN50, mean HR, and HR std derived from pre-sleep heart rate.
3. Evaluation
    Leave-One-Patient-Out cross-validation (train on 5 patients, test
     on the held-out patient, rotate through all 6).
    Class-balanced training to account for the imbalance between
     hypoglycemic and normal nights (~15-25% positive class).
4. Metrics
    AUROC, F1 score, precision, recall — reported as mean ± std across
     the leave-one-patient-out folds, plus a per-patient breakdown and
     Random Forest feature importances.

Additional Methodology

 Synthetic data validation A synthetic data generator was built to
  mirror the exact OhioT1DM XML schema, allowing the full pipeline to be
  built, tested, and validated end-to-end while real dataset access was
  pending approval.

## 🌍 Impact
If pre-sleep HRV features prove predictive, this work points toward a
low-cost, non-invasive early-warning approach for nocturnal hypoglycemia
that could be built into consumer wearables patients already own,
without requiring real-time CGM alerts during sleep. More broadly, it
contributes evidence on a question explicitly flagged as open in prior
literature, and provides a reusable pipeline for testing pre-event
physiological signals against more conventional CGMinsulin baselines.

## Future Work

 Validate on real OhioT1DM data Current results are on a synthetic
  dataset built to validate the pipeline; real dataset access is pending
  approval from Ohio State University.
 Statistical significance testing Add a Wilcoxon signed-rank test
  or permutation test across patient folds to confirm whether the
  baseline vs. HRV performance difference is significant, not noise.
 Cross-dataset replication Validate findings on an independent
  cohort such as D1NAMO, which includes higher-resolution chest-strap
  heart rate data, to test generalizability beyond the Basis Peak sensor.
 Personalized modeling Per-patient results showed HRV helped some
  patients and not others — worth investigating whether patient-specific
  models outperform a pooled model.

## Additional Sources

 Cichosz, S.L. et al. (2014) — real-time hypoglycemia detection using HRV.
 2024 nocturnal hypoglycemia prediction paper (children) identifying
  HRV as future work.
 [OhioT1DM Dataset Description Paper (PMC)](httpspmc.ncbi.nlm.nih.govarticlesPMC7881904)
