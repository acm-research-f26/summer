# SleepGuard: Predicting Nocturnal Hypoglycemia in Type 1 Diabetes Using Pre-Sleep Glucose Features

##  Project Summary
SleepGuard is a machine learning pipeline that investigates whether
pre-sleep glucose trajectory features can predict nocturnal hypoglycemia
in patients with Type 1 Diabetes. Using the real OhioT1DM 2018 dataset
(6 patients, 247 nights), a Random Forest classifier trained on 13
glucose-derived features from a 4-hour pre-sleep window achieves a mean
AUROC of 0.743 (p=0.0156) via Leave-One-Patient-Out cross-validation —
significantly above chance. The pipeline is designed to be extended with
HRV features once wearable heart rate data becomes available.

##  Motivation
Nocturnal hypoglycemia is one of the most dangerous complications of
Type 1 Diabetes: blood glucose drops below safe levels while patients
are asleep and unable to respond. Most existing wearable-sensor research
focuses on detecting hypoglycemia in real time, after it has already
begun. This project asks a different question — can signals available
before sleep onset predict whether a nocturnal low will happen at all?
Early prediction would give patients and caregivers actionable warning
time rather than a reactive alert during an event already in progress.

##  Novelty

* **Pre-sleep prediction, not real-time detection**: Frames the problem
  as predicting a future nocturnal event from a pre-sleep window (7–11 PM),
  rather than detecting an event as it is occurring — a fundamentally
  different and clinically more useful framing.
* **Rich glucose trajectory feature set**: Goes beyond simple mean glucose
  by extracting trajectory (slope, acceleration), variability (CV, MODD
  proxy), and risk-threshold features from the pre-sleep window — capturing
  the dynamics of glucose behavior rather than just its level.
* **Statistically validated on real patient data**: Achieves AUROC 0.743
  with Wilcoxon significance (p=0.0156) on the real OhioT1DM 2018 cohort,
  with 5 of 6 patients individually exceeding chance performance.

##  Methodology

1. **Dataset**: Uses the [OhioT1DM 2018 Dataset](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html)
   — 6 Type 1 Diabetes patients (IDs: 559, 563, 570, 575, 588, 591)
   with continuous glucose monitoring at 5-minute resolution. 247 nights
   total after quality filtering, 55 hypoglycemic (22.3%).
2. **Architecture**: Random Forest classifier (300 trees, max depth 6,
   class-balanced weighting) trained on 13 pre-sleep glucose features:
   * Trajectory: mean, std, min, max, last value, range
   * Trend: linear slope, quadratic acceleration
   * Variability: coefficient of variation, MODD proxy (mean absolute
     difference between consecutive readings)
   * Risk: % readings below 100 mg/dL, % readings below 80 mg/dL
   * Coverage: fraction of expected 5-minute readings present
3. **Evaluation**:
   * Leave-One-Patient-Out cross-validation (train on 5 patients,
     test on held-out patient, rotate through all 6)
   * Quality filter: nights with less than 50% pre-sleep window coverage
     excluded (9 nights dropped)
4. **Metrics**:
   * AUROC: 0.743 ± 0.104 (mean ± std across folds)
   * F1: 0.483 ± 0.072
   * Recall: 0.450 | Precision: 0.579
   * Wilcoxon signed-rank test vs chance: p = 0.0156 (significant)

**Additional Methodology:**

* **Synthetic data pipeline**: A synthetic OhioT1DM-format data generator
  was built to validate the full pipeline end-to-end before real dataset
  access was granted, ensuring the architecture was correct before touching
  real patient data.

## Impact
This work demonstrates that pre-sleep glucose dynamics carry statistically
significant predictive signal for nocturnal hypoglycemia, using only data
that a standard CGM already provides. If extended with HRV features from
consumer wearables, this approach could enable a low-cost, non-invasive
early-warning system built into devices patients already own — providing
actionable overnight alerts without requiring additional hardware. The
pipeline is fully open and reusable for future work on larger cohorts or
alternative sensor streams.

## Future Work

* **HRV feature integration**: The original research design called for
  HRV features from the Basis Peak wearable (available in the OhioT1DM
  2018 cohort). Heart rate data was found to be empty in the dataset
  version received; this has been flagged with the dataset maintainers
  and HRV feature extraction is planned for the next implementation once
  data is confirmed.
* **Statistical significance per patient**: Current Wilcoxon test is
  across folds; per-patient significance testing requires more nights
  per patient than currently available.
* **Cross-dataset validation**: Replicate findings on D1NAMO or another
  independent T1D cohort to test generalizability.
* **Personalized models**: Patient 559 underperformed (AUROC 0.588)
  while others reached 0.84+; patient-specific thresholds or models
  may improve individual prediction.

## Additional Sources

* Cichosz, S.L. et al. (2014) — real-time hypoglycemia detection using HRV signals.
* 2024 nocturnal hypoglycemia prediction paper (pediatric cohort) identifying
  HRV association as open future work.
* [OhioT1DM Dataset Description (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904/)