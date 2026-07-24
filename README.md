![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# MIRA: Fusing Satellite Vision with Water-Risk Features

- Tentative title; alternative is ATLAS

## 📌 Project Summary

MIRA (Multimodal Infrastructure Risk Analyzer) is working toward a multimodal
classifier that predicts data center impact tier from both satellite imagery
and location-specific water risk. The semester-long project will fuse a
five-band Sentinel-2 vision stream with the strongest WRI Aqueduct features,
then compare that fused model directly against a vision-only classifier.

The completed proof-of-concept work establishes both sides of that system:

- **Vision stream:** A ResNet-34 adapted from three to five input channels
(RGB, NIR, and SWIR) achieved **0.630 validation macro-F1** on an initial
55-site Virginia dataset. The 11-site validation set is too small for this
result to establish generalization, but it validates the end-to-end imagery
pipeline.
- **Water-feature stream:** CatBoost and XGBoost benchmarks on 1,039 IM3 sites
compared two representations of the same locations:
- **Original WRI indicators (9 features):** baseline water stress, baseline
water depletion, interannual variability, seasonal variability, riverine
flood risk, coastal flood risk, drought risk, coastal eutrophication
potential, and untreated connected wastewater.
- **Electric Power (ELP) aggregates (4 features):** water quantity, water
quality, regulatory and reputational risk, and total water risk, each
weighted for the electric-power sector.

The four ELP features match or improve on the original nine-feature
representation in both CatBoost and XGBoost. This reduces the water predictor
count by **56%** while preserving more task-relevant information, making ELP
the leading input set for the future multimodal fusion model.

## 🎯 Motivation

Data centers depend on both water and electricity, but general-purpose water
risk indicators may not represent the risks most relevant to electric-power
infrastructure. We therefore tested whether WRI's sector-weighted ELP
aggregates form a smaller and more informative feature set for predicting data
center impact tier.

Satellite imagery can capture visible characteristics such as building
footprint, roof geometry, surrounding infrastructure, land use, and spectral
signatures. Water features provide a different, non-visual description of a
site's environmental context. The next research question is whether this
complementary context improves impact-tier classification beyond imagery
alone.

The goal is not to claim that water risk determines data center scale. The
water-only experiments show that its independent signal is limited. The
semester project instead tests the narrower hypothesis that the best
sector-specific water features provide incremental value when combined with a
vision representation.

## 🧩 Novelty

- **Sector-specific feature compression:** Four electric-power-weighted
Aqueduct aggregates are benchmarked directly against nine disaggregated WRI
indicators.
- **Controlled comparison:** Both feature sets use the same 1,039 locations,
target labels, train/test IDs, cross-validation folds, and random seed.
- **Model-specific evidence:** The comparison is repeated with independently
tuned XGBoost and CatBoost models instead of relying on one estimator.
- **Interpretability check:** Native and test-set permutation importance are
used to examine which ELP aggregates carry the compressed signal.
- **Multimodal research direction:** The selected water representation will be
fused with learned Sentinel-2 embeddings and evaluated against an otherwise
identical vision-only baseline.



## 🧠 Methodology

1. **Dataset:** Approximately 1,039 shared building records from the
  [IM3 open-source data center atlas](https://im3.pnnl.gov/) joined by
   geolocation to [WRI Aqueduct](https://www.wri.org/aqueduct) water-risk
   scores.
2. **Target:** A three-class `impact_tier` label derived from estimated power
  and building-size thresholds: Edge/Enterprise, Colocation, or Hyperscale.
3. **Feature sets:**
  - **Original (9):** `bws`, `bwd`, `iav`, `sev`, `rfr`, `cfr`, `drr`, `cep`,
   and `udw`.
  - **ELP (4):** electric-power-weighted quantity (`qan`), quality (`qal`),
  regulatory/reputational (`rrr`), and total (`tot`) risk.
  - `gtd` was removed from the original representation because of its high
  sentinel rate; `ucw` and `rri` were constant; and `usa` had negligible
  variance.
4. **Architecture:**
  - **Completed water benchmarks:** Multiclass XGBoost and CatBoost with
   class balancing.
  - **Completed vision proof of concept:** ImageNet-pretrained ResNet-34
  adapted to accept 5-channel, 128×128 Sentinel-2 tiles. RGB weights are
  transferred, NIR/SWIR kernels are initialized from the RGB mean, and the
  final layer predicts three impact tiers.
  - **Planned fusion model:** A vision encoder will produce an image
  embedding, while a tabular branch will encode the selected ELP features.
  Their representations will be concatenated and passed to a shared
  classification head.
5. **Evaluation:**
  - One shared stratified 80/20 train/test split (831/208 rows).
  - One shared five-fold stratified cross-validation assignment.
  - Hyperparameters selected by mean validation macro-F1; the test set was
  used once for the final comparison.
6. **Metrics:** Macro-F1 is the primary metric because the three tiers are
  imbalanced. Weighted F1, accuracy, and class-specific F1 are also reported.



### CatBoost and XGBoost benchmarks

**XGBoost**

- The selected ELP model (`regularized`) achieved **0.474 ± 0.035 CV
macro-F1**, compared with **0.444 ± 0.033** for the selected original
nine-feature model (`deep_fast`).
- On the shared test set, ELP improved macro-F1 from **0.458 to 0.511**
(**+0.053**), weighted F1 from **0.525 to 0.556**, and accuracy from
**0.524 to 0.548**.
- The largest class-level gain was Edge/Enterprise F1, which increased from
**0.261 to 0.415**. Colocation increased from **0.617 to 0.630**, while
Hyperscale was nearly unchanged (**0.496 to 0.487**).

**CatBoost**

- Both feature sets selected the `balanced` configuration. ELP achieved
**0.463 ± 0.032 CV macro-F1**, compared with **0.441 ± 0.041** for the
original indicators.
- On the shared test set, ELP improved macro-F1 from **0.473 to 0.481**
(**+0.008**), weighted F1 from **0.526 to 0.547**, and accuracy from
**0.519 to 0.543**.
- ELP improved Colocation F1 from **0.604 to 0.643** and Hyperscale F1 from
**0.483 to 0.496**, although Edge/Enterprise F1 fell from **0.333 to
0.306**.

Across both model families, ELP produced higher cross-validation and test
macro-F1 with five fewer predictors. The improvement was strongest for
XGBoost; CatBoost's test macro-F1 gain was modest and should not be interpreted
as conclusive without repeated splits or external validation.

#### Additional Findings

- **Most useful ELP dimensions:** Test-set permutation importance ranked water
quality first and regulatory/reputational risk second for both models.
- **XGBoost ELP permutation importance:** quality **0.0815**,
regulatory/reputational **0.0750**, total **0.0610**, and quantity
**0.0552** macro-F1 decrease after permutation.
- **CatBoost ELP permutation importance:** quality **0.0916**,
regulatory/reputational **0.0861**, quantity **0.0386**, and total
**0.0280**.
- **Redundancy warning:** ELP total risk is strongly correlated with ELP
quantity risk (approximately **0.955**), so individual importances should be
read as model reliance within a correlated group, not as causal effects.
- **Baseline context:** The majority-class predictor has test macro-F1
**0.243**. Every trained model exceeds it on the imbalance-aware primary
metric, even though overall accuracy remains close to the majority-class
baseline.



### Semester-long multimodal study

The principal experiment will train and evaluate two models on the **same
sites, labels, splits, and imagery**:

1. **Vision-only baseline:** Five-band Sentinel-2 tiles processed by the vision
  encoder and classification head.
2. **Vision + water fusion:** The same vision encoder and head augmented with
  the strongest ELP water-risk representation.

This paired design isolates the value added by water context. The primary
comparison will be macro-F1, supplemented by weighted F1, per-class F1,
confusion matrices, and the fused model's performance difference from the
vision-only baseline across repeated splits.

The initial fusion candidate is the four-feature ELP set because it performed
best as a group. Water quality and regulatory/reputational risk are especially
important ablation candidates because both CatBoost and XGBoost ranked them
first and second by test-set permutation importance. We will also test whether
removing the highly correlated total or quantity feature yields a smaller,
equally effective fusion input.

The current vision and water results come from different cohorts (55 Virginia
sites for vision and 1,039 IM3 sites for water). They cannot yet be compared as
if they were one experiment. The first semester milestone is therefore to
build a single ID-aligned cohort containing imagery, ELP scores, and impact
labels for every included site.

## 🌍 Impact

The results support using the four ELP aggregates when a compact,
electric-power-relevant representation is preferred. They remove five
predictors and improve the benchmark most clearly in XGBoost, suggesting that
sector weighting can be more useful than retaining every underlying general
water-risk dimension.

The absolute scores remain moderate: water risk alone does not reliably
determine data center scale. The result is therefore evidence for better
feature representation, not evidence that these environmental risks cause
operators to build larger or smaller facilities.

The broader project will determine whether environmental context adds
measurable information to remote-sensing models. A successful fused model
could support scalable screening for grid planning and environmental review
without requiring proprietary power contracts or metering data. A null result
would also be useful: it would show that the selected water-risk features do
not improve the visual baseline and should not be presented as predictive
signals.

#### Future Work

- Build a larger, geographically diverse cohort in which each site has a
Sentinel-2 tile, ELP scores, and one consistently defined impact label.
- Establish the five-band ResNet-34 as a reproducible vision-only baseline,
then compare it with the fused vision + ELP model under identical splits.
- Evaluate early, intermediate, and late fusion strategies while controlling
model capacity so any gain is attributable to the water stream rather than
simply a larger network.
- Run ELP ablations: all four aggregates, quality plus
regulatory/reputational risk, and variants that remove either total or
quantity because of their high collinearity.
- Use repeated stratified splits or nested cross-validation and report
confidence intervals for the fused-minus-vision-only performance difference.
- Validate on geographically and temporally independent sites, including
regions beyond the current Virginia vision cohort.
- Explore temporal Sentinel-2 sequences to capture construction and seasonal
changes after establishing the single-image multimodal baseline.
- Replace the size-derived tier with observed power demand, water withdrawal,  
or another externally measured outcome.

**Additional Sources:**

- [WRI Aqueduct Water Risk Atlas](https://www.wri.org/aqueduct)
- [Integrated Multisector Multiscale Modeling (IM3)](https://im3.pnnl.gov/)
- [Sentinel-2 MSI Level-2A on Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

