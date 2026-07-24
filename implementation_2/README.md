![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# MIRA: Multimodal Infrastructure Risk Analyzer

## 📌 Project Summary
MIRA asks a focused question: **given the water-risk profile of a geolocation, can we predict the scale of a data center built there?** We join the IM3 open-source data center atlas with WRI Aqueduct water-risk indicators and train gradient-boosted classifiers (XGBoost, CatBoost) to predict a three-tier "impact" label (Edge/Enterprise, Colocation, Hyperscale).

> **Critical note:** the headline result is essentially a *negative* one. Water risk alone is a weak predictor of data-center scale. This README is written to be honest about that, because the interesting finding is the weakness of the signal, not a high accuracy number.

## 🎯 Motivation
Data centers are water- and power-hungry, and where operators choose to build — and how large they build — has real environmental consequences. If siting decisions correlated strongly with water risk, that pattern would be worth surfacing. The motivation for MIRA was to test whether such a pattern is *observable and learnable* from public data.

**Shortcoming of the framing:** the question presupposes that operators price water risk into build scale. In practice, scale is driven far more by market demand, land, power availability, and fiber than by hydrology — so a weak result should have been the prior expectation, not a surprise.

## 🧩 Novelty
- **Question, not architecture**: The novelty is the framing (water-risk → build scale), not the models, which are off-the-shelf.
- **Honest signal accounting**: We treat "how much signal actually exists" as the deliverable, using mutual information and per-indicator correlations rather than a single accuracy headline.

**Shortcoming:** there is limited genuine novelty. The label is synthetic (derived from square footage), so the "prediction" task is partly an artifact of our own construction rather than a natural phenomenon.

## 🧠 Methodology
1. **Dataset**: [IM3 open-source data center atlas](https://github.com/IMMM-SFA) (~1,242 rows; 1,039 after filtering to `type == "building"`) joined to [WRI Aqueduct](https://www.wri.org/aqueduct) water-risk scores by geolocation.
2. **Target — `impact_tier` (3 classes)**: derived *deterministically* from square footage via `est_mw = sqft × 150 / 1e6` and fixed thresholds → Edge/Enterprise (~17%), Colocation (~57%), Hyperscale (~26%).
   - **Shortcoming (central flaw):** the label is a pure function of size. If `sqft`/`est_mw` are used as features the model just re-derives our own thresholding rule (≈100% accuracy, but circular). We therefore *exclude* size — which leaves the features and label almost independent by construction.
3. **Features**: 9 WRI water-risk indicators (`bws, bwd, iav, sev, rfr, cfr, drr, cep, udw`) after pruning columns with high sentinel/missing rates or negligible variance (`gtd, ucw, rri, usa`).
   - **Shortcoming:** because WRI scores are a function of location, they are a *coarse proxy* for geography; they carry little information about how big a building is (mutual information ≈ 0.01–0.15 per feature).
4. **Architecture**: XGBoost and CatBoost multiclass classifiers, small hyperparameter grids, 5-fold stratified CV, inverse-frequency / `auto_class_weights="Balanced"` for imbalance.
5. **Evaluation & Metrics**:
   - 3-class WRI-only: **macro-F1 ≈ 0.47, accuracy ≈ 0.53** — *below* the 0.57 majority-class baseline.
   - Binary (Hyperscale vs. rest): **ROC-AUC ≈ 0.72** — a weak but genuine signal.
   - Per-indicator: flood and seasonal-variability risks (`cfr, rfr, sev, cep`) correlate *negatively* with size (|ρ| ≈ 0.10–0.15, p < 0.001) — large builds mildly avoid flood-prone/variable-supply sites.
   - **Shortcoming:** accuracy is a misleading metric here (dominated by class imbalance); the 3-class task masks the fact that essentially all learnable signal is "hyperscale vs. not."

#### Additional Methodology:
- **Data quality checks**: NaN / sentinel (`-9999`) / inf guards and row-count/ID validation before training.
- **Shortcoming:** with only ~1,039 rows across 3 imbalanced classes, CV variance is non-trivial and the smallest tier is fragile.

## 🌍 Impact
The practical takeaway is a **cautionary, negative result**: WRI water-risk indicators do *not* meaningfully predict data-center scale. Operators do not appear to size builds around water risk in a way that is recoverable from these indicators alone. Communicating "there is little signal here" is itself useful — it discourages over-interpreting weak correlations as siting policy.

**Shortcoming:** the impact is limited by the synthetic, size-derived target and the narrow feature set. Any real-world claim about siting behavior would need a non-circular outcome variable (e.g., actual power draw, water withdrawal, or reported incidents) rather than a tier we defined ourselves.

#### Future Work
- **Fix the target**: replace the size-derived tier with a genuine, externally-measured outcome, or reframe explicitly as binary Hyperscale-vs-rest where the signal lives.
- **Add real predictors**: operator identity and geography (state/county) carry more signal (macro-F1 ≈ 0.62) but risk memorizing location→tier bindings — worth studying carefully and separately from the water-risk question.
- **Interpretability-first**: lead with SHAP / correlation reporting on the binary model rather than chasing a classification score.
- **Better metrics**: report ROC-AUC / PR-AUC and calibrated probabilities instead of accuracy.

**Additional Sources:**
- [WRI Aqueduct Water Risk Atlas](https://www.wri.org/aqueduct)
- [IM3 / IMMM-SFA data center resources](https://im3.pnnl.gov/)
