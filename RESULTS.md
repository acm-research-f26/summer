# Results

Written progressively, one section per phase. See `PLAN.md` for phase definitions and exit gates.

## Phase 0 — Skeleton & environment

| item | value |
|---|---|
| Python | 3.13.11 (homebrew), venv at `.venv` |
| Package | `figomeas`, editable install from `src/` |
| Config | `config/default.yaml` — every threshold named, no magic numbers in `src/` |
| Dataset | `data/UMD/UMD`, 7.3 GB, 300 patients (267 `_seg` + 33 `_seq`), 6,845 DICOM |

**Exit gate: PASSED** — `make setup && make test` green from a clean tree.

## Phase 1 — Data setup & spacing truth table

`make manifest` → `results/manifest.csv` (300 rows, 30 columns). **All 8 exit gates PASSED.**

### Cohort as loaded

| | value |
|---|---|
| patients loaded | **300 / 300**, zero silent drops |
| masks found by `*_se[gq].nii.gz` | 267 `_seg` + 33 `_seq` |
| containers (sniffed, not inferred) | 266 gzip, **34 raw** |
| spacing source chosen | **DICOM for 300/300** |
| slice spacing measured from | `ImagePositionPatient` for 300/300 |
| labels present | ⊆ {0,1,2,3,4}, no violations |
| fibroid connected components | **1132** across 300 patients (median 3/patient) |

Slice spacing is measured geometrically — each slice origin projected onto the slice normal, median
consecutive gap — rather than read from `SliceThickness`. Thickness ignores inter-slice gaps, which
are real here (e.g. 4 mm thick / 5 mm spaced).

### F2 corrected: what the fake `(1,1,1)` actually costs

The original brief, and my own first reading of it, assumed reading `(1,1,1)` would blow up volumes
by ~100×. **That was wrong**, and the truth is worse:

| quantity | true (DICOM) | claimed by `(1,1,1)` | error |
|---|---|---|---|
| voxel volume | 0.829–1.291 mm³ (mean 1.021) | 1.000 mm³ | **0.77–1.21×** |
| anisotropy `sz/sx` | **11.6×** | 1.0× | **~12× wrong** |

The small in-plane spacing and the thick slice very nearly cancel in the product, so `(1,1,1)` gets
*volume* right to within ~20%. A volume sanity-check would never have caught it. What it destroys is
the **shape** of the voxel — and every mm-sized structuring element, max-diameter, sphericity, and
the Phase 5 perturbation depends on that, not on the volume.

### F3 — a third naming defect, found only because the container is sniffed

`UMD_221129_037` holds **`UMD_2221129_037_seg.nii.gz`** (extra digit) and is *raw*, so the
container/name split is not 33/267:

| | gzip | raw |
|---|---|---|
| `_seg` | 266 | **1** |
| `_seq` | 0 | 33 |

### Independent validation of the DICOM path

The `_seq` corrections are only trustworthy if the DICOM extraction is itself correct. On the 267
`_seg` patients — whose NIfTI pixdim *is* trustworthy — DICOM and NIfTI agree to
**max |Δ| = 7.7e-06 mm**. DICOM-derived slice counts match mask
depth for **300/300** patients. The DICOM path reproduces known-good spacing exactly, so its
corrections to the 33 can be believed.

### Cohort anisotropy (using corrected spacing)

median **13.3×**, IQR 9.1–14.1×,
range 7.0–21.1×. **Zero patients below 2×.**
Confirms F2: there is no isotropic subset — the 33 apparent ones were the artifact.

### Gate corrections made during this phase

- **G3** originally hardcoded `sz ∈ 4.0–6.0 mm` from a 6-patient sample and failed
  `UMD_221129_086` (6.05 mm), a spacing that occurs legitimately in the `_seg` cohort too. The gate
  now uses the config band. This was a bad gate, not bad data.
- **G6–G8** added after the fact to lock in the F3 discovery, the series/depth match, and the
  DICOM↔NIfTI corroboration.

**Exit gate: PASSED** (8/8). Test suite: 62 passed.

## Phase 2 — Uterine reference-surface reconstruction

**All 7 exit gates PASSED.** `make body` → `results/body_qc.csv`, `body_ablation.csv`,
`reports/body_qc/*.png`.

### F5 — the brief's percent-intramural formula is degenerate

The specified formula, `100 × |F ∩ body ∩ ¬C| / |F|` with `body = fill(close(W ∪ C ∪ F))`,
returns **100% for every fibroid under every reconstruction method**. The masks are a hard
partition, so `F ∩ ¬C ≡ F` identically; and `body` is built from a union containing `F`, so
`F ∩ body ≡ F`. It collapses to `100 × |F|/|F|`. Measured on 25 real patients:

| method | body/union | median p | **p IQR** | **% pinned at 100** |
|---|---|---|---|---|
| **closing (adopted)** | 0.97 | 91.0 | **35.2** | **39.5%** |
| fill only | 0.93 | 57.5 | 100.0 | 32.6% |
| union closing | 1.03 | 100.0 | 0.00 | 86.0% |
| convex hull | 1.09 | 100.0 | 0.00 | 97.7% |
| **brief's formula, literal** | — | 100.0 | **0.00** | **100.0%** |

**This re-diagnoses the earlier failed attempt.** The convex hull was blamed for collapsing the
submucosal count to ~9. The hull is an aggravating factor (97.7% pinned vs 100%), but with this
formula the count collapses under morphological closing too — every fibroid reads as type 3/4.
The formula was the root cause; swapping the hull for closing would not have saved it.

### The replacement: reference surfaces

A fibroid displaces the surfaces it should be measured against. So each is reconstructed as it
would be *without that fibroid's distortion*: the serosa from `wall ∪ cavity` with the measured
fibroid **withheld** (an intramural fibroid is then an enclosed hole that fill-holes restores; a
subserosal one is not enclosed, so the protruding part stays outside), and the endometrium by
closing the cavity across the fibroid's dent.

**The bridging radius must scale with the fibroid.** A partially buried fibroid cuts a crater as
wide as itself. With a fixed 5 mm closing a textbook type 5 measured **1.7% intramural**; with a
radius of 1.5 × the equivalent-sphere radius it reads 71% against an analytic truth of 81%.

### Cohort reconstruction (300 patients)

| gate | result |
|---|---|
| G1 body contains all wall+cavity voxels | **0 violations** |
| G2 body ≤ 1.5× the union | max ratio 1.082 |
| G3 one dominant component (≥99% of volume) | 95.0% of patients |
| G4 cavity preserved, not swallowed | **300/300**, max cavity/body 0.229 |
| G5 phantoms on the correct side of 50% | 8/8 |
| G6 measurement not degenerate | closing IQR 35.2 vs brief's 0.0 |
| G7 unmeasurable fibroids flagged | 41/931 (4.4%) |

### Two bugs the gates caught

**`close_mm` was deleting annotated tissue.** Dilation ran on the unpadded array, so near an array
face the ball was clipped; the following erosion then read that face as background and ate the
voxels. Closing was therefore **not extensive**. Every loss sat on the first or last slice a
structure occupied, and it cost **25% of the labelled cavity** on `UMD_221129_048`. Caught only
because G4 asserted a mathematical invariant (`close(A) ⊇ A`) rather than a summary statistic.

**Pruning order was deleting myometrium.** Taking the largest connected component *after* unioning
the scaffold removed real annotated tissue — >1% of the scaffold on 21 patients, 54% on one.
Containment is now the enforced invariant and component dominance is reported, not imposed.

### Fibroids with no reconstructable reference

**41 of 931 (4.4%)** are flagged `reliable=False`: the fibroid dominates the uterine mass, so no
myometrial envelope survives to reconstruct. `UMD_221129_016` is the extreme — 9.4k wall voxels
around a **1.43M-voxel, 918 cm³** fibroid, measuring 0.0% intramural, which would derive as
"subserosal pedunculated" for a mass that has replaced the entire uterus. These are flagged and
excluded from gates, never silently scored.

### Known bias (carried forward)

Phantom errors are directional: submucosal fibroids read **too intramural**, subserosal **too
little**, so the method inflates type 2 and type 6. This is systematic and is *not* covered by the
Phase 5 confidence intervals, which measure precision only.

## Phase 3 — Per-fibroid feature extraction

`make extract` → `results/fibroids.parquet`. **All 5 gates PASSED.**

| | value |
|---|---|
| fibroids | **931** from 295 patients (dataset documents ~1,077) |
| raw components | 1,132; **201 dropped** below the 100 mm³ floor |
| percent_intramural | range 0–100, 0 NaN, sd 31.8 |
| volume | median **1.01 cm³**, max 3,439 cm³ |
| through-plane extent | median **2 slices**; **68.7% span ≤3 slices** |
| contacts | cavity 130, serosa 504, both 68, neither 365 |
| reliable | 890/931 (95.6%) |

**The partition invariant holds to 1.4e-14.** Each fibroid decomposes exactly into
`frac_intracavitary + percent_intramural + frac_extraserosal = 100%`. This is the cheapest
available check that the two reference surfaces are mutually consistent, and it is asserted per
fibroid rather than assumed.

**That 68.7% figure is the headline limitation of the whole dataset.** Two thirds of fibroids are
resolved by three slices or fewer, which is precisely where through-plane geometry is least
trustworthy — and it is why Phases 5 and 7 exist.

The 201 dropped components are 18% of all raw components. The 100 mm³ floor is a config value, not
a constant, and its effect is worth revisiting.

## Phase 4 — Deterministic FIGO derivation

`make figo` → `results/fibroids_figo.parquet`. **All 4 gates PASSED**, on the 890 reliable fibroids.

| type | n | % | median p | meaning |
|---|---|---|---|---|
| 0 | 1 | 0.1% | 0.0% | pedunculated intracavitary |
| 1 | 2 | 0.2% | 35.0% | submucosal, <50% intramural |
| 2 | 55 | 6.2% | 90.9% | submucosal, >=50% intramural |
| 3 | 55 | 6.2% | 99.1% | 100% intramural, contacts endometrium |
| 4 | 365 | 41.0% | 100.0% | intramural, contacts neither surface |
| 5 | 331 | 37.2% | 95.1% | subserosal, >=50% intramural |
| 6 | 67 | 7.5% | 27.9% | subserosal, <50% intramural |
| 7 | 66 | 7.4% | 0.0% | subserosal pedunculated |
| 8 | 0 | 0.0% | n/a | other (cervical/parasitic) — **not derivable from a mask** (F6) |

Plus **52 hybrids** written as two numbers (`2-5` ×27, `3-5` ×27, `3-6` ×8, `3-7` ×3, `1-6` ×3).

| gate | result |
|---|---|
| G1 submucosal count plausible, not near-zero | **58** (6.5%) in 52 patients — the earlier attempt collapsed to ~9 |
| G2 all 8 mask-derivable types present | **8/8** |
| G3 no type absorbs the cohort | largest (type 4) holds 41.0% |
| G4 both sides of 50% populated per family | 1=2, 2=55, 5=331, 6=67 |

### What the gate caught: F7

The first run of this phase **failed**, and correctly. It produced **0 type-1 fibroids in 931**.
The cause was that `percent_intramural` had been defined as a volume fraction, and the endometrial
cavity is a virtual space — median **2.9 cm³**, 2.0% of the uterus, against a median fibroid of
**44 cm³**. A fibroid cannot have >50% of its volume inside a lumen fifteen times smaller than
itself, so type 1 was unreachable by construction. Replacing the submucosal side with a
nearest-label compartment split (F7) recovered it.

### Honest reading of the 1-vs-2 split

**Type 1 has n=2.** The gate passes, but two cases cannot support any claim about type-1-vs-2
discrimination — and 1-vs-2 is the project's clinical hook. The direction is exactly what the known
reconstruction bias predicts: submucosal fibroids read *too intramural*, pushing type 1 into type 2.
Whether UMD genuinely contains this few, or the bias is consuming them, is a question for Phase 7,
not something to be asserted here.

**40 fibroids (4.5%) lie within 10 points of the 50% line** — the ambiguous population Phase 6 is
built to flag.

## Phase 5 — Uncertainty quantification

`make uncertainty` → `results/fibroids_uncertainty.parquet`. **All 4 gates PASSED.**

Perturbations are anisotropic by *mechanism*, not merely by magnitude, because the two axes carry
different kinds of doubt and are resolvable at different scales:

- **in-plane** — a true ellipsoidal dilation/erosion of ±1 mm, sub-voxel resolvable.
- **through-plane** — discrete jitter of the first and last slice the fibroid occupies.

The second is not a shortcut. On a 5 mm grid a morphological dilation of 1, 2.5 or 4 mm
through-plane changes **nothing**; 5 mm adds a whole slice at each end. It is a step function, and
the ±2.5 mm boundary shift that dominates the error budget falls entirely inside the dead zone. On
this grid, "the boundary lies within half a slice of where it was drawn" *is* the statement that
the end slice may or may not belong to the fibroid — so that is what gets sampled.

### The headline number is misleading; here is the honest one

Median CI width across all reliable fibroids is **7.4 points**. That figure should not be quoted.

**60.4% of fibroids sit pinned at 0% or 100% intramural**, where perturbing the mask cannot move
the answer off the boundary. Their intervals are narrow by *saturation*, not precision (median
**1.6** points), and averaging them in roughly halves the apparent uncertainty of the fibroids whose
value is genuinely in play (median **17.6** points).

Restricted to fibroids whose measurement can actually move:

| slices spanned | n | median CI width |
|---|---|---|
| 1 | 75.0 | **46.7** |
| 2 | 55.0 | **31.6** |
| 3 | 58.0 | **23.0** |
| 4 | 27.0 | **14.5** |
| 5 | 29.0 | **14.8** |
| 6+ | 108.0 | **5.4** |

Spearman ρ(slices, CI width) = **−0.720**; ρ(volume, CI width) = **−0.690**. Monotonic, and exactly
what the anisotropy predicts. **A single-slice fibroid whose value is in play carries a ~47-point
interval** — the measurement is not meaningful for it.

Gate G3 is therefore judged on unsaturated fibroids. Scored over everything, a saturation artifact
would stand in for a precision trend.

**These intervals are PRECISION, not accuracy.** They propagate segmentation-boundary and
reconstruction-parameter noise. They do **not** include the systematic reconstruction bias measured
on phantoms (up to ~23 points, directional). A narrow interval does not mean a correct number.

**127 of 890 (14.3%)** have intervals straddling the 50% line — the FIGO call is not determined by
the imaging for those.

## Phase 6 — Borderline flagging

`make borderline` → `results/fibroids_borderline.parquet`. **All 4 gates PASSED.**

**133 of 890 (14.9%)** flagged ambiguous. Among fibroids where the call actually matters — those
touching a surface, so 1-vs-2 or 5-vs-6 is live — **113 of 525 (21.5%)** are ambiguous.

| flagged because | n |
|---|---|
| the interval crosses 50% | 93 |
| both point estimate and interval | 34 |
| point estimate alone, within ±10 | 6 |

Most flags come from the *interval*, not the point estimate — the Phase 5 propagation earning its
place, since a point-estimate-only rule would have missed 93 of 133.

### Triage model, patient-grouped CV (base rate 0.149, n=890)

| model | ROC AUC | PR AUC | Brier |
|---|---|---|---|
| logistic | 0.640 | 0.211 | 0.233 |
| gradient_boosting | 0.613 | 0.213 | 0.132 |
| random_forest | 0.601 | 0.207 | 0.141 |

**This is weak.** ROC AUC 0.64 against a 0.149 base rate beats chance but is not a dependable triage
tool, and is reported as such rather than dressed up. `flatness` carries most of the signal
(0.044); four of the nine predictors have *negative* permutation importance,
i.e. they are noise.

**Framing.** The label derives from our own measurement, so the model learns *which geometries this
pipeline cannot resolve* — a useful "look again" signal, not clinical truth. Predictors are
restricted to features obtainable without the reference-surface reconstruction, enforced by
`assert_no_predictor_leakage()`; patient grouping is asserted inside every CV split rather than
assumed.

## Phase 7 — Robustness

`make robustness` → `results/robustness_*.csv`. **All 4 gates PASSED.**

The brief's plan — contrast the full cohort against a "native-isotropic subset" — was not runnable,
because that subset does not exist (F2). Through-plane sensitivity is established by simulation
instead: make the sampling worse on the data we have and measure what breaks. We can always degrade
sampling; we cannot manufacture slices never acquired.

### 1. Slice decimation (100 patients)

| drop | pairs | median \|Δp\| | p95 \|Δp\| | **FIGO flip rate** | crosses 50% |
|---|---|---|---|---|---|
| ×2 | 210.0 | 1.9 | **91** | **36%** | 13% |
| ×3 | 160.0 | 2.6 | **88** | **41%** | 16% |

**This is the most important result in the project.** Dropping every second slice — a change no
radiologist would consider drastic — flips the FIGO type for **36% of fibroids**, and 41% at every
third slice. The median shift is small (1.9 points) but the p95 is **~91 points**: the distribution
is not merely noisy, it has a catastrophic tail where a fibroid moves from one end of the scale to
the other.

Flip rate tracks how well the fibroid was sampled to begin with — 44.4% for ≤2 slices versus 32.6%
for ≥5 — but note the floor: **even well-sampled fibroids flip a third of the time.** The FIGO call
derived from a single anisotropic acquisition should be treated as one sample from a wide
distribution, not as a fixed property of the patient.

### 2. Slice-count strata (the isotropic subset's replacement)

| subset | n | median p | median CI width | % borderline |
|---|---|---|---|---|
| all fibroids | 890 | 100.0% | 7.44 | 14.9% |
| >= 4 slices | 250 | 95.2% | 5.71 | 6.0% |
| >= 6 slices | 159 | 95.4% | 3.62 | 7.5% |

The ≥6-slice subset is what we consider trustworthy: CI width halves (7.44 → 3.62) and the
ambiguous fraction halves (14.9% → 7.6%). It is **159 fibroids, 18% of the cohort.**

### 3. Batch check — `_seg` vs `_seq` (closes deferred item O1)

| group | fibroids | median p | median volume (mm³) | % submucosal |
|---|---|---|---|---|
| `seg` | 710 | 100.0% | 974 | 6.6% |
| `seq` | 180 | 96.9% | 620 | 6.1% |

**O1 resolved.** After DICOM spacing recovery the two groups measure alike — median
percent-intramural 100.0 vs 96.9, submucosal rate 7% vs 6%. The `_seg`/`_seq` split is an export-tooling
artifact, not a clinical or scanner stratum, and needs no separate treatment.

One residual difference worth recording: the 33 `_seq` patients carry **180 fibroids (5.5 each)**
against 2.7 each for `_seg`. They are more fibroid-dense than the rest of the cohort, so they are
not a random sample of it.

### 4. Reconstruction sensitivity

Sweeping the Phase 2 bridging radius over 1.0–2.0× moves percent-intramural by a median of only
**0.1 points**, but changes the 50% call for **4 of 35 fibroids (11.4%)**. The reconstruction
parameter is not a major error source compared with slice sampling — but for fibroids already near
the threshold, it is not negligible either.

## Phase 8 — Reporting

`make demo` runs the full pipeline on synthetic phantoms with known FIGO geometry: **16/16 correct**,
a clean diagonal confusion matrix, no dataset or download required.

`make report` writes `reports/figures/`:

| figure | shows |
|---|---|
| `percent_intramural_distribution.png` | cohort distribution against the 50% line |
| `figo_distribution.png` | per-type counts, highlighting the four types the threshold decides |
| `uncertainty.png` | saturated vs live fibroids; CI width against sampling (ρ = −0.72) |
| `borderline.png` | which fibroids the imaging does not determine |

plus 24 reconstruction QC overlays in `reports/body_qc/`.

Palette and encodings follow the project's data-viz method; the categorical pair was validated
programmatically (worst adjacent CVD ΔE 24.7, normal-vision 33.6, all six checks pass) rather than
chosen by eye.
