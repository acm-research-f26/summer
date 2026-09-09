# FIGO Type & Percent-Intramural Measurement from Uterine Fibroid MRI Masks
## Build Plan — phased, gated, CPU-only

**Repo root:** `~/ACM/Sem 2/finalimp`
**Framing (non-negotiable):** this is a **measurement + uncertainty study**, not a FIGO classifier.
We compute the defining geometric quantity (`percent_intramural`), derive FIGO deterministically
from it, and quantify how confident each near-50% call is. "ML predicts FIGO from geometry" is
circular and is explicitly out of scope.

---

## FINDINGS FROM DATA VERIFICATION (read before Phase 0)

Verified directly against the 300-patient UMD copy. Three corrections to the original brief:

### F1 — The 33 outlier files are `_seq`, not mislabeled `_seg`
| pattern | n | container |
|---|---|---|
| `*_seg.nii.gz` | 267 | gzip |
| `*_seq.nii.gz` | **33** | **raw NIfTI, uncompressed despite `.gz`** |
| `*_t2.nii.gz` | 300 | gzip |

Two stacked traps: the filename is `_seq` (dataset typo) **and** those same 33 are uncompressed.
A loader globbing only `_seg` silently drops 33 patients before compression ever matters.
**The loader must glob `*_se[gq].nii.gz` and sniff the gzip magic (`1f 8b`) rather than trust the extension.**

### F2 — THERE IS NO NATIVE-ISOTROPIC SUBSET. The brief is wrong on this point.
The brief says ~258 fibroids were acquired natively isotropic (~1 mm³) and instructs us to use
them as the high-confidence subset. **This does not exist.** What is actually true:

- Exactly 33 patients have NIfTI `pixdim == (1.0, 1.0, 1.0)` — and they are *exactly* the 33 `_seq` files.
- Their **DICOM headers** (present in-repo, `*.dcm`) report the truth:
  `PixelSpacing = 0.446 × 0.446 mm`, `SliceThickness = 4–5 mm`, `SpacingBetweenSlices = 5–5.5 mm`,
  `MRAcquisitionType = 2D`, series `T2W_sag` / `T2W_TSE`.
- So the `(1,1,1)` is an **unset default pixdim** written by whatever tool re-exported those raw files.
  They are ordinary anisotropic 2D sagittal T2 scans, same as the other 267.

Consequence: **every one of the 300 patients is severely anisotropic.** Header anisotropy across
the cohort: median 13.3×, max 21.1×. The `< 2×` count is 33 — and all 33 are the fake ones.

**Two things follow, and both are mandatory:**
1. Voxel spacing for the 33 `_seq` patients **must be recovered from their DICOM headers**, never
   read from the NIfTI. **Corrected in Phase 1:** the damage is *not* the ~100× volume blow-up
   first assumed. True voxel volume for those 33 is 0.83–1.29 mm³ (mean 1.02), so `(1,1,1)` is
   accidentally right on volume to within ~20% — which is precisely what makes it dangerous, since
   a volume sanity-check sails straight past it. What it destroys is **anisotropy: true 11.6× vs
   claimed 1.0×**, corrupting every mm-sized structuring element, diameter, sphericity, and the
   Phase 5 perturbation.
2. The brief's "full cohort vs. native-isotropic subset" robustness pillar **is dead as written**
   and is replaced in Phase 7 by a *simulated* through-plane sensitivity analysis.

### F3 — A second naming defect: one `_seg` file is raw too, with a typo'd stem
`UMD_221129_037` contains **`UMD_2221129_037_seg.nii.gz`** — an extra digit — and is
raw-uncompressed as well. So **34 files are raw**, not 33, and the raw/gzip split does *not* line
up with the `_seg`/`_seq` split:

| | gzip | raw |
|---|---|---|
| `_seg` | 266 | **1** |
| `_seq` | 0 | 33 |

Any rule that infers the container from the filename is wrong. Sniff the magic number.
Discovered in Phase 1 only because the loader already sniffed rather than inferred.

### F4 — DICOM is present for all 300 patients (6,845 `.dcm` files)
This is what makes F2 fixable, and it gives us an independent check on spacing for all 300,
not just the 33. Do not delete the DICOMs.

### F5 — THE BRIEF'S `percent_intramural` FORMULA IS DEGENERATE (found in Phase 2)
The brief specifies

```
percent_intramural = 100 × |F ∩ body ∩ ¬C| / |F|,   body = fill(close(W ∪ C ∪ F))
```

This returns **100% for every fibroid, under every reconstruction method.** Two facts combine:

1. The masks are a **hard partition** — a voxel is wall *or* cavity *or* fibroid, never two. So
   wherever the fibroid sits, the cavity is simply not labelled: `F ∩ ¬C ≡ F`, identically.
2. `body` is built from a union that **includes `F`**, so `F ∩ body ≡ F` as well.

The expression collapses to `100 × |F|/|F|`. Verified on real patients before writing any
reconstruction code, and locked by a regression test (`test_the_briefs_formula_is_degenerate`).

**This re-diagnoses the earlier failed attempt.** The convex hull was blamed for collapsing the
submucosal count to ~9. The hull did make things worse, but with this formula the count collapses
under morphological closing *just as completely* — every fibroid reads 100% intramural, so every
fibroid is type 3/4. The formula was the root cause; the hull was an aggravating factor.

**The corrected measurement — reference surfaces.** A fibroid displaces the very surfaces it should
be measured against: a submucosal one dents the cavity, a subserosal one bulges the serosa. So each
surface is reconstructed as it would be *without that distortion*:

| surface | built from | why it works |
|---|---|---|
| `body_ref` | `fill(close(W ∪ C ∪ other fibroids))` — **this fibroid excluded** | an intramural fibroid is an enclosed hole that fill-holes restores (counts as inside); a subserosal one is not enclosed, so the protruding part stays outside |
| `cavity_ref` | `fill(close(C))` | closing bridges the dent a submucosal fibroid presses into the endometrium |

```
percent_intramural = 100 × |F ∩ body_ref ∩ ¬cavity_ref| / |F|
```

**The bridging radius must scale with the fibroid.** A partially buried fibroid cuts a crater in
the myometrium as wide as the fibroid itself. A fixed 5 mm closing leaves a 14 mm crater wide open,
the serosa reference caves in around it, and a textbook type 5 measures **1.7% intramural**. Using
the fibroid's equivalent-sphere radius (× `adaptive_radius_scale`) fixes it — the same phantom then
reads 71%, against an analytic truth of 81%.

### F7 — percent-intramural as a VOLUME FRACTION is ill-posed on the submucosal side
Found by the Phase 4 correctness gate, which is exactly what that gate exists for.

The first cohort run produced **0 type-1 fibroids out of 931**, and only 2 of 110 cavity-touching
fibroids with an intracavitary fraction above 50%. The serosal side was healthy at the same time
(median 14% extraserosal, 132 of 464 above 50%), and that asymmetry is the tell.

The cause is anatomical, not a coding bug:

| | median |
|---|---|
| endometrial cavity volume | **2.9 cm³** — 2.0% of the uterus |
| fibroid volume | **44 cm³** — 15× larger |

The endometrial cavity is a **virtual space**. A fibroid cannot have >50% of its volume "inside" a
lumen fifteen times smaller than itself, so **FIGO type 1 was unreachable by construction**. The
serosal side has no such problem because the serosa bounds real tissue against nothing.

Clinically the submucosal grade was never a lumen overlap: it is the proportion of the fibroid
protruding past the **endometrial surface**, which the fibroid itself displaces.

**Fix — nearest-label compartment split.** The grade asks a counterfactual: *what tissue would
occupy this space if the fibroid were not here?* Each fibroid voxel is assigned to whichever
compartment is nearer in millimetres, the annotated endometrium or the myometrium. No surface fit,
no orientation guess, and it degrades gracefully at both extremes. The serosal fraction is
unchanged. The three fractions still partition the fibroid exactly.

**A rejected alternative, recorded so it is not retried.** Fitting a local plane to the endometrium
fails on the *least* ambiguous case: a wholly intracavitary fibroid is ringed by endometrium on all
sides, so there is no base rim to fit — the plane passes through the fibroid's own centre and
splits it 50/50.

**Cost, stated plainly.** Nearest-label places the boundary midway between the annotated cavity and
myometrium rather than on the true surface. Worst phantom magnitude error rose from 12.8 to **23.4
points** (type 1 reads 33% against a truth of 10%), though all 8 phantoms still derive the correct
FIGO type. This is a real accuracy trade taken to make the submucosal grade reachable at all, and
it matters because the 50% threshold is the project's clinical hook.

### F8 — contact detection must be resolution-aware
Type 3 and type 4 differ only by whether the fibroid abuts the endometrium — a sub-voxel
distinction. A fixed 1.0 mm contact threshold made it a coin-flip on voxel size: the same phantom
measured **0.968 mm** from the endometrium at 0.484 mm in-plane (contact → type 3) and **1.488 mm**
at 0.496 mm (no contact → type 4), because tangent surfaces are separated by a discretisation gap.
Contact is now `max(1.0 mm, 3 in-plane voxels)`, the floor set by the annotation's own boundary
uncertainty. The 3-vs-4 call remains inherently marginal and is reported as such.

### Verified label semantics (spot-checked)
`0=background, 1=uterine wall, 2=cavity, 3=myoma, 4=nabothian cyst` — confirmed; label 4 is
sparse and absent in many patients. `_seg` dtype `uint16`, `_seq` dtype `int16`.

### Environment
System `python3` is 3.9.6. Homebrew `python3.13` is available and is what we will build the venv from.
Free disk at plan time: ~31 GiB (dataset copy is 7.3 GiB).

---

## Repo layout to build (Phase 0)

```
finalimp/
├── README.md               # what this is, clinical framing, how to run
├── PLAN.md                 # this file
├── RESULTS.md              # written progressively, one section per phase
├── Makefile                # the targets below
├── requirements.txt
├── config/
│   └── default.yaml        # thresholds: eps contact, 50% boundary, borderline band, closing radius
├── data/
│   ├── UMD/UMD/UMD_<id>/   # the copied dataset (DONE)
│   └── demo/               # synthetic masks, generated, git-ignored
├── src/figomeas/
│   ├── __init__.py
│   ├── io.py               # robust NIfTI loader (F1), DICOM spacing recovery (F2)
│   ├── manifest.py         # per-patient inventory + spacing truth table
│   ├── body.py             # uterine-body reconstruction (the hard part)
│   ├── features.py         # per-fibroid geometry + percent_intramural
│   ├── figo.py             # deterministic FIGO rules, incl. hybrids
│   ├── uncertainty.py      # mask perturbation -> CI
│   ├── borderline.py       # ambiguity flag + patient-grouped model
│   ├── robustness.py       # through-plane decimation study
│   ├── synthetic.py        # hand-built masks at known FIGO positions
│   └── viz.py              # overlays, distributions, QC contact sheets
├── scripts/                # thin CLI wrappers, one per make target
├── tests/                  # pytest; one test per FIGO rule + body reconstruction
├── results/                # tables (csv/parquet), git-ignored
└── reports/                # figures + QC sheets, git-ignored
```

---

# PHASES

Each phase has an **exit gate**. Do not start phase *N+1* until phase *N*'s gate passes.

---

## Phase 0 — Skeleton & environment
**Goal:** an installable, testable, empty-but-runnable repo.

- `git init`; `.gitignore` for `data/`, `results/`, `reports/`, `.venv/`, `__pycache__/`.
- venv from `python3.13`; `requirements.txt` = numpy, scipy, scikit-image, scikit-learn,
  nibabel, pydicom, pandas, pyarrow, matplotlib, pyyaml, pytest, tqdm.
- `Makefile` with all targets stubbed; `config/default.yaml` with every threshold named.
- `make setup` and `make test` both run green (zero tests is fine).

**Exit gate:** `make setup && make test` succeeds from a clean clone.

---

## Phase 1 — Data setup, robust I/O, and the spacing truth table
**Goal:** all 300 patients load, with *correct* spacing. This phase exists because of F1 and F2.

- `io.load_mask(path)` — glob `*_se[gq].nii.gz`, sniff gzip magic, fall back to raw NIfTI.
- `io.dicom_spacing(patient_dir)` — read `PixelSpacing`, `SliceThickness`, `SpacingBetweenSlices`,
  `MRAcquisitionType`, `SeriesDescription` from the first `.dcm`; prefer `SpacingBetweenSlices`
  over `SliceThickness` for `sz` when both exist (gaps are real).
- `io.resolve_spacing(patient)` — **DICOM is authoritative.** Use the NIfTI pixdim only when DICOM
  is missing. Record which source won, per patient, in the manifest.
- `manifest.build()` → `results/manifest.csv`: patient_id, mask_path, container (gzip|raw),
  filename_kind (seg|seq), shape, nifti_spacing, dicom_spacing, spacing_used, spacing_source,
  n_dcm, labels_present, n_fibroid_components, anisotropy_ratio.

**Exit gate (hard):**
1. 300/300 patients load. Zero silent drops.
2. Every patient's `labels_present ⊆ {0,1,2,3,4}`.
3. **All 33 `_seq` patients have `spacing_source == "dicom"`** and a plausible `sz` in 4.0–6.0 mm.
4. Zero patients end up with `spacing_used == (1.0, 1.0, 1.0)`.
5. Cohort anisotropy median lands ~10–13× with **no** sub-2× outliers remaining.

---

## Phase 2 — Uterine body reconstruction  ← *the phase that killed earlier attempts*
**Goal:** a serosa-bounded uterine solid that does **not** over-inflate.

The masks are a hard partition (a voxel is wall *or* fibroid *or* cavity), so `percent_intramural`
can never come from "fibroid voxels also labelled wall" — that is identically 0. The body must be
reconstructed:

- `body.reconstruct(mask, spacing, cfg, fibroid=...)`:
  - scaffold `wall ∪ cavity ∪ other fibroids` — **the measured fibroid is withheld** (F5)
  - **anisotropy-aware** morphological closing — structuring element sized in *millimetres*, so it
    is near-1 voxel through-plane and several voxels in-plane. A cubic voxel element is wrong here.
  - `binary_fill_holes`, keep the largest connected component
  - **NO convex hull.** A previous attempt used one: it over-inflated the body, swallowed the
    cavity, and collapsed the submucosal count to an implausible ~9.
  - fallback/comparator: alpha shape, for the Phase 2 ablation.
- Derive `serosa_surface` (outer boundary of the body) and `endometrial_surface` (boundary of label 2).
- Write QC overlays for ~24 patients: original labels vs. reconstructed body, mid-sagittal + 3 slices.

**Exit gate:**
1. Body ⊇ (wall ∪ cavity ∪ fibroid) for every patient, and body volume ≤ **1.5×** the union volume
   (the hull comparator should visibly blow past this — record the number for the ablation).
2. Cavity is **not** swallowed: `cavity ∩ body == cavity`, and the cavity remains a distinct
   interior void, not absorbed into solid wall.
3. Body is a single connected component in ≥95% of patients.
4. Eyeball the 24 QC overlays. This gate is visual and is not optional.
5. Ablation table recorded: closing vs. fill-holes-only vs. alpha-shape vs. convex-hull —
   body volume ratio + resulting submucosal count for each.

---

## Phase 3 — Per-fibroid feature extraction
**Goal:** one row per fibroid, with `percent_intramural` as the headline column.

- 3D connected components on `mask == 3`; drop components below a configured `min_volume_mm3`
  (config, not hard-coded) and log how many were dropped.
- Per fibroid: `volume_voxels`, `volume_mm3`, `n_slices_spanned`, `max_diameter_mm`,
  PCA elongation/flatness, `solidity`, `sphericity` (marching cubes, spacing-aware),
  `touch_cavity`, `touch_serosa` (contact = surface-voxel adjacency above an `eps` threshold, config).
- **`percent_intramural = 100 × |fibroid ∩ body ∩ ¬cavity| / |fibroid|`**
- Output `results/fibroids.parquet` + `.csv`.

**Exit gate:**
1. Total fibroid count is in the plausible 900–1,300 range (dataset descriptor says ~1,077).
2. `percent_intramural ∈ [0, 100]` for every row; no NaNs.
3. `volume_mm3` distribution is clinically sane (median on the order of ~1–20 cm³; no 100× outliers
   — a 100× error here is the F2 spacing bug resurfacing).
4. The `percent_intramural` histogram is inspected and is not degenerate (not all-0 / all-100).

---

## Phase 4 — Deterministic FIGO derivation + the correctness gate
**Goal:** FIGO type per fibroid, and proof the geometry is right.

- `figo.derive(row, cfg)` implementing 0–8 + hybrids (`touch_cavity AND touch_serosa` → `"2-5"` style).
- Type 8 (cervical/parasitic) behind a config flag: bucket or exclude.
- `tests/` builds tiny synthetic masks at each known FIGO position and asserts the derived type.
  **Write these tests before trusting any cohort number.**

**F6 — type 8 is not derivable from a UMD mask, and the original gate was unsatisfiable.**
The plan first required "all 9 types are represented". Type 8 is *other* — cervical, parasitic,
broad-ligament — and the UMD label scheme is `{background, wall, cavity, myoma, nabothian}`. There
is no cervix label and no landmark separating cervix from corpus, so a cervical fibroid is
geometrically identical to a low intramural one. Deriving type 8 would mean inventing an anatomical
boundary the data does not contain. The gate is therefore **8 derivable types**, with type 8
reported as not-assessable. This is a limitation of the dataset, not of the method, and belongs in
the write-up as such.

**Exit gate (this is the project's main correctness gate):**
1. Every FIGO rule has a passing synthetic unit test, including both hybrid orderings and both
   sides of the 50% boundary.
2. **Submucosal (types 0/1/2) count is clinically plausible — NOT near-zero.** A single-digit
   submucosal count means the Phase 2 reconstruction is wrong; go back to Phase 2. Do not proceed.
3. All **8 mask-derivable** types are represented (see F6); no single type absorbs >80% of the
   cohort; both sides of the 50% line are populated in each family.
4. Per-type distribution written to `RESULTS.md`.

**Additional invariant introduced in Phase 3.** Each fibroid is partitioned into three fractions
measured against the same reference surfaces — `frac_intracavitary`, `percent_intramural`,
`frac_extraserosal` — which must sum to exactly 100%. Asserted per fibroid. This is the cheapest
check that the two reference surfaces are mutually consistent, and it is what would have exposed
the degenerate formula (F5) instantly.

---

## Phase 5 — Uncertainty quantification
**Goal:** a confidence interval on every fibroid's `percent_intramural`.

- Perturb the mask ±1–2 voxels (erode/dilate) — **in mm, anisotropy-aware**, so a "1 voxel"
  through-plane perturbation is understood as a 5 mm perturbation, which is the dominant error term.
- Also perturb the body reconstruction (closing radius) and re-run.
- N resamples per fibroid (config); report median, 2.5/97.5 percentiles, and the width of the band.

**Exit gate:** every fibroid has a CI; the CI width is reported as a function of `volume_mm3` and
`n_slices_spanned`, and the expected relationship holds (small / few-slice fibroids are the most
uncertain). If it does not hold, the perturbation is not anisotropy-aware.

---

## Phase 6 — Borderline flagging
**Goal:** the clinical deliverable — flag the fibroids where the 1-vs-2 / 5-vs-6 call is genuinely ambiguous.

- `is_borderline` = point estimate within a configured band of 50%, **or** CI straddles 50%.
- Small classical model (logistic regression → random forest / gradient boosting) predicting
  `is_borderline` from shape features that a radiologist could assess *without* the body
  reconstruction. **Patient-grouped CV — no patient's fibroids split across folds.**
- Report AUC / PR-AUC / calibration, plus which features carry the signal.

**Exit gate:** grouped-CV metrics reported with the grouping *proven* (assert no patient id appears
in both train and test). Framed as "which cases need a second look," never as "predicts FIGO."

---

## Phase 7 — Robustness  *(revised — replaces the dead isotropic-subset plan, see F2)*
Since no isotropic subset exists, we establish through-plane sensitivity by **simulation** instead:

- **Through-plane decimation:** drop every 2nd / 3rd slice, re-run the full pipeline, measure how
  `percent_intramural` and the FIGO call drift. This directly bounds "how much does slice thickness
  cost us?" using the data we actually have.
- **Slice-count restriction:** report everything on the subset of fibroids spanning ≥4 and ≥6 slices;
  these are the trustworthy numbers, and they replace the "isotropic subset" role in the write-up.
- **Spacing-source split:** `_seg` (267) vs `_seq` (33) as a batch-effect check.
- **Reconstruction sensitivity:** re-run with the Phase 2 ablation variants; report FIGO-call churn.

**Exit gate:** a table of FIGO-call flip rate under each perturbation, and an explicit statement of
which cohort slice we consider trustworthy.

---

## Phase 8 — Reporting
- `make demo` — full pipeline on synthetic masks, no download, runs in under a minute.
- Figures: `percent_intramural` distribution; CI width vs. size; per-type counts; borderline
  scatter around the 50% line; QC overlay contact sheet.
- `RESULTS.md` finalized; `README.md` with the honest-limitations section:
  geometry-derived labels (no independent clinician FIGO ground truth ships with UMD),
  universal anisotropy, and requires-a-mask deployment scope.

---

## Standing constraints
- **CPU only.** No neural nets, no auto-segmentation, no super-resolution / plane synthesis.
  The 3D mask already carries full 3D information.
- Every threshold lives in `config/default.yaml`. No magic numbers in `src/`.
- Every phase appends to `RESULTS.md` as it completes.
- Deterministic: seed everything; same input → same table.

---

## DEFERRED / OPEN ITEMS (revisit after Phase 8)

- **O1 — the 33 `_seq` patients.** Phase 1 recovers their true spacing from DICOM because using
  `(1,1,1)` would be flatly wrong. What is *deferred* is the wider question: does the brief's
  "native-isotropic subset" claim have any salvageable basis (a different UMD release? a subset
  we haven't identified?), and does the `_seg`/`_seq` split correlate with anything real
  (scanner, site, annotation batch) beyond the export tool? Re-evaluate once the full pipeline
  is complete and Phase 7's batch-effect check has data to speak from. Do not let this block
  Phases 1–8.
