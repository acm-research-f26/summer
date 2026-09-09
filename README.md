# FIGO Type & Percent-Intramural Measurement from Uterine Fibroid MRI Masks

**What this is.** A CPU-only medical-imaging *measurement* tool. Given expert segmentation masks
from pelvic MRI, it computes — for each individual fibroid — its **percent-intramural** value (how
deeply it is buried in the uterine muscle), derives the FIGO type deterministically from that
geometry, and quantifies how confident each measurement is.

**What this is not.** It is not a classifier and not a diagnosis. FIGO type is *defined* by
geometry, so "predicting" it from geometry would be circular and worthless. Radiologists can
already see whether a fibroid sits in the cavity, in the wall, or on the outer surface. What they
cannot reliably eyeball is whether it is exactly **≥50% vs <50% intramural** — the boundary that
separates FIGO type 1 from 2, and 5 from 6, and a documented source of inter-reader disagreement.
This tool supplies that exact number and flags the cases where the call is genuinely ambiguous.

**Why the 50% line matters.** Types 0/1/2 are submucosal and are the candidates for minimally
invasive hysteroscopic removal. Whether a submucosal fibroid is type 1 or type 2 predicts whether
that surgery will succeed. An objective measurement at that threshold is the clinical hook.

## Data

[UMD — Uterine Myoma MRI Dataset](https://figshare.com/articles/dataset/UMD_zip/23541312):
300 patients, sagittal T2-weighted pelvic MRI, pixel-level masks reviewed by 11 clinicians,
~1,077 individual fibroids, all 9 FIGO types.
Mask labels: `0=background, 1=uterine wall, 2=cavity, 3=myoma, 4=nabothian cyst`.

Fetch it with `make data` (or `./scripts/download_umd.sh`): a resumable, md5-verified pull of the
4.76 GB `UMD.zip` from figshare, extracted to `data/UMD/` — the layout `config/default.yaml`
expects. Set `KEEP_ZIP=0` to drop the archive after extraction.

Two verified dataset traps this repo defeats (details in `PLAN.md`):
- **33 masks are named `_seq`, not `_seg`** — and are raw uncompressed NIfTI despite a `.gz`
  extension. A naive loader silently drops 11% of the cohort.
- **Those same 33 carry an unset `pixdim` of `(1,1,1)`.** Their DICOM headers show the truth:
  0.446 × 0.446 × ~5 mm. There is **no native-isotropic subset** in UMD; all 300 patients are
  severely anisotropic (median 13.3×). Spacing is therefore taken from DICOM, not from the NIfTI.

## Quick start

```bash
make setup     # venv (python3.13) + dependencies
make test      # unit tests
make demo      # end-to-end on synthetic masks, no dataset needed
make help      # all targets
```

Pipeline targets run in order: `manifest → extract → figo → uncertainty → borderline → robustness → report`.

## Limitations (stated up front)

Four of these were found by the pipeline's own correctness gates, not anticipated. Each would
otherwise have produced confident, plausible-looking, wrong numbers. Full detail in `PLAN.md`
(findings F1–F8) and `RESULTS.md`.

**Geometry-derived labels.** UMD ships no independent clinician FIGO annotations, so the reference
types here are computed from the masks. Nothing in this repo is validated against a radiologist's
call. This is the central limitation and it bounds every claim below.

**Universal anisotropy, and no isotropic subset.** The brief stated that ~258 fibroids were
acquired natively isotropic and should serve as the trustworthy subset. **That subset does not
exist.** The 33 patients whose NIfTI headers claim 1 mm isotropic voxels are exactly the 33
raw-written files with an unset `pixdim`; their DICOM headers report ~0.45 × 0.45 × 5 mm, 2D
acquisition. Every UMD patient is anisotropic (median 13.3×), **68.7% of fibroids span ≤3 slices**,
and through-plane geometry is the dominant error term throughout.

**Percent-intramural is not a single well-posed quantity on both sides.** On the serosal side a
volume fraction works: the serosa bounds tissue against nothing. On the submucosal side it does
not — the endometrial cavity is a virtual space (median 2.9 cm³, 2.0% of the uterus) against a
median fibroid of 44 cm³, so no fibroid can be >50% "inside" it and FIGO type 1 is unreachable by
construction. The submucosal side therefore uses a nearest-label compartment split instead, which
asks what tissue would occupy the space if the fibroid were absent. The two sides are measured by
different, individually defensible rules; this is a property of the anatomy, not a convenience.

**Known systematic bias, not covered by the confidence intervals.** On phantoms with analytically
known answers the error is directional — submucosal fibroids read too intramural, subserosal too
little — inflating types 2 and 6. Worst magnitude error is ~23 points. The Phase 5 intervals
measure **precision** (boundary and reconstruction-parameter noise) and do **not** include this
bias. A narrow interval does not mean an accurate measurement.

**Type 1 has n=2.** The 1-vs-2 distinction is the project's clinical hook, and the cohort supports
almost no inference about it. The direction is what the bias above predicts.

**Type 8 is not derivable.** Cervical and parasitic fibroids need a landmark UMD does not label —
there is no cervix label, so a cervical fibroid is geometrically identical to a low intramural one.
Eight of nine types are derived; type 8 is reported as not assessable rather than guessed.

**4.4% of fibroids have no reconstructable reference.** Where a fibroid dominates the uterine mass
no myometrial envelope survives (one is 918 cm³ against 9.4k wall voxels). These are flagged
`reliable=False` and excluded from every gate and statistic, never silently scored.

**Requires a mask.** Deployment on raw scans would need auto-segmentation (nnU-Net-class, GPU),
explicitly out of scope. The scope is *quantification on already-segmented scans*.

**CPU only.** No neural networks, no super-resolution, no plane synthesis. The 3D mask already
contains full 3D information.

See `PLAN.md` for the phased build plan and its findings, and `RESULTS.md` for results per phase.
