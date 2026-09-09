"""PHASE 5 — per-fibroid confidence intervals on percent_intramural."""

# Cap per-process math threads BEFORE numpy is imported. On macOS numpy uses
# Accelerate, which spawns its own thread pool inside *every* worker process --
# 8 workers x ~4 threads drove load average to 31 on a 10-core machine, so the
# pool spent its time context-switching rather than computing.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from figomeas import load_config
from figomeas.config import REPO_ROOT
from figomeas.io import load_mask
from figomeas.uncertainty import confidence_interval, resample_fibroid


def _one_patient(job):
    patient_id, mask_path, spacing, seed = job
    cfg = load_config()
    mask = load_mask(mask_path).data
    rng = np.random.default_rng(seed)
    st = ndimage.generate_binary_structure(3, int(cfg["fibroid.connectivity"]))
    labelled, n = ndimage.label(mask == cfg["labels.fibroid"], structure=st)
    voxel_mm3 = float(np.prod(spacing))
    out = []
    for i in range(1, n + 1):
        f = labelled == i
        if f.sum() * voxel_mm3 < float(cfg["fibroid.min_volume_mm3"]):
            continue
        ci = confidence_interval(
            resample_fibroid(mask, f, spacing, cfg, rng), cfg)
        ci.update({"patient_id": patient_id, "component": i})
        out.append(ci)
    return out


def main() -> int:
    cfg = load_config()
    man = pd.read_csv(REPO_ROOT / "results" / "manifest.csv")
    fib = pd.read_parquet(REPO_ROOT / "results" / "fibroids_figo.parquet")

    seed = int(cfg["seed"])
    jobs = [(r.patient_id, str(REPO_ROOT / r.mask_path),
             (r.used_sx, r.used_sy, r.used_sz), seed + i)
            for i, r in enumerate(man.itertuples())]
    n_workers = max(1, min(int(cfg.get("compute.n_workers", 0)) or (os.cpu_count() or 2) - 2,
                           len(jobs)))
    print(f"resampling with {n_workers} workers, "
          f"{cfg['uncertainty.n_radius_draws']}x{cfg['uncertainty.n_mask_draws']} draws per fibroid")

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for recs in tqdm(pool.map(_one_patient, jobs, chunksize=2),
                         total=len(jobs), desc="uncertainty", unit="pt"):
            rows.extend(recs)

    ci = pd.DataFrame(rows)
    df = fib.merge(ci, on=["patient_id", "component"], how="left")
    df["figo_flips_under_uncertainty"] = df.ci_straddles_50 & (
        df.touch_cavity | df.touch_serosa)
    out = REPO_ROOT / "results" / "fibroids_uncertainty.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)

    rel = df[df.reliable]
    print(f"\n=== {len(df)} fibroids with confidence intervals -> {out.name} ===\n")
    print("CI width (percentage points):")
    print(rel.ci_width.describe().to_string(), "\n")

    print("CI width by fibroid size (reliable only):")
    bins = [0, 1000, 5000, 20000, 100000, np.inf]
    labels = ["<1", "1-5", "5-20", "20-100", ">100"]
    rel = rel.assign(size_band=pd.cut(rel.volume_mm3, bins=bins, labels=labels))
    print(rel.groupby("size_band", observed=True)
          .agg(n=("ci_width", "size"), median_width=("ci_width", "median"))
          .to_string(), "\n")

    print("CI width by through-plane extent (the dominant error term):")
    print(rel.assign(slices=rel.n_slices_spanned.clip(upper=6))
          .groupby("slices", observed=True)
          .agg(n=("ci_width", "size"), median_width=("ci_width", "median"))
          .to_string(), "\n")

    # A narrow interval is not the same as a precise measurement. 60% of fibroids
    # sit pinned at 0% or 100% intramural, where perturbing the mask cannot move
    # the answer off the boundary -- their CIs are narrow by saturation, not by
    # precision, and averaging them in halves the apparent uncertainty of the
    # fibroids whose value is actually in play.
    rel = rel.assign(saturated=(rel.percent_intramural > 99.5)
                     | (rel.percent_intramural < 0.5))
    print("CI width, split by whether the measurement can move at all:")
    print(rel.groupby("saturated").agg(n=("ci_width", "size"),
                                       median_ci_width=("ci_width", "median"),
                                       median_slices=("n_slices_spanned", "median"))
          .to_string(), "\n")
    uns = rel[~rel.saturated]
    print("Unsaturated fibroids only — CI width by through-plane extent:")
    print(uns.assign(slices=uns.n_slices_spanned.clip(upper=6))
          .groupby("slices", observed=True)
          .agg(n=("ci_width", "size"), median_width=("ci_width", "median"))
          .to_string())
    print(f"  Spearman rho(slices, CI width) = "
          f"{uns.n_slices_spanned.corr(uns.ci_width, method='spearman'):+.3f}\n")

    n_straddle = int(rel.ci_straddles_50.sum())
    print(f"CIs straddling the 50% line: {n_straddle}/{len(rel)} "
          f"({100*n_straddle/len(rel):.1f}%) -- the FIGO call is not determined for these\n")

    # Judged on unsaturated fibroids: including boundary-pinned ones lets a
    # saturation artifact stand in for a real precision trend.
    small = uns[uns.n_slices_spanned <= 2].ci_width.median()
    large = uns[uns.n_slices_spanned >= 5].ci_width.median()
    gates = [
        ("G1  every fibroid has a confidence interval",
         bool(df.ci_width.notna().all()), f"{int(df.ci_width.isna().sum())} missing"),
        ("G2  CIs are non-degenerate (perturbation actually moved the measurement)",
         bool(rel.ci_width.median() > 0.5),
         f"median width {rel.ci_width.median():.2f} points"),
        ("G3  few-slice fibroids are measurably less certain (unsaturated fibroids only)",
         bool(small > large),
         f"<=2 slices: {small:.2f} pts vs >=5 slices: {large:.2f} pts "
         f"(n={len(uns)} unsaturated of {len(rel)})"),
        ("G4  CI bounds are ordered and in range",
         bool((rel.ci_low <= rel.ci_high).all() and rel.ci_low.min() >= 0
              and rel.ci_high.max() <= 100),
         f"[{rel.ci_low.min():.1f}, {rel.ci_high.max():.1f}]"),
    ]
    print("=== PHASE 5 EXIT GATES ===")
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    failed = [n for n, ok, _ in gates if not ok]
    print()
    print("NOTE: these intervals are PRECISION, not accuracy. They propagate boundary")
    print("and reconstruction-parameter noise. The systematic reconstruction bias")
    print("measured on phantoms (up to ~13 points) is a separate term and is NOT")
    print("included here -- see RESULTS.md.\n")
    if failed:
        print(f"PHASE 5 BLOCKED — {len(failed)} gate(s) failed.")
        return 1
    print("PHASE 5 EXIT GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
