#!/usr/bin/env python3
"""PHASE 7 — robustness: slice decimation, slice-count strata, batch and radius checks.

The brief's plan was to contrast the full cohort against a "native-isotropic
subset". That subset does not exist (PLAN.md F2), so through-plane sensitivity is
established by simulation instead: make the sampling worse on the data we have
and measure what breaks.
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from figomeas import load_config
from figomeas.config import REPO_ROOT
from figomeas.io import load_mask
from figomeas.robustness import (_match_by_rank, measure_under_decimation,
                                 reconstruction_sensitivity, slice_count_strata,
                                 spacing_source_split)


def _one_patient(job):
    """Decimated measurements for one patient, compared against stored full-res.

    The full-resolution baseline is read from the Phase 3 table rather than
    recomputed. Re-extracting it here was a third of the sweep's compute *and* a
    third of its peak memory, on a 16 GB machine already 11 GB into swap -- and it
    would only reproduce numbers already on disk.
    """
    patient_id, mask_path, spacing, factors, base = job
    if not base:
        return []
    cfg = load_config()
    mask = load_mask(mask_path).data
    rows = []
    for factor in factors:
        if mask.shape[2] // factor < 2:
            continue
        other = measure_under_decimation(mask, spacing, cfg, factor)
        for b, o in _match_by_rank(base, other):
            rows.append({
                "patient_id": patient_id, "factor": factor,
                "volume_mm3": b["volume_mm3"],
                "n_slices_full": b["n_slices_spanned"],
                "p_full": b["percent_intramural"],
                "p_decimated": o["percent_intramural"],
                "delta_p": o["percent_intramural"] - b["percent_intramural"],
                "figo_full": b["figo"], "figo_decimated": o["figo"],
                "figo_flipped": b["figo"] != o["figo"],
                "crossed_50": (b["percent_intramural"] >= 50) != (o["percent_intramural"] >= 50),
            })
    return rows


def main() -> int:
    cfg = load_config()
    man = pd.read_csv(REPO_ROOT / "results" / "manifest.csv")
    src = REPO_ROOT / "results" / "fibroids_borderline.parquet"
    if not src.exists():
        src = REPO_ROOT / "results" / "fibroids_figo.parquet"
    df = pd.read_parquet(src)
    rel = df[df.reliable]

    limit = int(cfg["robustness.decimation_patients"])
    sample = man.head(limit) if limit else man
    factors = tuple(cfg["robustness.decimation_factors"])
    # Full-resolution baseline, straight from Phase 3.
    base_cols = ["volume_mm3", "n_slices_spanned", "percent_intramural", "figo"]
    by_patient = {pid: g[base_cols].to_dict("records")
                  for pid, g in df.groupby("patient_id")}
    jobs = [(r.patient_id, str(REPO_ROOT / r.mask_path),
             (r.used_sx, r.used_sy, r.used_sz), factors,
             by_patient.get(r.patient_id, [])) for r in sample.itertuples()]
    n_workers = max(1, min(int(cfg.get("compute.n_workers", 0)) or (os.cpu_count() or 2) - 2,
                           len(jobs)))

    print(f"=== 1. slice decimation ({len(jobs)} patients, factors {factors}) ===")
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for recs in tqdm(pool.map(_one_patient, jobs, chunksize=2),
                         total=len(jobs), desc="decimation", unit="pt"):
            rows.extend(recs)
    dec = pd.DataFrame(rows)
    dec.to_csv(REPO_ROOT / "results" / "robustness_decimation.csv", index=False)

    print()
    print(dec.groupby("factor").agg(
        n=("delta_p", "size"),
        median_abs_delta_p=("delta_p", lambda s: s.abs().median()),
        p95_abs_delta_p=("delta_p", lambda s: s.abs().quantile(0.95)),
        figo_flip_rate=("figo_flipped", "mean"),
        crossed_50_rate=("crossed_50", "mean"),
    ).to_string(float_format=lambda v: f"{v:.3f}"), "\n")

    print("FIGO flip rate by how well the fibroid was sampled to begin with:")
    dec2 = dec.assign(slices=dec.n_slices_full.clip(upper=6))
    print(dec2.groupby(["factor", "slices"]).agg(
        n=("figo_flipped", "size"), flip_rate=("figo_flipped", "mean")
    ).to_string(float_format=lambda v: f"{v:.3f}"), "\n")

    print("=== 2. slice-count strata (replaces the non-existent isotropic subset) ===")
    strata = slice_count_strata(rel, tuple(cfg["robustness.min_slices_trustworthy"]))
    print(strata.to_string(index=False, float_format=lambda v: f"{v:.2f}"), "\n")
    strata.to_csv(REPO_ROOT / "results" / "robustness_strata.csv", index=False)

    print("=== 3. batch check: _seg (267) vs _seq (33) ===")
    split = spacing_source_split(rel, man)
    print(split.to_string(index=False, float_format=lambda v: f"{v:.2f}"), "\n")
    split.to_csv(REPO_ROOT / "results" / "robustness_batch.csv", index=False)

    print("=== 4. reconstruction sensitivity (bridging radius) ===")
    scales = [1.0, 1.25, 1.5, 1.75, 2.0]
    sens_rows = []
    for r in tqdm(man.head(20).itertuples(), total=20, desc="radius", unit="pt", leave=False):
        mask = load_mask(REPO_ROOT / r.mask_path).data
        sp = (r.used_sx, r.used_sy, r.used_sz)
        lab, n = ndimage.label(mask == cfg["labels.fibroid"],
                               ndimage.generate_binary_structure(3, cfg["fibroid.connectivity"]))
        for i in range(1, n + 1):
            f = lab == i
            if f.sum() * np.prod(sp) < cfg["fibroid.min_volume_mm3"]:
                continue
            s = reconstruction_sensitivity(mask, f, sp, cfg, scales)
            sens_rows.append({"patient_id": r.patient_id, "component": i,
                              "p_range": s.percent_intramural.max() - s.percent_intramural.min(),
                              "crossed_50": (s.percent_intramural >= 50).nunique() > 1})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(REPO_ROOT / "results" / "robustness_radius.csv", index=False)
    print(f"  {len(sens)} fibroids, bridging radius swept {scales[0]}-{scales[-1]}x")
    print(f"  median percent_intramural range across scales: {sens.p_range.median():.1f} points")
    print(f"  fibroids whose 50% call changes: {int(sens.crossed_50.sum())}/{len(sens)} "
          f"({100*sens.crossed_50.mean():.1f}%)\n")

    d2 = dec[dec.factor == 2]
    gates = [
        ("G1  decimation actually degrades the measurement (the study measures something)",
         bool(d2.delta_p.abs().median() > 0.1),
         f"median |delta p| at factor 2: {d2.delta_p.abs().median():.2f} points"),
        ("G2  poorly sampled fibroids flip more often than well sampled ones",
         bool(dec2[dec2.slices <= 2].figo_flipped.mean()
              > dec2[dec2.slices >= 5].figo_flipped.mean()),
         f"<=2 slices {100*dec2[dec2.slices<=2].figo_flipped.mean():.1f}% vs "
         f">=5 slices {100*dec2[dec2.slices>=5].figo_flipped.mean():.1f}%"),
        ("G3  a trustworthy subset is identified and reported",
         bool(len(strata) >= 3 and strata.n.iloc[-1] > 0),
         f"{strata.subset.iloc[-1]}: n={strata.n.iloc[-1]}"),
        ("G4  batch effect between _seg and _seq is quantified",
         bool(len(split) >= 1), f"{len(split)} groups compared"),
    ]
    print("=== PHASE 7 EXIT GATES ===")
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    failed = [n for n, ok, _ in gates if not ok]
    print()
    if failed:
        print(f"PHASE 7 BLOCKED — {len(failed)} gate(s) failed.")
        return 1
    print("PHASE 7 EXIT GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
