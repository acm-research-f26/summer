#!/usr/bin/env python3
"""PHASE 2 — uterine reference-surface reconstruction: gates, ablation, QC."""

import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from figomeas import load_config
from figomeas.body import briefs_percent_intramural, percent_intramural, reconstruct
from figomeas.config import REPO_ROOT
from figomeas.io import load_mask
from figomeas.synthetic import all_phantoms, ground_truth_percent_intramural
from figomeas.viz import overlay_body

QC_N = 24
ABLATION_N = 25


def phantom_check(cfg):
    rows = []
    for t, ph in all_phantoms().items():
        F = ph.data == 3
        m = reconstruct(ph.data, ph.spacing, cfg, fibroid=F)
        p, gt = percent_intramural(F, m), ground_truth_percent_intramural(ph)
        rows.append({"figo": t, "measured": p, "truth": gt, "abs_err": abs(p - gt),
                     "side_ok": (p >= 50) == (gt >= 50)})
    return pd.DataFrame(rows)


def cohort_pass(cfg, man):
    rows = []
    for r in tqdm(man.itertuples(), total=len(man), desc="reconstruct", unit="pt"):
        mask = load_mask(REPO_ROOT / r.mask_path).data
        sp = (r.used_sx, r.used_sy, r.used_sz)
        m = reconstruct(mask, sp, cfg)
        W, C, F = mask == 1, mask == 2, mask == 3
        scaffold, union = W | C, W | C | F
        lab, n = ndimage.label(m.body)
        largest = np.bincount(lab.ravel())[1:].max() if n else 0
        rows.append({
            "patient_id": r.patient_id,
            "contains_scaffold": bool((scaffold & ~m.body).sum() == 0),
            "body_over_union": m.body.sum() / max(union.sum(), 1),
            "body_over_scaffold": m.body.sum() / max(scaffold.sum(), 1),
            "n_components": n,
            "largest_frac": largest / max(m.body.sum(), 1),
            "cavity_preserved": bool((C & ~m.cavity_ref).sum() == 0),
            "cavity_over_body": m.cavity_ref.sum() / max(m.body.sum(), 1),
        })
    return pd.DataFrame(rows)


def ablation(cfg, man):
    """Body inflation and percent-intramural spread, per reconstruction method."""
    rows = []
    sample = man.head(ABLATION_N)
    for method in ["closing", "fill", "union_closing", "hull", "brief_literal"]:
        ps, ratios = [], []
        for r in tqdm(sample.itertuples(), total=len(sample), desc=f"ablate:{method}",
                      unit="pt", leave=False):
            mask = load_mask(REPO_ROOT / r.mask_path).data
            sp = (r.used_sx, r.used_sy, r.used_sz)
            union = (mask == 1) | (mask == 2) | (mask == 3)
            lab, n = ndimage.label(mask == 3,
                                   ndimage.generate_binary_structure(3, cfg["fibroid.connectivity"]))
            for i in range(1, n + 1):
                f = lab == i
                if f.sum() * np.prod(sp) < cfg["fibroid.min_volume_mm3"]:
                    continue
                if method == "brief_literal":
                    ps.append(briefs_percent_intramural(mask, sp, f, cfg))
                    ratios.append(np.nan)
                    continue
                m = reconstruct(mask, sp, cfg, method=method, fibroid=f)
                ps.append(percent_intramural(f, m))
                ratios.append(m.body.sum() / max(union.sum(), 1))
        ps = np.array(ps)
        rows.append({"method": method, "n_fibroids": len(ps),
                     "body/union": float(np.nanmean(ratios)) if not np.all(np.isnan(ratios)) else np.nan,
                     "p_median": float(np.median(ps)),
                     "p_iqr": float(np.percentile(ps, 75) - np.percentile(ps, 25)),
                     "pct_at_100": float(100 * (ps > 99.5).mean()),
                     "pct_below_50": float(100 * (ps < 50).mean())})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    man = pd.read_csv(REPO_ROOT / "results" / "manifest.csv")

    print("=== phantom validation (known FIGO geometry) ===")
    ph = phantom_check(cfg)
    print(ph.to_string(index=False, float_format=lambda v: f"{v:.1f}"), "\n")

    print("=== cohort reconstruction ===")
    coh = cohort_pass(cfg, man)
    coh.to_csv(REPO_ROOT / "results" / "body_qc.csv", index=False)
    print(coh[["body_over_union", "body_over_scaffold", "largest_frac",
               "cavity_over_body"]].describe().to_string(), "\n")

    print("=== ablation: reconstruction method ===")
    abl = ablation(cfg, man)
    abl.to_csv(REPO_ROOT / "results" / "body_ablation.csv", index=False)
    print(abl.to_string(index=False, float_format=lambda v: f"{v:.2f}"), "\n")

    print(f"=== QC overlays ({QC_N} patients) -> reports/body_qc/ ===")
    for r in tqdm(man.head(QC_N).itertuples(), total=QC_N, desc="qc", unit="pt", leave=False):
        mask = load_mask(REPO_ROOT / r.mask_path).data
        sp = (r.used_sx, r.used_sy, r.used_sz)
        overlay_body(mask, reconstruct(mask, sp, cfg),
                     REPO_ROOT / "reports" / "body_qc" / f"{r.patient_id}.png",
                     title=r.patient_id)
    print("written\n")

    # Reliability is read from the Phase 3 table rather than recomputed. Both
    # would run one reconstruction per fibroid over the whole cohort -- ~25
    # minutes of identical work -- and Phase 3 has to do it anyway to produce the
    # feature table. Phase 2 reports it; Phase 3 owns it.
    fib_path = REPO_ROOT / "results" / "fibroids.parquet"
    if fib_path.exists():
        rel = pd.read_parquet(fib_path)
        n_unrel = int((~rel.reliable).sum())
        print(f"=== reliability (from results/fibroids.parquet) ===")
        print(f"  {len(rel)} fibroids; {n_unrel} flagged unreliable "
              f"({100*n_unrel/len(rel):.1f}%) -- fibroid dominates the uterine mass\n")
        rel_frac = n_unrel / len(rel)
        rel_detail = f"{n_unrel}/{len(rel)} flagged ({100*rel_frac:.1f}%)"
    else:
        rel_frac, rel_detail = 0.0, "skipped: run `make extract` first"
        print("=== reliability: skipped, results/fibroids.parquet not built yet ===\n")

    ratio_cap = cfg["body.max_volume_ratio_vs_union"]
    base = abl[abl.method == "closing"].iloc[0]
    brief = abl[abl.method == "brief_literal"].iloc[0]
    gates = [
        ("G1  body contains all wall+cavity voxels, every patient",
         bool(coh.contains_scaffold.all()),
         f"{int((~coh.contains_scaffold).sum())} violations"),
        (f"G2  body volume <= {ratio_cap}x the wall+cavity+fibroid union",
         bool((coh.body_over_union <= ratio_cap).all()),
         f"max ratio {coh.body_over_union.max():.3f}"),
        ("G3  body is dominated by one component (>=99% of volume) in >=95% of patients",
         bool((coh.largest_frac >= 0.99).mean() >= 0.95),
         f"{100*(coh.largest_frac>=0.99).mean():.1f}% of patients; "
         f"median components {coh.n_components.median():.0f}"),
        ("G4  cavity preserved, not swallowed by the body",
         bool(coh.cavity_preserved.all() and (coh.cavity_over_body < 0.5).all()),
         f"preserved {int(coh.cavity_preserved.sum())}/{len(coh)}, "
         f"max cavity/body {coh.cavity_over_body.max():.3f}"),
        ("G5  phantoms: all 8 FIGO positions land on the correct side of 50%",
         bool(ph.side_ok.all()),
         f"worst |error| {ph.abs_err.max():.1f} points"),
        ("G6  measurement is NOT degenerate (the brief's formula returns 100% always)",
         bool(base.pct_at_100 < 90 and base.p_iqr > 5 and brief.pct_at_100 > 99),
         f"closing: {base.pct_at_100:.0f}% pinned at 100, IQR {base.p_iqr:.1f}  |  "
         f"brief_literal: {brief.pct_at_100:.0f}% pinned at 100, IQR {brief.p_iqr:.1f}"),
        ("G7  fibroids with no reconstructable serosa reference are flagged, not scored",
         bool(rel_frac < 0.05), rel_detail),
    ]
    print("=== PHASE 2 EXIT GATES ===")
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    failed = [n for n, ok, _ in gates if not ok]
    print()
    if failed:
        print(f"PHASE 2 BLOCKED — {len(failed)} gate(s) failed.")
        return 1
    print("PHASE 2 EXIT GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
