"""PHASE 3 — masks -> per-fibroid feature table with percent_intramural."""

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
from tqdm import tqdm

from figomeas import load_config
from figomeas.config import REPO_ROOT
from figomeas.features import extract_patient, partition_residual
from figomeas.io import load_mask


def _one_patient(job):
    """Worker entry point. Config is reloaded per process; it is small and immutable."""
    patient_id, mask_path, spacing = job
    cfg = load_config()
    mask = load_mask(mask_path).data
    return extract_patient(mask, spacing, cfg, patient_id)


def main() -> int:
    cfg = load_config()
    man = pd.read_csv(REPO_ROOT / "results" / "manifest.csv")

    rows, dropped_total, raw_total = [], 0, 0
    jobs = [(r.patient_id, str(REPO_ROOT / r.mask_path),
             (r.used_sx, r.used_sy, r.used_sz)) for r in man.itertuples()]
    n_workers = max(1, min(int(cfg.get("compute.n_workers", 0)) or (os.cpu_count() or 2) - 2,
                           len(jobs)))
    print(f"extracting with {n_workers} worker processes")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for recs, dropped, n_raw in tqdm(pool.map(_one_patient, jobs, chunksize=2),
                                         total=len(jobs), desc="extract", unit="pt"):
            rows.extend(recs)
            dropped_total += dropped
            raw_total += n_raw

    df = pd.DataFrame(rows)
    df["partition_residual"] = df.apply(partition_residual, axis=1)
    out = REPO_ROOT / "results" / "fibroids.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)

    print(f"\n=== {len(df)} fibroids from {df.patient_id.nunique()} patients "
          f"-> results/fibroids.parquet ===")
    print(f"raw components {raw_total}; dropped below "
          f"{cfg['fibroid.min_volume_mm3']} mm3: {dropped_total}\n")

    print("percent_intramural:")
    print(df.percent_intramural.describe().to_string(), "\n")
    print("volume_mm3:")
    print(df.volume_mm3.describe().to_string(), "\n")
    print(f"slices spanned: median {df.n_slices_spanned.median():.0f}, "
          f"{100*(df.n_slices_spanned<=3).mean():.1f}% span <=3 slices\n")
    print(f"contacts: touch_cavity {int(df.touch_cavity.sum())}, "
          f"touch_serosa {int(df.touch_serosa.sum())}, "
          f"both {int((df.touch_cavity & df.touch_serosa).sum())}, "
          f"neither {int((~df.touch_cavity & ~df.touch_serosa).sum())}")
    print(f"reliable: {int(df.reliable.sum())}/{len(df)} "
          f"({100*df.reliable.mean():.1f}%)\n")

    hist = np.histogram(df.percent_intramural, bins=10, range=(0, 100))[0]
    print("percent_intramural histogram (0-100 in 10 bins):")
    for i, c in enumerate(hist):
        print(f"  {i*10:3d}-{i*10+10:3d}%  {'#'*int(60*c/max(hist.max(),1)):60s} {c}")
    print()

    gates = [
        ("G1  fibroid count plausible (dataset documents ~1,077)",
         900 <= len(df) <= 1400, f"{len(df)} fibroids"),
        ("G2  percent_intramural in [0,100] with no NaNs",
         bool(df.percent_intramural.between(0, 100).all()
              and not df.percent_intramural.isna().any()),
         f"range {df.percent_intramural.min():.1f}-{df.percent_intramural.max():.1f}, "
         f"{int(df.percent_intramural.isna().sum())} NaN"),
        ("G3  the three volume fractions partition the fibroid exactly",
         bool(df.partition_residual.max() < 0.01),
         f"max residual {df.partition_residual.max():.2e} points"),
        ("G4  percent_intramural is not degenerate",
         bool(df.percent_intramural.std() > 10
              and (df.percent_intramural > 99.5).mean() < 0.9),
         f"sd {df.percent_intramural.std():.1f}, "
         f"{100*(df.percent_intramural>99.5).mean():.1f}% pinned at 100"),
        ("G5  volumes are clinically sane (median 0.1-100 cm3, no 100x spacing error)",
         bool(100 <= df.volume_mm3.median() <= 100000),
         f"median {df.volume_mm3.median()/1000:.2f} cm3, "
         f"max {df.volume_mm3.max()/1000:.1f} cm3"),
    ]
    print("=== PHASE 3 EXIT GATES ===")
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    failed = [n for n, ok, _ in gates if not ok]
    print()
    if failed:
        print(f"PHASE 3 BLOCKED — {len(failed)} gate(s) failed.")
        return 1
    print("PHASE 3 EXIT GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
