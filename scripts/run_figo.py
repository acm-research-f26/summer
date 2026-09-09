#!/usr/bin/env python3
"""PHASE 4 — deterministic FIGO derivation + the submucosal correctness gate."""

import sys
from collections import Counter

import pandas as pd

from figomeas import load_config
from figomeas.config import REPO_ROOT
from figomeas.figo import (TYPE_MEANINGS, components, derive, is_submucosal,
                           is_subserosal, near_threshold)


def main() -> int:
    cfg = load_config()
    df = pd.read_parquet(REPO_ROOT / "results" / "fibroids.parquet")

    df["figo"] = df.apply(lambda r: derive(r, cfg), axis=1)
    df["is_hybrid"] = df.figo.str.contains(cfg["figo.hybrid_separator"], regex=False)
    df["submucosal"] = df.figo.apply(lambda t: is_submucosal(t, cfg))
    df["subserosal"] = df.figo.apply(lambda t: is_subserosal(t, cfg))
    df["near_50"] = df.apply(lambda r: near_threshold(r, cfg), axis=1)
    df.to_parquet(REPO_ROOT / "results" / "fibroids_figo.parquet", index=False)
    df.to_csv(REPO_ROOT / "results" / "fibroids_figo.csv", index=False)

    print(f"\n=== FIGO distribution ({len(df)} fibroids) ===\n")
    print(f"{'type':>6} {'n':>5} {'%':>6}  {'median p':>9}  meaning")
    counts = Counter()
    for t in df.figo:
        for c in components(t, cfg):
            counts[c] += 1
    for t in sorted(counts):
        sub = df[df.figo.apply(lambda x: t in components(x, cfg))]
        print(f"{t:>6} {counts[t]:5d} {100*counts[t]/len(df):5.1f}%  "
              f"{sub.percent_intramural.median():8.1f}%  {TYPE_MEANINGS.get(t,'')}")
    print(f"\n{'8':>6} {0:5d} {0.0:5.1f}%  {'n/a':>9}  {TYPE_MEANINGS['8']}")

    print(f"\nas written (hybrids kept whole):")
    for t, c in df.figo.value_counts().items():
        print(f"  {t:>6}  {c:4d}  ({100*c/len(df):.1f}%)")

    # The unreliable fibroids (no reconstructable serosa reference) would otherwise
    # pollute the distribution with confident nonsense: UMD_221129_016 is a 918 cm3
    # fibroid measuring 0.0% intramural, which derives as type 7 -- "subserosal
    # pedunculated" for a mass that has replaced the entire uterus. Gates are
    # judged on the reliable subset; the flagged ones are reported, not hidden.
    unrel = df[~df.reliable]
    if len(unrel):
        print(f"\n{len(unrel)} fibroids flagged unreliable and excluded from the gates below:")
        print(f"  median volume {unrel.volume_mm3.median()/1000:.0f} cm3, "
              f"median fibroid mass fraction {unrel.fibroid_mass_frac.median():.2f}")
        print(f"  their derived types (not trustworthy): "
              f"{dict(unrel.figo.value_counts().head(5))}")
    df = df[df.reliable].copy()
    counts = Counter()
    for t in df.figo:
        for c in components(t, cfg):
            counts[c] += 1
    print(f"\n=== reliable subset: {len(df)} fibroids ===")

    n_sub = int(df.submucosal.sum())
    n_ser = int(df.subserosal.sum())
    print(f"\nsubmucosal (0/1/2): {n_sub} fibroids in "
          f"{df[df.submucosal].patient_id.nunique()} patients  ({100*n_sub/len(df):.1f}%)")
    print(f"subserosal (5/6/7): {n_ser}  ({100*n_ser/len(df):.1f}%)")
    print(f"hybrids           : {int(df.is_hybrid.sum())}")
    print(f"within {cfg['borderline.band_pct']:.0f} points of the 50% line: "
          f"{int(df.near_50.sum())} ({100*df.near_50.mean():.1f}%)")

    print(f"\n1-vs-2 split (hysteroscopic resectability): "
          f"type1={counts.get('1',0)}, type2={counts.get('2',0)}")
    print(f"5-vs-6 split: type5={counts.get('5',0)}, type6={counts.get('6',0)}")

    n_types = len(counts)
    gates = [
        ("G1  submucosal count is clinically plausible, NOT near-zero",
         n_sub >= 30,
         f"{n_sub} submucosal fibroids ({100*n_sub/len(df):.1f}%). "
         f"The earlier attempt collapsed to ~9."),
        ("G2  all 8 mask-derivable types are represented (type 8 is not derivable)",
         n_types >= 8, f"{n_types} distinct types present: {sorted(counts)}"),
        ("G3  no single type absorbs the whole cohort",
         max(counts.values()) / len(df) < 0.80,
         f"largest type {max(counts, key=counts.get)} holds "
         f"{100*max(counts.values())/len(df):.1f}%"),
        ("G4  both sides of the 50% boundary are populated in each family",
         counts.get("1", 0) > 0 and counts.get("2", 0) > 0
         and counts.get("5", 0) > 0 and counts.get("6", 0) > 0,
         f"1={counts.get('1',0)} 2={counts.get('2',0)} "
         f"5={counts.get('5',0)} 6={counts.get('6',0)}"),
    ]
    print("\n=== PHASE 4 EXIT GATES ===")
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    failed = [n for n, ok, _ in gates if not ok]
    print()
    if failed:
        print(f"PHASE 4 BLOCKED — {len(failed)} gate(s) failed.")
        return 1
    print("PHASE 4 EXIT GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
