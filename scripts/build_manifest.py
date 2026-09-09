#!/usr/bin/env python3
"""PHASE 1 — build the per-patient inventory and spacing truth table."""

import sys

import pandas as pd

from figomeas import load_config
from figomeas.manifest import build, check_gates


def main() -> int:
    cfg = load_config()
    df = build(cfg)

    print(f"\n=== manifest: {len(df)} patients -> results/manifest.csv ===\n")

    print("container x filename_kind:")
    print(pd.crosstab(df.container, df.filename_kind).to_string(), "\n")

    print("spacing source chosen:")
    print(df.spacing_source.value_counts().to_string(), "\n")

    print("how DICOM slice spacing was measured:")
    print(df.dicom_slice_source.value_counts().to_string(), "\n")

    print("the 33 _seq patients — NIfTI claim vs DICOM truth:")
    seq = df[df.filename_kind == "seq"]
    print(f"  NIfTI pixdim (all): {sorted({(r.nifti_sx, r.nifti_sy, r.nifti_sz) for r in seq.itertuples()})}")
    print(f"  DICOM  sx: {seq.dicom_sx.min():.3f}-{seq.dicom_sx.max():.3f} mm   "
          f"sz: {seq.dicom_sz.min():.2f}-{seq.dicom_sz.max():.2f} mm")
    print(f"  volume error avoided: {(seq.dicom_sx*seq.dicom_sy*seq.dicom_sz).mean():.3f} mm^3/voxel "
          f"vs 1.000 claimed -> {1/(seq.dicom_sx*seq.dicom_sy*seq.dicom_sz).mean():.1f}x\n")

    print("cohort anisotropy (used spacing):")
    print(df.anisotropy_ratio.describe().to_string(), "\n")

    print(f"fibroid components: {int(df.n_fibroid_components.sum())} total across "
          f"{int((df.n_fibroid_components > 0).sum())} patients "
          f"(median {df.n_fibroid_components.median():.0f}/patient)\n")

    print("=== PHASE 1 EXIT GATES ===")
    gates = check_gates(df, cfg)
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    failed = [n for n, ok, _ in gates if not ok]
    print()
    if failed:
        print(f"PHASE 1 BLOCKED — {len(failed)} gate(s) failed.")
        return 1
    print("PHASE 1 EXIT GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
