"""Per-patient inventory + spacing truth table (Phase 1).

The manifest is the gate everything downstream depends on: if a patient's spacing
is wrong here, every volume, diameter and percent-intramural value derived from it
is wrong by the same factor, and nothing later in the pipeline will notice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import REPO_ROOT, Config
from .io import dicom_spacing, find_masks, load_mask, resolve_spacing

MANIFEST_COLUMNS = [
    "patient_id", "mask_path", "container", "filename_kind",
    "shape_x", "shape_y", "shape_z",
    "nifti_sx", "nifti_sy", "nifti_sz",
    "dicom_sx", "dicom_sy", "dicom_sz",
    "used_sx", "used_sy", "used_sz",
    "spacing_source", "dicom_slice_source", "dicom_n_slices",
    "dicom_acq_type", "dicom_series_desc",
    "n_dcm", "labels_present", "has_cavity", "has_fibroid",
    "n_fibroid_components", "fibroid_voxels", "voxel_volume_mm3",
    "anisotropy_ratio",
]


def _row(mask_path: Path, cfg: Config) -> dict:
    patient_dir = mask_path.parent
    m = load_mask(mask_path)
    geom = dicom_spacing(patient_dir, match_shape=m.shape)
    spacing, source = resolve_spacing(m.nifti_spacing, geom, cfg)

    labels = sorted(m.labels())
    fib = m.data == cfg["labels.fibroid"]
    structure = ndimage.generate_binary_structure(3, cfg["fibroid.connectivity"])
    _, n_comp = ndimage.label(fib, structure=structure)

    d_sx, d_sy, d_sz = geom.spacing if geom.spacing else (np.nan,) * 3
    u_sx, u_sy, u_sz = spacing if spacing else (np.nan,) * 3

    return {
        "patient_id": patient_dir.name,
        "mask_path": str(mask_path.relative_to(REPO_ROOT)),
        "container": m.container,
        "filename_kind": m.filename_kind,
        "shape_x": m.shape[0], "shape_y": m.shape[1], "shape_z": m.shape[2],
        "nifti_sx": m.nifti_spacing[0], "nifti_sy": m.nifti_spacing[1],
        "nifti_sz": m.nifti_spacing[2],
        "dicom_sx": d_sx, "dicom_sy": d_sy, "dicom_sz": d_sz,
        "used_sx": u_sx, "used_sy": u_sy, "used_sz": u_sz,
        "spacing_source": source,
        "dicom_slice_source": geom.slice_source,
        "dicom_n_slices": geom.n_slices,
        "dicom_acq_type": geom.acquisition_type,
        "dicom_series_desc": geom.series_description,
        "n_dcm": len(list(patient_dir.glob("*.dcm"))),
        "labels_present": ",".join(str(v) for v in labels),
        "has_cavity": cfg["labels.cavity"] in labels,
        "has_fibroid": cfg["labels.fibroid"] in labels,
        "n_fibroid_components": int(n_comp),
        "fibroid_voxels": int(fib.sum()),
        "voxel_volume_mm3": float(u_sx * u_sy * u_sz) if spacing else np.nan,
        "anisotropy_ratio": float(u_sz / u_sx) if spacing else np.nan,
    }


def build(cfg: Config, progress: bool = True) -> pd.DataFrame:
    """Scan the dataset root and emit ``results/manifest.csv``."""
    root = REPO_ROOT / cfg["data.root"]
    masks = find_masks(root, cfg["data.mask_glob"])
    if not masks:
        raise FileNotFoundError(f"no masks matched {cfg['data.mask_glob']!r} under {root}")

    rows = []
    iterator = masks
    if progress:
        from tqdm import tqdm
        iterator = tqdm(masks, desc="manifest", unit="pt")
    for p in iterator:
        rows.append(_row(p, cfg))

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out = REPO_ROOT / "results" / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


# ------------------------------------------------------------------- exit gate


def check_gates(df: pd.DataFrame, cfg: Config) -> list[tuple[str, bool, str]]:
    """Phase 1 exit gates from PLAN.md. Returns (name, passed, detail) per gate."""
    valid_labels = {cfg[f"labels.{n}"] for n in
                    ("background", "wall", "cavity", "fibroid", "nabothian")}

    seq = df[df.filename_kind == "seq"]
    seg = df[df.filename_kind == "seg"]
    bad_labels = [
        r.patient_id for r in df.itertuples()
        if not {int(v) for v in r.labels_present.split(",")} <= valid_labels
    ]
    seq_not_dicom = seq[seq.spacing_source != "dicom"]

    # Sanity band comes from config, not from a hand-picked sample. An earlier
    # hardcoded 4.0-6.0 band flagged UMD_221129_086 (6.05 mm) as a failure when
    # 6.05 mm is a spacing that also occurs, legitimately, in the _seg cohort.
    lo, hi = cfg["spacing.min_slice_mm"], cfg["spacing.max_slice_mm"]
    seq_bad_sz = seq[~seq.used_sz.between(lo, hi)]

    unit_iso = df[(df.used_sx == 1.0) & (df.used_sy == 1.0) & (df.used_sz == 1.0)]
    unresolved = df[df.spacing_source == "none"]
    aniso_median = float(df.anisotropy_ratio.median())
    sub2 = df[df.anisotropy_ratio < 2.0]
    n_raw = int((df.container == "raw").sum())
    shape_mismatch = df[df.dicom_n_slices != df.shape_z]

    # Independent validation: on patients whose NIfTI pixdim is trustworthy, the
    # DICOM-derived spacing must agree with it. If it does not, the DICOM path is
    # broken and the _seq corrections it produces cannot be believed either.
    dz = (seg.dicom_sz - seg.nifti_sz).abs()
    dxy = (seg.dicom_sx - seg.nifti_sx).abs()
    max_disagree = float(max(dz.max(), dxy.max()))

    return [
        ("G1  300/300 patients load, zero silent drops",
         len(df) == 300 and unresolved.empty,
         f"loaded {len(df)}, unresolved spacing {len(unresolved)}"),
        ("G2  labels subset of {0,1,2,3,4}",
         not bad_labels,
         f"violations: {bad_labels[:5] or 'none'}"),
        (f"G3  all 33 _seq patients use DICOM spacing, sz within config band {lo}-{hi} mm",
         len(seq) == 33 and seq_not_dicom.empty and seq_bad_sz.empty,
         f"n_seq={len(seq)}, non-dicom={len(seq_not_dicom)}, sz out of band={len(seq_bad_sz)}"),
        ("G4  zero patients left at (1,1,1)",
         unit_iso.empty,
         f"{len(unit_iso)} patients"),
        ("G5  anisotropy median ~10-13x, no sub-2x survivors",
         8.0 <= aniso_median <= 14.0 and sub2.empty,
         f"median={aniso_median:.2f}x, sub-2x={len(sub2)}"),
        ("G6  container sniffed, not inferred: 34 raw files found (33 _seq + 1 _seg)",
         n_raw == 34,
         f"raw={n_raw} (gzip={len(df)-n_raw}); raw _seg = "
         f"{list(df[(df.container=='raw') & (df.filename_kind=='seg')].patient_id)}"),
        ("G7  DICOM series matches mask depth for every patient",
         shape_mismatch.empty,
         f"{len(shape_mismatch)} mismatches"),
        ("G8  DICOM spacing corroborated by NIfTI on the 266 trustworthy _seg patients",
         max_disagree < 1e-3,
         f"max |dicom - nifti| = {max_disagree:.2e} mm across {len(seg)} patients"),
    ]
