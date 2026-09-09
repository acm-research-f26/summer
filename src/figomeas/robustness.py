"""Robustness studies (Phase 7).

The brief's plan here was: report everything on the full cohort and again on the
"native-isotropic subset", treating the latter as the trustworthy numbers.
**That subset does not exist** (PLAN.md F2). The 33 patients whose NIfTI headers
claimed 1 mm isotropic voxels are the 33 raw-written files with an unset pixdim;
their DICOM headers show ~0.45 x 0.45 x 5 mm, 2D acquisition, exactly like the
other 267. Every patient in UMD is anisotropic, median 13.3x.

So through-plane sensitivity has to be established by *simulation* on the data we
actually have, plus a stratification by how well each fibroid happens to be
sampled:

1. **Slice decimation.** Drop every 2nd or 3rd slice, re-run the measurement, and
   see how far percent_intramural moves and how often the FIGO call flips. This
   answers "what does slice thickness cost us?" directly, by making it worse and
   measuring the damage.
2. **Slice-count restriction.** Report on fibroids spanning >= 4 and >= 6 slices.
   These take over the role the isotropic subset was supposed to play: they are
   the fibroids whose through-plane geometry is actually resolved.
3. **Spacing-source split.** `_seg` (267) vs `_seq` (33) as a batch check -- the
   two groups came out of different export tooling, so if that tracks anything
   real it should show up here.
4. **Reconstruction sensitivity.** Vary the Phase 2 bridging radius and measure
   FIGO-call churn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .body import percent_intramural, reconstruct
from .features import extract_patient
from .figo import derive


def decimate_slices(mask: np.ndarray, factor: int, offset: int = 0):
    """Keep every ``factor``-th slice, simulating a thicker acquisition.

    Returns the reduced mask and its new spacing multiplier. Dropping slices is
    the honest direction to perturb in: we can always make the sampling worse and
    watch the answer degrade, but we cannot manufacture slices we never acquired.
    """
    if factor < 2:
        return mask.copy(), 1
    keep = np.arange(offset, mask.shape[2], factor)
    return mask[:, :, keep], factor


def measure_under_decimation(mask, spacing, cfg, factor):
    """Per-fibroid percent_intramural and FIGO after dropping slices."""
    thin, mult = decimate_slices(mask, factor)
    sp = (spacing[0], spacing[1], spacing[2] * mult)
    records, _, _ = extract_patient(thin, sp, cfg, "")
    for r in records:
        r["figo"] = derive(r, cfg)
    return records


def _match_by_rank(base, other):
    """Pair fibroids across resolutions by descending volume.

    Connected components are relabelled when slices are removed, and two fibroids
    can even merge, so component ids do not survive decimation. Volume rank is
    the most stable correspondence available; pairs beyond the shorter list are
    dropped rather than guessed.
    """
    b = sorted(base, key=lambda r: -r["volume_mm3"])
    o = sorted(other, key=lambda r: -r["volume_mm3"])
    return list(zip(b, o))


def decimation_study(manifest, cfg, load_mask_fn, repo_root, factors=(2, 3),
                     limit=None, progress=True):
    """Compare full-resolution measurements against decimated ones."""
    rows = []
    man = manifest.head(limit) if limit else manifest
    it = man.itertuples()
    if progress:
        from tqdm import tqdm
        it = tqdm(it, total=len(man), desc="decimation", unit="pt")

    for r in it:
        mask = load_mask_fn(repo_root / r.mask_path).data
        sp = (r.used_sx, r.used_sy, r.used_sz)
        base = measure_under_decimation(mask, sp, cfg, 1)
        if not base:
            continue
        for factor in factors:
            if mask.shape[2] // factor < 2:
                continue
            other = measure_under_decimation(mask, sp, cfg, factor)
            for b, o in _match_by_rank(base, other):
                rows.append({
                    "patient_id": r.patient_id, "factor": factor,
                    "volume_mm3": b["volume_mm3"],
                    "n_slices_full": b["n_slices_spanned"],
                    "p_full": b["percent_intramural"],
                    "p_decimated": o["percent_intramural"],
                    "delta_p": o["percent_intramural"] - b["percent_intramural"],
                    "figo_full": b["figo"], "figo_decimated": o["figo"],
                    "figo_flipped": b["figo"] != o["figo"],
                    "crossed_50": (b["percent_intramural"] >= 50)
                                  != (o["percent_intramural"] >= 50),
                })
    return pd.DataFrame(rows)


def slice_count_strata(df: pd.DataFrame, thresholds=(4, 6)) -> pd.DataFrame:
    """Summaries on progressively better-sampled subsets of fibroids."""
    rows = [{"subset": "all fibroids", "n": len(df),
             "median_p": df.percent_intramural.median(),
             "median_ci_width": df.ci_width.median() if "ci_width" in df else np.nan,
             "pct_borderline": 100 * df.is_borderline.mean()
             if "is_borderline" in df else np.nan}]
    for t in thresholds:
        sub = df[df.n_slices_spanned >= t]
        rows.append({"subset": f">= {t} slices", "n": len(sub),
                     "median_p": sub.percent_intramural.median(),
                     "median_ci_width": sub.ci_width.median() if "ci_width" in sub else np.nan,
                     "pct_borderline": 100 * sub.is_borderline.mean()
                     if "is_borderline" in sub else np.nan})
    return pd.DataFrame(rows)


def spacing_source_split(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """`_seg` vs `_seq` batch check (see PLAN.md F1/F2)."""
    m = df.merge(manifest[["patient_id", "filename_kind"]], on="patient_id", how="left")
    return (m.groupby("filename_kind")
            .agg(n=("percent_intramural", "size"),
                 median_p=("percent_intramural", "median"),
                 median_volume_mm3=("volume_mm3", "median"),
                 pct_submucosal=("submucosal", "mean"))
            .reset_index())


def reconstruction_sensitivity(mask, fibroid, spacing, cfg, scales):
    """percent_intramural as the Phase 2 bridging radius is varied."""
    import copy

    from .config import Config
    out = []
    for sc in scales:
        d = copy.deepcopy(dict(cfg._data))
        d["body"]["adaptive_radius_scale"] = float(sc)
        model = reconstruct(mask, spacing, Config(d), fibroid=fibroid)
        out.append({"radius_scale": float(sc),
                    "percent_intramural": percent_intramural(fibroid, model)})
    return pd.DataFrame(out)


def run(cfg):
    raise NotImplementedError("driven by scripts/run_robustness.py")
