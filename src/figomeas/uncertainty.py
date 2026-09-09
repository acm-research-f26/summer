"""Confidence intervals on percent_intramural via mask perturbation (Phase 5).

What the error budget actually looks like
-----------------------------------------
Perturbations are specified in millimetres and applied **anisotropically**,
because the two axes carry completely different amounts of doubt:

* **In-plane** the annotator drew a boundary on a 0.45 mm grid. Being off by a
  voxel or two is a sub-millimetre question.
* **Through-plane** the slices are 4.4-6.6 mm apart with nothing sampled in
  between. The true fibroid surface between two slices is unknown to roughly
  half a slice, i.e. +-2 to +-3 mm -- five times the in-plane uncertainty. Many
  fibroids span only 2-3 slices, so this term dominates the whole budget.

A perturbation specified in *voxels* would get this exactly backwards, applying
the same 1-voxel nudge to both axes and understating the dominant term by an
order of magnitude.

But an mm-specified *morphological* perturbation cannot express the through-plane
term either, and this is worth being explicit about. On a 5 mm grid, dilating by
1, 2.5 or 4 mm through-plane changes nothing at all; dilating by 5 mm adds a
whole slice at each end. It is a step function, and the +-2.5 mm boundary shift
that actually dominates the error budget falls entirely inside the dead zone.

So the two axes are perturbed by different mechanisms, each matched to its own
granularity:

* **in-plane**: a true ellipsoidal dilation/erosion in mm, sub-voxel resolvable.
* **through-plane**: discrete jitter of the first and last slice the fibroid
  occupies. "The boundary lies within half a slice of where it was drawn" is,
  on this grid, exactly the statement that the end slice may or may not belong
  to the fibroid -- so that is what gets sampled.

Three sources are propagated: the in-plane boundary, the through-plane slice
extent, and the reconstruction's own free parameter (the Phase 2 bridging radius).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .body import percent_intramural, reconstruct


def ellipsoid_offset(mask, spacing, inplane_mm, through_mm):
    """Grow (or shrink) ``mask`` by an ellipsoid with distinct in-plane/through-plane radii.

    Scaling each axis of the EDT sampling by its own semi-axis turns the unit
    ball into the desired ellipsoid, so this stays a single O(n) transform rather
    than an explicit structuring element.
    """
    if inplane_mm == 0.0 and through_mm == 0.0:
        return mask.copy()
    grow = inplane_mm > 0 or through_mm > 0
    a = max(abs(inplane_mm), 1e-6)
    c = max(abs(through_mm), 1e-6)
    sampling = (spacing[0] / a, spacing[1] / a, spacing[2] / c)

    # Work in the fibroid's own neighbourhood. Running the distance transform
    # over the whole 672x672x20 field of view for a fibroid occupying ~1% of it
    # made this step dominate the sweep: 140 s per patient, an 11-hour cohort.
    idx = np.nonzero(mask)
    if idx[0].size == 0:
        return mask.copy()
    pad = [int(np.ceil(r / sp)) + 2 for r, sp in zip((a, a, c), spacing)]
    roi = tuple(slice(max(int(i.min()) - q, 0), min(int(i.max()) + q + 1, n))
                for i, q, n in zip(idx, pad, mask.shape))
    sub = mask[roi]

    padded = np.pad(sub, [(q, q) for q in pad], mode="constant", constant_values=False)
    if grow:
        inner = ndimage.distance_transform_edt(~padded, sampling=sampling) <= 1.0
    else:
        inner = ndimage.distance_transform_edt(padded, sampling=sampling) > 1.0
    sl = tuple(slice(q, q + n) for q, n in zip(pad, sub.shape))

    out = np.zeros_like(mask)
    out[roi] = inner[sl]
    return out


def jitter_end_slices(fibroid, rng):
    """Randomly drop or extend the first/last slice the fibroid occupies.

    This is the through-plane error term made discrete. Each end is decided
    independently, since the boundary at the top of the fibroid is no more known
    than the boundary at the bottom.
    """
    zs = np.nonzero(fibroid.any(axis=(0, 1)))[0]
    if zs.size == 0:
        return fibroid.copy()
    out = fibroid.copy()
    for end, nbr in ((int(zs.min()), -1), (int(zs.max()), +1)):
        draw = rng.uniform(-0.5, 0.5)
        if draw < -0.25:                       # boundary sat inside this slice
            out[:, :, end] = False
        elif draw > 0.25:                      # boundary sat beyond it
            tgt = end + nbr
            if 0 <= tgt < out.shape[2]:
                out[:, :, tgt] |= fibroid[:, :, end]
    return out


def perturbation_draws(spacing, cfg, rng):
    """Sample (in-plane mm, through-plane mm, radius-scale) triples.

    Stratified rather than fully random: reconstruction is the expensive step, so
    each drawn bridging radius is reused across several boundary perturbations.
    That buys the same number of samples for a quarter of the reconstructions.
    """
    n_r = int(cfg["uncertainty.n_radius_draws"])
    n_m = int(cfg["uncertainty.n_mask_draws"])
    lo, hi = cfg["uncertainty.radius_scale_range"]
    inplane = float(cfg["uncertainty.inplane_mm"])

    scales = np.linspace(lo, hi, n_r) if n_r > 1 else np.array([(lo + hi) / 2])
    draws = []
    for sc in scales:
        for _ in range(n_m):
            draws.append((float(rng.uniform(-inplane, inplane)), float(sc)))
    return draws


def _cfg_with_scale(cfg, scale):
    """Config clone with one field overridden. Only the `body` block is copied --
    a deepcopy of the whole config ran once per draw, sixteen times per fibroid."""
    from .config import Config
    d = dict(cfg._data)
    d["body"] = dict(d["body"])
    d["body"]["adaptive_radius_scale"] = float(scale)
    return Config(d)


def resample_fibroid(mask, fibroid, spacing, cfg, rng):
    """Percent-intramural under each perturbation. Returns the sample vector.

    The scaffold is rebuilt for each bridging radius but held fixed across the
    boundary perturbations that share it. Dilating the fibroid does slightly
    enlarge the crater it leaves in the scaffold; that is a second-order effect
    next to the direct change in |F|, and treating it as fixed is what makes the
    sweep affordable. Stated here rather than buried.
    """
    draws = perturbation_draws(spacing, cfg, rng)
    samples, by_scale = [], {}
    for inplane, scale in draws:
        if scale not in by_scale:
            by_scale[scale] = reconstruct(mask, spacing, _cfg_with_scale(cfg, scale),
                                          fibroid=fibroid)
        model = by_scale[scale]
        f = ellipsoid_offset(fibroid, spacing, inplane, 0.0)
        f = jitter_end_slices(f, rng)
        if not f.any():
            continue
        samples.append(percent_intramural(f, model))
    return np.asarray(samples, dtype=float)


def confidence_interval(samples, cfg):
    """Median and percentile interval, plus whether the band straddles the 50% line."""
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return {"p_median": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "ci_width": np.nan, "ci_straddles_50": False, "n_samples": 0}
    lo = float(np.percentile(s, float(cfg["uncertainty.ci_low_pct"])))
    hi = float(np.percentile(s, float(cfg["uncertainty.ci_high_pct"])))
    thr = float(cfg["figo.intramural_threshold_pct"])
    return {"p_median": float(np.median(s)), "ci_low": lo, "ci_high": hi,
            "ci_width": hi - lo, "ci_straddles_50": bool(lo < thr <= hi),
            "n_samples": int(s.size)}
