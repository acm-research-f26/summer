"""Per-fibroid geometry, including the headline percent_intramural (Phase 3).

Every fibroid is decomposed into exactly three disjoint fractions of its own
volume, measured against the Phase 2 reference surfaces:

    frac_intracavitary  = |F & cavity_ref|          / |F|    bulging into the cavity
    percent_intramural  = |F & body & ~cavity_ref|  / |F|    buried in myometrium
    frac_extraserosal   = |F & ~body|               / |F|    protruding past the serosa

Because ``cavity_ref`` is constrained to lie inside ``body``, these three
partition the fibroid exactly and must sum to 1. That identity is asserted as an
invariant rather than assumed -- it is the cheapest available check that the
reference surfaces are mutually consistent, and it is what would have caught the
degenerate formula (F5) immediately, since that one puts every fibroid at
percent_intramural = 1 with the other two fractions still non-zero.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from .body import compartment_split, reconstruct


def _min_distance_mm(fibroid, target, spacing) -> float:
    """Shortest physical distance from the fibroid to ``target``, in millimetres.

    Contact must be measured in mm, not by voxel dilation. One voxel of dilation
    is 0.45 mm in-plane but 5 mm through-plane, so a one-voxel rule silently
    declares a fibroid "touching" a surface a whole slice away -- it put the
    type-3 phantom in contact with a serosa 2 mm distant and derived type 5.
    """
    if not target.any() or not fibroid.any():
        return float("inf")
    dist = ndimage.distance_transform_edt(~target, sampling=spacing)
    return float(dist[fibroid].min())


def _shape_features(fibroid, spacing):
    """PCA axes, max diameter, solidity and sphericity, all in millimetres.

    Everything is computed on the fibroid's own bounding box, and the hull on its
    surface voxels only. Feeding the full 672x672x20 FOV to marching cubes, and
    all 1.4M interior voxels to ConvexHull, made this the slowest step in the
    pipeline by a wide margin -- and the interior points cannot affect a convex
    hull anyway.
    """
    out = {"max_diameter_mm": np.nan, "elongation": np.nan, "flatness": np.nan,
           "solidity": np.nan, "sphericity": np.nan, "surface_area_mm2": np.nan}
    idx = np.nonzero(fibroid)
    if idx[0].size < 4:
        return out
    sl = tuple(slice(a.min(), a.max() + 1) for a in idx)
    crop = fibroid[sl]

    pts = np.array(np.nonzero(crop), dtype=float).T * np.asarray(spacing)
    cov = np.cov((pts - pts.mean(0)).T)
    ev = np.clip(np.sort(np.linalg.eigvalsh(cov))[::-1], 1e-12, None)
    out["elongation"] = float(np.sqrt(ev[1] / ev[0]))
    out["flatness"] = float(np.sqrt(ev[2] / ev[1]))

    volume_mm3 = float(crop.sum()) * float(np.prod(spacing))
    shell = crop & ~ndimage.binary_erosion(crop)
    spts = np.array(np.nonzero(shell), dtype=float).T * np.asarray(spacing)

    # Max diameter comes straight from the surface points, never from the hull.
    # A fibroid spanning 1-2 slices is coplanar in physical space, so the 3D hull
    # raises and both features come back NaN -- and few-slice fibroids are exactly
    # the anisotropy-limited ones we most need shape features for.
    if spts.shape[0] >= 2:
        pts_for_diam = spts
        if pts_for_diam.shape[0] > 4000:
            rng = np.random.default_rng(0)
            pts_for_diam = pts_for_diam[rng.choice(pts_for_diam.shape[0], 4000, replace=False)]
        out["max_diameter_mm"] = float(pdist(pts_for_diam).max())

    if spts.shape[0] >= 4:
        try:
            hull = ConvexHull(spts)
            out["solidity"] = float(volume_mm3 / hull.volume) if hull.volume > 0 else np.nan
        except Exception:
            # Degenerate (coplanar) in 3D: fall back to stacked 2D hull areas.
            try:
                from skimage.morphology import convex_hull_image
                area_vox = sum(convex_hull_image(crop[:, :, k]).sum()
                               for k in range(crop.shape[2]) if crop[:, :, k].any())
                hull_vol = float(area_vox) * float(np.prod(spacing))
                out["solidity"] = float(volume_mm3 / hull_vol) if hull_vol > 0 else np.nan
            except Exception:
                pass

    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        verts, faces, _, _ = marching_cubes(np.pad(crop, 1).astype(np.uint8),
                                            level=0.5, spacing=spacing)
        area = float(mesh_surface_area(verts, faces))
        out["surface_area_mm2"] = area
        if area > 0:
            out["sphericity"] = float((np.pi ** (1 / 3)) * ((6 * volume_mm3) ** (2 / 3)) / area)
    except Exception:
        pass
    return out


def extract_patient(mask, spacing, cfg, patient_id=""):
    """One record per connected fibroid component that clears the size filter."""
    spacing = tuple(float(s) for s in spacing)
    voxel_mm3 = float(np.prod(spacing))
    st = ndimage.generate_binary_structure(3, int(cfg["fibroid.connectivity"]))
    labelled, n = ndimage.label(mask == cfg["labels.fibroid"], structure=st)

    cavity_lbl = mask == cfg["labels.cavity"]
    records, dropped = [], 0

    for i in range(1, n + 1):
        f = labelled == i
        n_vox = int(f.sum())
        volume_mm3 = n_vox * voxel_mm3
        if volume_mm3 < float(cfg["fibroid.min_volume_mm3"]):
            dropped += 1
            continue

        model = reconstruct(mask, spacing, cfg, fibroid=f)
        pct_ser, pct_cav, p = compartment_split(
            f, cavity_lbl, mask == cfg["labels.wall"], model.body, spacing)
        frac_cav, frac_ser = pct_cav / 100.0, pct_ser / 100.0

        # Distance to the endometrium, and to the world outside the serosa.
        dist_cav = _min_distance_mm(f, cavity_lbl | model.cavity_ref, spacing)
        dist_ser = _min_distance_mm(f, ~model.body, spacing)
        # Contact must be resolution-aware. Two tangent surfaces are separated by
        # a discretisation gap of a few voxels, so a fixed 1.0 mm rule made the
        # type-3/type-4 distinction a coin-flip on voxel size: the same phantom
        # measured 0.97 mm from the endometrium at 0.484 mm in-plane (contact)
        # and 1.49 mm at 0.496 mm (no contact). The floor below is the annotation's
        # own boundary uncertainty, which is what actually limits the call.
        touch_mm = max(float(cfg["fibroid.contact_distance_mm"]),
                       float(cfg["fibroid.contact_distance_voxels"])
                       * 0.5 * (spacing[0] + spacing[1]))
        eps_frac = float(cfg["fibroid.min_contact_fraction"])

        rec = {
            "patient_id": patient_id, "component": i,
            "volume_voxels": n_vox, "volume_mm3": volume_mm3,
            "n_slices_spanned": int(np.count_nonzero(f.any(axis=(0, 1)))),
            "percent_intramural": p,
            "frac_intracavitary": 100.0 * frac_cav,
            "frac_extraserosal": 100.0 * frac_ser,
            "dist_to_cavity_mm": dist_cav,
            "dist_to_serosa_mm": dist_ser,
            "touch_cavity": bool(dist_cav <= touch_mm or frac_cav > eps_frac),
            "touch_serosa": bool(dist_ser <= touch_mm or frac_ser > eps_frac),
            "reliable": bool(model.reliable),
            "radius_clamped": bool(model.radius_clamped),
            "fibroid_mass_frac": float(model.fibroid_mass_frac),
            "sx": spacing[0], "sy": spacing[1], "sz": spacing[2],
        }
        rec.update(_shape_features(f, spacing))
        records.append(rec)

    return records, dropped, n


def partition_residual(rec) -> float:
    """|1 - (intramural + intracavitary + extraserosal)|, in percentage points."""
    return abs(100.0 - (rec["percent_intramural"] + rec["frac_intracavitary"]
                        + rec["frac_extraserosal"]))
