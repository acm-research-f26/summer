"""Hand-built synthetic masks at known FIGO positions.

These exist so the geometry can be validated against answers known by
construction. A reconstruction bug that quietly returns "100% intramural" for
everything is invisible on real data and obvious here.

The phantom is a uterus built from two concentric ellipsoids -- an outer solid
bounded by the serosa, and an inner endometrial cavity -- with a spherical
fibroid placed at a radius chosen to realise each FIGO type. Voxel spacing is
deliberately anisotropic, matching the real cohort (~0.5 x 0.5 x 5 mm), so the
phantoms exercise the same through-plane weakness the real data has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fibroid centre offset along +x (mm from uterine centre) per FIGO type, with the
# fibroid radius in mm. Cavity outer edge sits at x=10 mm, serosa at x=30 mm.
# Cavity outer edge sits at x = 16 mm, serosa at x = 34 mm.
#
# The cavity must be wide enough to actually *contain* a submucosal fibroid. An
# earlier version used a 7 mm cavity half-width with an 8 mm fibroid radius, which
# makes "<50% intramural" geometrically impossible -- the phantom, not the
# measurement, was wrong. Offsets below are chosen so the analytic spherical-cap
# fraction lands where the FIGO definition says it should.
# Offsets are chosen against the *numerically computed* ground truth below, not
# hand-derived spherical-cap algebra -- the cavity boundary is curved, so the
# planar-cap approximation was wrong by ~10 points. Offsets are also kept out of
# the regime where the fibroid is wider than the cavity at that x: there the
# fibroid truncates the cavity rather than indenting it, no morphological closing
# can restore a convex truncation, and the phantom stops representing the FIGO
# type it claims to.
#
# type: (offset_mm, radius_mm)  ->  analytic percent-intramural
FIGO_PHANTOM_GEOMETRY: dict[str, tuple[float, float]] = {
    "0": (0.0, 6.0),    # pedunculated intracavitary, entirely in cavity ->   0.0%
    "1": (10.0, 8.0),   # submucosal <50% intramural                     ->   9.7%
    "2": (20.0, 8.0),   # submucosal >=50%, touches cavity               ->  87.9%
    "3": (22.0, 6.0),   # ~100% intramural, abuts the endometrium        -> 100.0%
                        #   r=6 not 8: an r=8 sphere at x=24 reaches x=32, only 2 mm
                        #   inside a serosa at x=34, so it registered serosa contact
    "4": (25.0, 6.0),   # intramural, touches neither surface            -> 100.0%
    "5": (30.0, 8.0),   # subserosal >=50%, touches serosa               ->  81.2%
    "6": (37.0, 8.0),   # subserosal <50%, bulges through the serosa     ->  20.0%
    "7": (44.0, 8.0),   # subserosal pedunculated, hangs off the surface ->   0.0%
}

UTERUS_SEMI_AXES_MM = (34.0, 29.0, 32.0)
CAVITY_SEMI_AXES_MM = (16.0, 12.0, 20.0)


@dataclass(frozen=True)
class Phantom:
    data: np.ndarray
    spacing: tuple[float, float, float]
    figo_type: str
    fibroid_offset_mm: float
    fibroid_radius_mm: float


def _grid(shape, spacing):
    """Physical coordinates in mm, centred on the array."""
    axes = [(np.arange(n) - (n - 1) / 2.0) * s for n, s in zip(shape, spacing)]
    return np.meshgrid(*axes, indexing="ij")


def _ellipsoid(shape, spacing, semi_axes, centre=(0.0, 0.0, 0.0)):
    X, Y, Z = _grid(shape, spacing)
    a, b, c = semi_axes
    cx, cy, cz = centre
    return ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 + ((Z - cz) / c) ** 2 <= 1.0


def _sphere(shape, spacing, radius_mm, centre):
    X, Y, Z = _grid(shape, spacing)
    cx, cy, cz = centre
    return (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius_mm ** 2


def make_phantom(figo_type,
                 spacing: tuple[float, float, float] = (0.5, 0.5, 5.0),
                 shape: tuple[int, int, int] = (260, 200, 24),
                 labels: dict | None = None) -> Phantom:
    """Build a labelled uterus phantom whose correct FIGO type is known.

    The fibroid is carved out of whatever it overlaps, reproducing the hard
    partition of the real UMD masks (a voxel is wall *or* cavity *or* fibroid,
    never two at once) -- which is exactly the property that makes the naive
    percent-intramural formula degenerate.
    """
    figo_type = str(figo_type)
    if figo_type not in FIGO_PHANTOM_GEOMETRY:
        raise ValueError(f"no phantom geometry for FIGO type {figo_type!r}")

    lab = labels or {"background": 0, "wall": 1, "cavity": 2, "fibroid": 3}
    offset_mm, radius_mm = FIGO_PHANTOM_GEOMETRY[figo_type]

    uterus = _ellipsoid(shape, spacing, UTERUS_SEMI_AXES_MM)
    cavity = _ellipsoid(shape, spacing, CAVITY_SEMI_AXES_MM)
    fibroid = _sphere(shape, spacing, radius_mm, (offset_mm, 0.0, 0.0))

    data = np.zeros(shape, dtype=np.uint8)
    data[uterus] = lab["wall"]
    data[cavity] = lab["cavity"]
    data[fibroid] = lab["fibroid"]          # carve: hard partition, fibroid wins

    return Phantom(data=data, spacing=spacing, figo_type=figo_type,
                   fibroid_offset_mm=offset_mm, fibroid_radius_mm=radius_mm)


def ground_truth_percent_intramural(phantom: Phantom) -> float:
    """Exact percent-intramural, computed from the *undistorted* ellipsoids.

    This is the number the reconstruction is trying to recover. It is measured
    against the uterus and cavity as they exist before the fibroid displaces
    them, which is precisely the information a real mask does not contain -- and
    precisely why the phantoms are worth having.
    """
    shape, spacing = phantom.data.shape, phantom.spacing
    uterus = _ellipsoid(shape, spacing, UTERUS_SEMI_AXES_MM)
    cavity = _ellipsoid(shape, spacing, CAVITY_SEMI_AXES_MM)
    fibroid = _sphere(shape, spacing, phantom.fibroid_radius_mm,
                      (phantom.fibroid_offset_mm, 0.0, 0.0))
    n = int(fibroid.sum())
    return 100.0 * float((fibroid & uterus & ~cavity).sum()) / n if n else float("nan")


def all_phantoms(**kwargs) -> dict[str, Phantom]:
    return {t: make_phantom(t, **kwargs) for t in FIGO_PHANTOM_GEOMETRY}


# Spacings drawn from the real cohort, so the demo exercises the same anisotropy
# the pipeline actually faces (PLAN.md F2: every UMD patient is anisotropic).
DEMO_SPACINGS = [(0.446, 0.446, 6.6), (0.496, 0.496, 6.6), (0.484, 0.484, 4.4),
                 (0.754, 0.754, 5.5), (0.313, 0.313, 5.5)]


def build_demo_cohort(cfg, n_per_type: int = 2, out_dir=None):
    """Write a synthetic cohort of masks with known FIGO types to disk.

    Each phantom is emitted as a real ``.nii.gz`` so ``make demo`` exercises the
    genuine loader, spacing handling and feature extractor rather than a
    shortcut path. Voxel spacings are sampled from the real cohort so the demo
    inherits its anisotropy instead of pretending to be isotropic.
    """
    import nibabel as nib

    from .config import REPO_ROOT

    out_dir = pathlib_path(out_dir or (REPO_ROOT / cfg["data.demo_root"] / "raw"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    idx = 0
    for figo_type in sorted(FIGO_PHANTOM_GEOMETRY):
        for k in range(n_per_type):
            spacing = DEMO_SPACINGS[idx % len(DEMO_SPACINGS)]
            ph = make_phantom(figo_type, spacing=spacing)
            pid = f"DEMO_{idx:03d}_figo{figo_type}"
            path = out_dir / f"{pid}_seg.nii.gz"
            img = nib.Nifti1Image(ph.data, np.diag([*spacing, 1.0]))
            img.header.set_zooms(spacing)
            nib.save(img, path)
            manifest.append({
                "patient_id": pid, "mask_path": str(path),
                "true_figo": figo_type,
                "true_percent_intramural": ground_truth_percent_intramural(ph),
                "used_sx": spacing[0], "used_sy": spacing[1], "used_sz": spacing[2],
            })
            idx += 1
    return manifest


def pathlib_path(p):
    from pathlib import Path
    return Path(p)
