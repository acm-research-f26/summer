"""Uterine reference-surface reconstruction (Phase 2).

Why this module is not simply "fill the union into a solid"
-----------------------------------------------------------
The brief specifies

    percent_intramural = 100 * |F & body & ~C| / |F|,   body = fill(close(W | C | F))

which is **degenerate**: it returns 100% for every fibroid under every
reconstruction method, so no FIGO type but 3/4 can ever be produced. Two facts
combine to cause it. First, the UMD masks are a hard partition -- a voxel is wall
*or* cavity *or* fibroid, never two -- so wherever the fibroid sits the cavity is
simply not labelled, and ``F & ~C == F`` identically. Second, ``body`` is built
from a union that *includes* ``F``, so ``F & body == F`` too. The expression
collapses to ``100 * |F| / |F|``. Verified on real patients before this module
was written.

This also re-diagnoses the earlier failed attempt. A convex hull was blamed for
collapsing the submucosal count, but with this formula the count collapses under
morphological closing just as completely: every fibroid reads as 100% intramural.
The hull made it worse; the formula made it inevitable.

The measurement that actually means something
---------------------------------------------
A fibroid displaces the very surfaces it should be measured against: a submucosal
one dents the cavity, a subserosal one bulges the serosa outward. So we
reconstruct each surface as it would be **without that distortion**, and measure
the fibroid against those references.

``body_ref``   = fill_holes(close(W | C))  -- fibroids deliberately excluded.
                 An intramural fibroid is then a fully enclosed hole in the wall,
                 which fill_holes restores, so it counts as inside. A subserosal
                 fibroid is not enclosed by wall, so the part protruding past the
                 myometrium stays outside. That asymmetry is the whole measurement.

``cavity_ref`` = fill_holes(close(C)) -- closing bridges the concavity a
                 submucosal fibroid presses into the endometrium, recovering the
                 cavity outline the fibroid is bulging into.

``percent_intramural = 100 * |F & body_ref & ~cavity_ref| / |F|``

All morphology is done in millimetres via the Euclidean distance transform with
``sampling=spacing``. At ~0.45 x 0.45 x 5 mm a structuring element specified in
*voxels* is ~11x larger through-plane than in-plane; specifying it in mm makes a
"5 mm closing" mean the same physical thing on every patient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------ mm morphology


def dilate_mm(mask: np.ndarray, spacing, radius_mm: float) -> np.ndarray:
    """Dilate by a true ball of ``radius_mm``, honouring anisotropic spacing."""
    if radius_mm <= 0:
        return mask.copy()
    return ndimage.distance_transform_edt(~mask, sampling=spacing) <= radius_mm


def erode_mm(mask: np.ndarray, spacing, radius_mm: float) -> np.ndarray:
    """Erode by a true ball of ``radius_mm``. Pads so the array edge is not a wall."""
    if radius_mm <= 0:
        return mask.copy()
    pad = [int(np.ceil(radius_mm / s)) + 1 for s in spacing]
    padded = np.pad(mask, [(p, p) for p in pad], mode="constant", constant_values=False)
    inner = ndimage.distance_transform_edt(padded, sampling=spacing) > radius_mm
    sl = tuple(slice(p, p + n) for p, n in zip(pad, mask.shape))
    return inner[sl]


def close_mm(mask: np.ndarray, spacing, radius_mm: float) -> np.ndarray:
    """Morphological closing with a ball of ``radius_mm``: dilate, then erode.

    Both steps run on a padded array. Dilating in place would clip the ball
    against the array face, and the following erosion then reads that clipped
    face as background and eats the voxels away -- so closing would not be
    extensive and would silently *delete* annotated tissue near the first and
    last slice. On UMD_221129_048 that cost 25% of the labelled cavity, all of it
    on one edge slice, which is exactly where a percent-intramural measurement
    can least afford to lose the endometrium.
    """
    if radius_mm <= 0:
        return mask.copy()
    pad = [int(np.ceil(radius_mm / s)) + 1 for s in spacing]
    padded = np.pad(mask, [(q, q) for q in pad], mode="constant", constant_values=False)
    out = erode_mm(dilate_mm(padded, spacing, radius_mm), spacing, radius_mm)
    sl = tuple(slice(q, q + n) for q, n in zip(pad, mask.shape))
    return out[sl]


def fill_holes(mask: np.ndarray, slicewise: bool = True) -> np.ndarray:
    """Fill enclosed voids in 3D, then per-slice in 2D.

    The 2D pass matters at this anisotropy: a fibroid can be fully enclosed by
    myometrium within its slice while leaking through the top or bottom of a
    2-3 slice stack, which defeats the 3D pass alone.
    """
    filled = ndimage.binary_fill_holes(mask)
    if slicewise:
        for k in range(filled.shape[2]):
            filled[:, :, k] = ndimage.binary_fill_holes(filled[:, :, k])
    return filled


def bounding_box(mask: np.ndarray, spacing, margin_mm: float):
    """Slices bounding ``mask`` with a physical margin.

    The uterus occupies ~1% of a 672x672 pelvic FOV, so every morphological
    operation below would otherwise spend 99% of its time on empty air.
    """
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        return tuple(slice(0, n) for n in mask.shape)
    out = []
    for ax in range(3):
        pad = int(np.ceil(margin_mm / spacing[ax])) + 2
        lo = max(int(idx[ax].min()) - pad, 0)
        hi = min(int(idx[ax].max()) + pad + 1, mask.shape[ax])
        out.append(slice(lo, hi))
    return tuple(out)


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask.copy()
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


# ------------------------------------------------------------------ reconstruction


@dataclass(frozen=True)
class BodyModel:
    body: np.ndarray            # serosa-bounded solid, fibroid distortion excluded
    cavity_ref: np.ndarray      # endometrial envelope, fibroid indentation bridged
    myometrium: np.ndarray      # body & ~cavity_ref -- the intramural compartment
    spacing: tuple[float, float, float]
    method: str
    radius_clamped: bool = False    # fibroid too large for its crater to be bridged
    fibroid_mass_frac: float = 0.0  # fibroid volume / (fibroid + wall + cavity)

    @property
    def reliable(self) -> bool:
        """False when the fibroid is too large for a serosa reference to exist.

        On UMD_221129_016 the wall is 9.4k voxels and the fibroid is 1.43M -- 99.3%
        of the uterine mass. There is no myometrial envelope left to reconstruct,
        and any percent-intramural reported for it would be a confident fiction.
        Such fibroids are measured but flagged, never silently scored.
        """
        return not (self.radius_clamped and self.fibroid_mass_frac > 0.5)

    @property
    def n_components(self) -> int:
        return int(ndimage.label(self.body)[1])


def _hull_ablation(union: np.ndarray) -> np.ndarray:
    """Per-slice 2D convex hull -- the ablation comparator, never the method."""
    from skimage.morphology import convex_hull_image
    out = np.zeros_like(union)
    for k in range(union.shape[2]):
        if union[:, :, k].any():
            out[:, :, k] = convex_hull_image(union[:, :, k])
    return out


def equivalent_radius_mm(fibroid: np.ndarray, spacing) -> float:
    """Radius of the sphere with the same volume as this fibroid."""
    vol = float(fibroid.sum()) * float(np.prod(spacing))
    return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)) if vol > 0 else 0.0


def bridging_radius_mm(fibroid: np.ndarray, spacing, cfg) -> float:
    """Closing radius needed to bridge the crater this fibroid leaves in the wall.

    A partially buried fibroid removes a bite of myometrium whose mouth is as wide
    as the fibroid itself, so a *fixed* radius cannot work: a 5 mm closing leaves a
    14 mm crater in a 16 mm fibroid wide open, the serosa reference caves in around
    it, and a textbook type 5 reads as ~2% intramural. The radius must scale with
    the fibroid. The equivalent-sphere radius is the natural scale -- it is exactly
    half the widest crater the fibroid can cut.
    """
    r_eq = equivalent_radius_mm(fibroid, spacing) * float(cfg["body.adaptive_radius_scale"])
    return float(np.clip(r_eq, cfg["body.min_closing_radius_mm"],
                         cfg["body.max_closing_radius_mm"]))


def reconstruct(mask: np.ndarray, spacing, cfg, method: str | None = None,
                fibroid: np.ndarray | None = None) -> BodyModel:
    """Reconstruct the serosa and endometrial reference surfaces.

    ``method`` overrides ``body.method`` for the Phase 2 ablation. ``fibroid``
    selects one component to measure: it is withheld from the scaffold (so it
    cannot drag its own reference surface outward) and sets the bridging radius.
    Other fibroids stay in the scaffold -- they are real uterine tissue.
    """
    method = method or cfg["body.method"]
    spacing = tuple(float(s) for s in spacing)

    full_shape = mask.shape

    # Size the crop to *this* fibroid's bridging radius, not to the global maximum.
    # A flat 2 x max_closing_radius margin is 50 mm, which at 0.45 mm in-plane is
    # 111 voxels of empty air per side and inflates every distance transform by
    # roughly 4x in area. close_mm pads internally for correctness, so the crop
    # only has to keep the body right in the fibroid's neighbourhood.
    clamped, mass_frac = False, 0.0
    if fibroid is not None and cfg["body.adaptive_radius"]:
        r_body = bridging_radius_mm(fibroid, spacing, cfg)
        r_needed = equivalent_radius_mm(fibroid, spacing) * float(cfg["body.adaptive_radius_scale"])
        clamped = r_needed > float(cfg["body.max_closing_radius_mm"])
        r_cav_extra = r_body
    else:
        r_body = float(cfg["body.closing_radius_mm"])
        r_cav_extra = 0.0
    margin = max(r_body, float(cfg["body.cavity_closing_radius_mm"])) + 2.0 * max(spacing)

    roi = bounding_box(
        (mask == cfg["labels.wall"]) | (mask == cfg["labels.cavity"])
        | (mask == cfg["labels.fibroid"]), spacing, margin)
    mask = mask[roi]
    if fibroid is not None:
        fibroid = fibroid[roi]

    wall = mask == cfg["labels.wall"]
    cavity = mask == cfg["labels.cavity"]
    fibroid_all = mask == cfg["labels.fibroid"]

    # Fibroids are excluded on purpose: the serosa reference must not be dragged
    # outward by a fibroid that is bulging through it.
    if fibroid is not None and cfg["body.adaptive_radius"]:
        scaffold = wall | cavity | (fibroid_all & ~fibroid)
        denom = float((wall | cavity).sum() + fibroid.sum())
        mass_frac = float(fibroid.sum()) / denom if denom else 0.0
    else:
        scaffold = wall | cavity

    if method == "closing":
        body = fill_holes(close_mm(scaffold, spacing, r_body))
    elif method == "fill":
        body = fill_holes(scaffold)
    elif method == "hull":
        body = _hull_ablation(wall | cavity | fibroid_all)
    elif method == "union_closing":
        # The brief's original recipe, kept so the ablation can show what it does.
        body = fill_holes(close_mm(wall | cavity | fibroid_all, spacing, r_body))
    else:
        raise ValueError(f"unknown body.method {method!r}")

    # Prune the *filled* body to its dominant component, then add the raw scaffold
    # back. Pruning after the union would delete real annotated tissue -- on 21
    # patients that is >1% of the scaffold and on one it is 54%. Losing labelled
    # myometrium to tidy up the component count is the wrong trade, so the
    # invariant kept here is containment, and component dominance is reported as a
    # statistic instead of enforced.
    if cfg["body.keep_largest_component"]:
        body = largest_component(body)
    body |= scaffold                    # the reference must never lose real tissue

    # The endometrial reference needs the same treatment: a submucosal fibroid
    # dents the cavity by its own width, so the bridging radius scales with it too.
    r_cav = max(float(cfg["body.cavity_closing_radius_mm"]), r_cav_extra)
    r_cav = min(r_cav, float(cfg["body.max_closing_radius_mm"]))
    cavity_ref = fill_holes(close_mm(cavity, spacing, r_cav)) & body

    def _uncrop(a):
        full = np.zeros(full_shape, dtype=bool)
        full[roi] = a
        return full

    body, cavity_ref = _uncrop(body), _uncrop(cavity_ref)
    return BodyModel(body=body, cavity_ref=cavity_ref,
                     myometrium=body & ~cavity_ref, spacing=spacing, method=method,
                     radius_clamped=clamped, fibroid_mass_frac=mass_frac)


def surfaces(model: BodyModel) -> tuple[np.ndarray, np.ndarray]:
    """(serosa, endometrial) boundary voxels of the reconstruction."""
    er = ndimage.binary_erosion(model.body, ndimage.generate_binary_structure(3, 1))
    serosa = model.body & ~er
    ec = ndimage.binary_erosion(model.cavity_ref, ndimage.generate_binary_structure(3, 1))
    endo = model.cavity_ref & ~ec
    return serosa, endo


def percent_intramural(fibroid: np.ndarray, model: BodyModel) -> float:
    """100 * |F & body_ref & ~cavity_ref| / |F| -- the key measurement."""
    n = int(fibroid.sum())
    if n == 0:
        return float("nan")
    return 100.0 * float((fibroid & model.myometrium).sum()) / n


def briefs_percent_intramural(mask, spacing, fibroid, cfg) -> float:
    """The brief's formula, implemented literally, purely to prove it is degenerate.

    ``100 * |F & fill(close(W|C|F)) & ~C| / |F|`` with the *raw* cavity label. It
    returns 100% for every fibroid ever measured. Kept so a regression test can
    assert that, and so the ablation table can show it beside the real method.
    Never use it to produce a number anyone will read as a measurement.
    """
    wall = mask == cfg["labels.wall"]
    cavity = mask == cfg["labels.cavity"]
    fib_all = mask == cfg["labels.fibroid"]
    roi = bounding_box(wall | cavity | fib_all, spacing,
                       float(cfg["body.max_closing_radius_mm"]) * 2.0)
    body = fill_holes(close_mm((wall | cavity | fib_all)[roi], spacing,
                               float(cfg["body.closing_radius_mm"])))
    f = fibroid[roi]
    n = int(f.sum())
    return 100.0 * float((f & body & ~cavity[roi]).sum()) / n if n else float("nan")


# --------------------------------------------------- endometrial compartment


def compartment_split(fibroid, cavity, wall, body, spacing):
    """Split a fibroid into extraserosal / intracavitary / intramural fractions.

    Why the submucosal side is not a volume overlap with the cavity
    --------------------------------------------------------------
    The endometrial cavity is a *virtual* space. Across UMD its median volume is
    2.9 cm3 -- 2.0% of the uterus -- while the median fibroid is 44 cm3, fifteen
    times larger. Measuring "how much of the fibroid lies inside the cavity" as a
    volume intersection therefore cannot exceed 50% for any normally sized
    fibroid, and FIGO type 1 becomes unreachable by construction. That is exactly
    what the Phase 4 gate caught: 0 type-1 fibroids in 931, and only 2 of 110
    cavity-touching fibroids above 50% intracavitary.

    The question the FIGO grade actually asks is counterfactual: *what tissue
    would occupy this space if the fibroid were not here?* So each fibroid voxel
    is assigned to whichever compartment is nearer in millimetres -- the
    annotated endometrium or the myometrium. This is nearest-label interpolation
    of the two structures the fibroid displaced, it needs no surface fit and no
    orientation guess, and it degrades gracefully: a fibroid deep in the muscle
    is uniformly nearer the wall, one sitting in the cavity is uniformly nearer
    the endometrium, and one straddling the boundary splits along it.

    An earlier attempt fitted a local plane to the endometrium instead. It failed
    on exactly the case that matters least ambiguously -- a wholly intracavitary
    fibroid, which is ringed by endometrium on all sides and so has no base rim
    to fit a plane to, yielding a plane through its own centre and a 50/50 split.

    The serosal side needs none of this: the serosa bounds real tissue against
    nothing, so a volume fraction is well posed there and behaves.

    Returns ``(pct_extraserosal, pct_intracavitary, pct_intramural)``, which
    partition the fibroid exactly.
    """
    n = int(fibroid.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    outside = fibroid & ~body
    n_out = int(outside.sum())
    inner = fibroid & body
    n_in = int(inner.sum())
    if n_in == 0:
        return 100.0 * n_out / n, 0.0, 0.0

    if not cavity.any():
        return 100.0 * n_out / n, 0.0, 100.0 * n_in / n

    d_cav = ndimage.distance_transform_edt(~cavity, sampling=spacing)
    if wall.any():
        d_myo = ndimage.distance_transform_edt(~wall, sampling=spacing)
    else:
        d_myo = np.full_like(d_cav, np.inf)

    sel = np.nonzero(inner)
    toward_cavity = d_cav[sel] < d_myo[sel]
    n_cav = int(toward_cavity.sum())

    return (100.0 * n_out / n, 100.0 * n_cav / n, 100.0 * (n_in - n_cav) / n)
