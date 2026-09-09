"""Robust NIfTI / DICOM I/O.

Phase 1. Two verified dataset traps this module exists to defeat (PLAN.md F1/F2):

F1  33 of the 300 masks are named ``_seq`` (a dataset typo) instead of ``_seg``,
    and those same 33 are *raw uncompressed* NIfTI despite carrying a ``.gz``
    extension. Globbing ``_seg`` or trusting the extension drops 11% of the cohort.

F2  Those 33 also carry an unset ``pixdim`` of (1, 1, 1). Their DICOM headers show
    the truth: ~0.446 x 0.446 x ~5 mm, ``MRAcquisitionType = 2D``. There is no
    native-isotropic subset in UMD.

    The damage is subtle rather than loud: true voxel volume across those 33 is
    0.83-1.29 mm^3 (mean 1.02), so (1, 1, 1) gets *volume* right to within ~20%
    and a volume sanity-check sails straight past it. What it destroys is the
    anisotropy -- true 11.6x, claimed 1.0x. Every millimetre-sized structuring
    element, diameter, sphericity and perturbation downstream would be computed
    on a voxel that is 11x taller than the code believes.

F3  Naming defects are not confined to the ``_seq`` files. UMD_221129_037 holds
    ``UMD_2221129_037_seg.nii.gz`` -- an extra digit -- and is also raw. 34 files
    are raw in total. This is why the container is sniffed and never inferred.

Therefore: sniff the container, and treat DICOM as authoritative for spacing.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class Mask:
    """A loaded segmentation mask plus the provenance needed to trust it."""

    data: np.ndarray
    nifti_spacing: tuple[float, ...]
    container: str          # "gzip" | "raw"
    filename_kind: str      # "seg" | "seq"
    path: Path

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def labels(self) -> set[int]:
        return {int(v) for v in np.unique(self.data)}


def _read_maybe_gzip(path: Path) -> tuple[bytes, str]:
    """Return decompressed bytes and which container they actually came in.

    The extension is not consulted. Only the magic number is.
    """
    raw = path.read_bytes()
    if raw[:2] == GZIP_MAGIC:
        return gzip.decompress(raw), "gzip"
    return raw, "raw"


def load_mask(path: str | Path) -> Mask:
    """Load a segmentation mask, sniffing gzip magic rather than trusting ``.gz``."""
    path = Path(path)
    payload, container = _read_maybe_gzip(path)

    fh = nib.FileHolder(fileobj=BytesIO(payload))
    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})

    data = np.asarray(img.dataobj)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3])
    data = np.rint(data).astype(np.uint8)

    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    kind = "seq" if "_seq" in path.name else "seg"
    return Mask(data=data, nifti_spacing=zooms, container=container,
                filename_kind=kind, path=path)


def find_masks(root: str | Path, pattern: str) -> list[Path]:
    """All mask files under ``root``. The pattern must cover the ``_seq`` typo."""
    return sorted(Path(root).glob(f"*/{pattern}"))


# --------------------------------------------------------------------------- DICOM


@dataclass(frozen=True)
class DicomGeometry:
    spacing: tuple[float, float, float] | None
    n_slices: int
    rows: int | None
    cols: int | None
    slice_source: str          # how sz was determined
    acquisition_type: str | None
    series_description: str | None
    series_uid: str | None


def _slice_spacing_from_positions(slices: list) -> tuple[float | None, str]:
    """Slice spacing measured from ImagePositionPatient — the most authoritative source.

    Projects each slice origin onto the slice normal (from ImageOrientationPatient)
    and takes the median consecutive gap. Falls back to header fields when the
    positions are missing or degenerate.
    """
    pts, normal = [], None
    for s in slices:
        ipp = getattr(s, "ImagePositionPatient", None)
        iop = getattr(s, "ImageOrientationPatient", None)
        if ipp is None:
            continue
        if normal is None and iop is not None:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            normal = np.cross(r, c)
        pts.append(np.array(ipp, float))

    if len(pts) < 2:
        return None, "insufficient-positions"
    if normal is None or not np.any(normal):
        normal = np.array([0.0, 0.0, 1.0])

    proj = np.sort([float(np.dot(p, normal)) for p in pts])
    gaps = np.diff(proj)
    gaps = gaps[gaps > 1e-6]
    if gaps.size == 0:
        return None, "degenerate-positions"
    return float(np.median(gaps)), "ImagePositionPatient"


def dicom_spacing(patient_dir: str | Path,
                  match_shape: tuple[int, ...] | None = None) -> DicomGeometry:
    """Authoritative voxel spacing from a patient's DICOM series.

    When ``match_shape`` is given, the series whose (rows, cols, n_slices) matches
    the mask is preferred; otherwise the largest series wins. This matters because
    a patient directory can hold more than one acquisition.
    """
    patient_dir = Path(patient_dir)
    files = sorted(patient_dir.glob("*.dcm"))
    if not files:
        return DicomGeometry(None, 0, None, None, "no-dicom", None, None, None)

    headers = []
    for f in files:
        try:
            headers.append(pydicom.dcmread(f, stop_before_pixels=True, force=True))
        except Exception:
            continue
    if not headers:
        return DicomGeometry(None, 0, None, None, "unreadable-dicom", None, None, None)

    series: dict[str, list] = {}
    for h in headers:
        series.setdefault(str(getattr(h, "SeriesInstanceUID", "unknown")), []).append(h)

    def score(item):
        uid, group = item
        h = group[0]
        rows, cols = getattr(h, "Rows", None), getattr(h, "Columns", None)
        exact = (match_shape is not None and rows == match_shape[0]
                 and cols == match_shape[1] and len(group) == match_shape[2])
        return (exact, len(group))

    uid, group = max(series.items(), key=score)
    head = group[0]

    ps = getattr(head, "PixelSpacing", None)
    sx = sy = None
    if ps is not None and len(ps) >= 2:
        sy, sx = float(ps[0]), float(ps[1])   # DICOM orders PixelSpacing as [row, col]

    sz, src = _slice_spacing_from_positions(group)
    if sz is None:
        sbs = getattr(head, "SpacingBetweenSlices", None)
        thk = getattr(head, "SliceThickness", None)
        if sbs is not None:
            sz, src = float(sbs), "SpacingBetweenSlices"
        elif thk is not None:
            sz, src = float(thk), "SliceThickness"
        else:
            src = "unavailable"

    spacing = (sx, sy, sz) if None not in (sx, sy, sz) else None
    return DicomGeometry(
        spacing=spacing,
        n_slices=len(group),
        rows=getattr(head, "Rows", None),
        cols=getattr(head, "Columns", None),
        slice_source=src,
        acquisition_type=str(getattr(head, "MRAcquisitionType", "") or "") or None,
        series_description=str(getattr(head, "SeriesDescription", "") or "") or None,
        series_uid=uid,
    )


# ------------------------------------------------------------------- spacing policy


def _plausible(spacing, cfg) -> bool:
    if spacing is None or any(s is None or s <= 0 for s in spacing):
        return False
    sx, sy, sz = spacing
    if not (cfg["spacing.min_in_plane_mm"] <= sx <= cfg["spacing.max_in_plane_mm"]):
        return False
    if not (cfg["spacing.min_in_plane_mm"] <= sy <= cfg["spacing.max_in_plane_mm"]):
        return False
    if not (cfg["spacing.min_slice_mm"] <= sz <= cfg["spacing.max_slice_mm"]):
        return False
    if cfg["spacing.reject_unit_isotropic"] and np.allclose(spacing, (1.0, 1.0, 1.0)):
        return False
    return True


def resolve_spacing(nifti_spacing, dicom_geom: DicomGeometry, cfg):
    """Choose the spacing to use, honouring ``spacing.source_priority``.

    DICOM outranks NIfTI. A candidate is only accepted if it is physically
    plausible, which is what rejects the (1, 1, 1) unset-pixdim artifact.
    Returns ``(spacing, source)``; source is "dicom", "nifti", or "none".
    """
    candidates = {
        "dicom": tuple(dicom_geom.spacing) if dicom_geom.spacing else None,
        "nifti": tuple(float(s) for s in nifti_spacing[:3]) if nifti_spacing else None,
    }
    for source in cfg["spacing.source_priority"]:
        cand = candidates.get(source)
        if _plausible(cand, cfg):
            return cand, source
    return None, "none"
