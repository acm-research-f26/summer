"""Shared label and feature definitions."""

BACKGROUND_LABEL = 0
UTERINE_WALL_LABEL = 1
CAVITY_LABEL = 2
FIBROID_LABEL = 3
NABOTHIAN_CYST_LABEL = 4

AUDIT_COLUMNS = [
    "boundary_contact_count",
    "boundary_contact_ratio",
    "overlap_count",
    "overlap_ratio",
    "min_distance_to_cavity_mm",
]

PREDICTOR_COLUMNS = [
    "volume_voxels",
    "volume_mm3",
    "centroid_to_cavity_dist_mm",
    "aspect_ratio",
]

TARGET_COLUMN = "cavity_touching"
GROUP_COLUMN = "patient_id"
