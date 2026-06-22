import math

import numpy as np

from fibroid_cavity.features import extract_patient_features


def test_extract_patient_features_separates_fibroids_and_labels_contact():
    mask = np.zeros((10, 10, 10), dtype=np.int16)
    mask[4:6, 4:6, 4:6] = 2
    mask[2:4, 4:6, 4:6] = 3
    mask[7:9, 7:9, 7:9] = 3

    rows = extract_patient_features(mask, spacing=(1.0, 1.0, 1.0), patient_id="case_001")

    assert len(rows) == 2
    assert sorted(row["cavity_touching"] for row in rows) == [0, 1]
    assert all(row["patient_id"] == "case_001" for row in rows)
    assert all(row["volume_voxels"] == 8 for row in rows)

    touching = next(row for row in rows if row["cavity_touching"] == 1)
    distant = next(row for row in rows if row["cavity_touching"] == 0)

    assert touching["boundary_contact_count"] > 0
    assert distant["boundary_contact_count"] == 0
    assert touching["min_distance_to_cavity_mm"] == 1.0
    assert distant["min_distance_to_cavity_mm"] > touching["min_distance_to_cavity_mm"]


def test_extract_patient_features_uses_voxel_spacing_for_volume_and_bbox():
    mask = np.zeros((8, 8, 8), dtype=np.int16)
    mask[3:5, 3:5, 3:5] = 2
    mask[0:2, 0:4, 0:2] = 3

    rows = extract_patient_features(mask, spacing=(2.0, 1.0, 3.0), patient_id="case_002")

    assert len(rows) == 1
    row = rows[0]
    assert row["volume_voxels"] == 16
    assert row["volume_mm3"] == 96.0
    assert row["bbox_size_x_mm"] == 4.0
    assert row["bbox_size_y_mm"] == 4.0
    assert row["bbox_size_z_mm"] == 6.0
    assert row["aspect_ratio"] == 1.5


def test_extract_patient_features_returns_empty_when_no_fibroids():
    mask = np.zeros((5, 5, 5), dtype=np.int16)
    mask[2:4, 2:4, 2:4] = 2

    rows = extract_patient_features(mask, spacing=(1.0, 1.0, 1.0), patient_id="case_003")

    assert rows == []


def test_extract_patient_features_handles_missing_cavity():
    mask = np.zeros((5, 5, 5), dtype=np.int16)
    mask[1:3, 1:3, 1:3] = 3

    rows = extract_patient_features(mask, spacing=(1.0, 1.0, 1.0), patient_id="case_004")

    assert len(rows) == 1
    assert rows[0]["cavity_touching"] == 0
    assert math.isinf(rows[0]["min_distance_to_cavity_mm"])
    assert math.isnan(rows[0]["centroid_to_cavity_dist_mm"])
