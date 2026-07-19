
from __future__ import annotations

import csv
import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from config import PATHS, SEED

# One row per participant-screen; shared by the human instrument and the
# synthetic generator. Severity is deliberately not logged — analysis joins
# it via screen_id, so the log cannot drift from the score table.
RESPONSE_COLUMNS = [
    "response_id",
    "ts_iso",
    "participant_id",
    "source",       # "synthetic" | "human" — drives the SYNTHETIC labeling
    "screen_id",
    "condition",    # "manipulated" | "neutralized"
    "image_shown",
    "decision",     # 1 = complied with the screen's push, 0 = declined
    "rt_ms",
]

VALID_CONDITIONS = {"manipulated", "neutralized"}
VALID_SOURCES = {"synthetic", "human"}


def new_response_id() -> str:
    return uuid.uuid4().hex


def append_response(row: dict, path: Path | None = None) -> None:
    """Validate and append one response row. The only write path to the log —
    the Flask app and the synthetic generator both come through here."""
    path = Path(path) if path is not None else PATHS["responses"]

    missing = set(RESPONSE_COLUMNS) - set(row)
    extra = set(row) - set(RESPONSE_COLUMNS)
    if missing or extra:
        raise ValueError(f"bad response row: missing={sorted(missing)} extra={sorted(extra)}")
    if row["condition"] not in VALID_CONDITIONS:
        raise ValueError(f"condition {row['condition']!r} not in {VALID_CONDITIONS}")
    if row["source"] not in VALID_SOURCES:
        raise ValueError(f"source {row['source']!r} not in {VALID_SOURCES}")
    if int(row["decision"]) not in (0, 1):
        raise ValueError(f"decision must be 0/1, got {row['decision']!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESPONSE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_responses(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else PATHS["responses"]
    if not path.exists():
        raise FileNotFoundError(f"no responses logged yet at {path}")
    df = pd.read_csv(path)
    df["decision"] = df["decision"].astype(int)
    return df


def assign_conditions(participant_id: str, screen_ids: list, seed: int = SEED) -> dict:
    """Deterministic per-participant condition assignment, shared by the app
    and the synthetic generator. Each participant sees every stimulus once,
    each independently assigned manipulated or neutralized by a rng derived
    from (seed, participant_id).
    """
    pid_hash = int(hashlib.sha256(str(participant_id).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([seed, pid_hash])
    flips = rng.random(len(screen_ids)) < 0.5
    return {
        sid: ("manipulated" if flip else "neutralized")
        for sid, flip in zip(screen_ids, flips)
    }
