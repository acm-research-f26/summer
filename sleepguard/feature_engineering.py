"""
feature_engineering.py  —  SleepGuard v2
Glucose-only feature extraction from the pre-sleep window.

Nocturnal window  : 23:00 on date D  →  07:00 on date D+1  (label)
Pre-sleep window  : 19:00 – 23:00 on date D  (features, 4 hours)

Features extracted (13 total):
  Trajectory  : mean, std, min, max, last value, range
  Trend       : linear slope, acceleration (2nd-order coeff)
  Variability : coefficient of variation, MODD proxy
  Risk        : % readings < 100 mg/dL, % readings < 80 mg/dL
  Coverage    : fraction of expected 5-min readings present
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time

HYPO_THRESHOLD    = 70.0
SLEEP_START_HOUR  = 23
SLEEP_END_HOUR    = 7
PRE_SLEEP_START   = 19     # feature window starts 4h before sleep
PRE_SLEEP_END     = 23
MIN_COVERAGE      = 0.50   # drop nights where < 50% of window has data
MIN_READINGS      = 6      # absolute minimum readings in pre-sleep window


def _get_dates(patient_data):
    g = patient_data["glucose"]
    if g.empty:
        return []
    return sorted(pd.to_datetime(g["timestamp"]).dt.date.unique())


def label_nights(patient_data):
    """
    Label each night D as 1 (hypoglycemic) or 0 (normal).
    Night window: D 23:00 → D+1 07:00.
    Requires at least 6 readings in the nocturnal window to label.
    """
    g = patient_data["glucose"].copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"])
    labels = {}

    for d in _get_dates(patient_data):
        night_start = datetime.combine(d,                      time(SLEEP_START_HOUR, 0))
        night_end   = datetime.combine(d + timedelta(days=1),  time(SLEEP_END_HOUR,   0))
        mask   = (g["timestamp"] >= night_start) & (g["timestamp"] < night_end)
        window = g.loc[mask, "glucose"]
        if len(window) < 6:
            continue
        labels[d] = int((window < HYPO_THRESHOLD).any())

    return pd.Series(labels, name="label")


def extract_features(patient_data, date):
    """
    Extract 13 glucose features from the pre-sleep window
    (PRE_SLEEP_START:00 – PRE_SLEEP_END:00) on `date`.
    Returns dict of features, or None if coverage too low.
    """
    g = patient_data["glucose"].copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"])

    win_start = datetime.combine(date, time(PRE_SLEEP_START, 0))
    win_end   = datetime.combine(date, time(PRE_SLEEP_END,   0))
    mask      = (g["timestamp"] >= win_start) & (g["timestamp"] < win_end)
    win       = g.loc[mask].copy().reset_index(drop=True)

    # Coverage check
    expected  = (PRE_SLEEP_END - PRE_SLEEP_START) * 12  # 12 readings/hour at 5-min
    coverage  = len(win) / expected
    if len(win) < MIN_READINGS or coverage < MIN_COVERAGE:
        return None

    vals    = win["glucose"].values
    minutes = ((win["timestamp"] - win_start)
               .dt.total_seconds().values / 60)

    # Trajectory
    g_mean  = float(np.mean(vals))
    g_std   = float(np.std(vals))
    g_min   = float(np.min(vals))
    g_max   = float(np.max(vals))
    g_last  = float(vals[-1])
    g_range = float(g_max - g_min)

    # Trend — linear slope (mg/dL per minute)
    coeffs    = np.polyfit(minutes, vals, 2)
    g_slope   = float(coeffs[1])        # linear term
    g_accel   = float(coeffs[0])        # quadratic term (acceleration)

    # Variability
    g_cv      = float(g_std / g_mean) if g_mean > 0 else 0.0
    # MODD proxy: mean absolute difference between consecutive readings
    g_modd    = float(np.mean(np.abs(np.diff(vals)))) if len(vals) > 1 else 0.0

    # Risk indicators
    pct_u100  = float(np.mean(vals < 100) * 100)
    pct_u80   = float(np.mean(vals < 80)  * 100)

    # Coverage
    cov       = float(coverage)

    return {
        "g_mean":    g_mean,
        "g_std":     g_std,
        "g_min":     g_min,
        "g_max":     g_max,
        "g_last":    g_last,
        "g_range":   g_range,
        "g_slope":   g_slope,
        "g_accel":   g_accel,
        "g_cv":      g_cv,
        "g_modd":    g_modd,
        "pct_u100":  pct_u100,
        "pct_u80":   pct_u80,
        "coverage":  cov,
    }


def build_feature_matrix(patients_data):
    """
    Build feature matrix across all patients and nights.
    Returns: X (DataFrame), y (Series), meta (DataFrame)
    """
    rows = []
    skipped_coverage = 0
    skipped_label    = 0

    for pid, pdata in patients_data.items():
        labels = label_nights(pdata)

        for date, label in labels.items():
            feats = extract_features(pdata, date)
            if feats is None:
                skipped_coverage += 1
                continue
            row = {"patient_id": pid, "date": date, "label": label}
            row.update(feats)
            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\nNights included  : {len(df)}")
    print(f"Nights skipped   : {skipped_coverage} (low coverage)")
    print(f"Hypoglycemic     : {int(df['label'].sum())} "
          f"({100*df['label'].mean():.1f}%)")
    print(f"Normal           : {int((df['label']==0).sum())}")

    feature_cols = [c for c in df.columns
                    if c not in ("patient_id", "date", "label")]
    X    = df[feature_cols].copy()
    y    = df["label"].copy()
    meta = df[["patient_id", "date"]].copy()

    return X, y, meta
