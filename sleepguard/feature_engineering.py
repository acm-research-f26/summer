"""
feature_engineering.py
For each patient, for each night:
  1. Label the night as hypoglycemic (1) or normal (0)
  2. Extract pre-sleep baseline features  (glucose, insulin, carbs)
  3. Extract pre-sleep HRV proxy features (RMSSD, SDNN, pNN50, mean HR)
Returns a combined DataFrame ready for modelling.

NOTE ON HRV FROM 5-MIN DATA
The Basis Peak aggregates HR every 5 minutes, so we don't have raw
RR intervals.  We derive pseudo-HRV features by treating each 5-min
HR value as a proxy for the mean RR in that window:
  RR_ms = 60000 / HR_bpm
Then compute RMSSD, SDNN, pNN50 over the sequence of RR values.
This is standard practice in the literature when only aggregated HR
is available (e.g. Mohebbi et al. 2022).
"""

import numpy as np
import pandas as pd
from datetime import time, timedelta

# ── Constants ──────────────────────────────────────────────────────────────────
HYPO_THRESHOLD   = 70.0        # mg/dL
SLEEP_START_HOUR = 23          # 11 PM
SLEEP_END_HOUR   = 7           #  7 AM
PRE_SLEEP_HOURS  = 2           # window before sleep_start to extract features


# ── Helper: get date range covered by a patient ────────────────────────────────

def _get_date_range(patient_data):
    g = patient_data["glucose"]
    if g.empty:
        return []
    dates = pd.to_datetime(g["timestamp"]).dt.date.unique()
    return sorted(dates)


# ── 1. Nocturnal hypoglycemia labeling ─────────────────────────────────────────

def label_nights(patient_data):
    """
    For each calendar date D, define the nocturnal window as
    D 23:00 → D+1 07:00.  Label = 1 if any glucose < 70 in that window.

    Returns: pd.Series indexed by date (D), values 0 or 1.
    """
    g = patient_data["glucose"].copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"])
    g["date"]      = g["timestamp"].dt.date

    labels = {}
    dates  = _get_date_range(patient_data)

    for d in dates:
        from datetime import datetime, date
        night_start = datetime.combine(d,                   time(SLEEP_START_HOUR, 0))
        night_end   = datetime.combine(d + timedelta(days=1), time(SLEEP_END_HOUR,   0))

        mask   = (g["timestamp"] >= night_start) & (g["timestamp"] < night_end)
        window = g.loc[mask, "value"]

        if len(window) < 3:       # not enough data — skip
            continue
        labels[d] = int((window < HYPO_THRESHOLD).any())

    return pd.Series(labels, name="label")


# ── 2. Baseline features ───────────────────────────────────────────────────────

def extract_baseline_features(patient_data, date):
    """
    Features from the 2-hour pre-sleep window (21:00-23:00 on `date`):
      - glucose_mean, glucose_std, glucose_last, glucose_slope
      - total_bolus_dose, total_basal_rate, total_carbs
    """
    from datetime import datetime
    win_end   = datetime.combine(date, time(SLEEP_START_HOUR, 0))
    win_start = win_end - timedelta(hours=PRE_SLEEP_HOURS)

    def _window(df, ts_col="timestamp"):
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col])
        mask = (df[ts_col] >= win_start) & (df[ts_col] < win_end)
        return df.loc[mask]

    # Glucose features
    g_win = _window(patient_data["glucose"])
    if len(g_win) < 2:
        g_mean  = np.nan
        g_std   = np.nan
        g_last  = np.nan
        g_slope = np.nan
    else:
        vals    = g_win["value"].values
        g_mean  = float(np.mean(vals))
        g_std   = float(np.std(vals))
        g_last  = float(vals[-1])
        # slope via linear regression on minute index
        minutes = ((g_win["timestamp"] - win_start)
                   .dt.total_seconds() / 60).values
        g_slope = float(np.polyfit(minutes, vals, 1)[0])

    # Insulin features
    b_win        = _window(patient_data["bolus"])
    total_bolus  = float(b_win["value"].sum()) if not b_win.empty else 0.0

    bas_win      = _window(patient_data["basal"])
    total_basal  = float(bas_win["value"].mean()) if not bas_win.empty else 0.0

    # Carb features
    m_win = patient_data["meal"].copy()
    if not m_win.empty:
        m_win["timestamp"] = pd.to_datetime(m_win["timestamp"])
        mask = (m_win["timestamp"] >= win_start) & (m_win["timestamp"] < win_end)
        total_carbs = float(m_win.loc[mask, "carbs"].sum())
    else:
        total_carbs = 0.0

    return {
        "glucose_mean":  g_mean,
        "glucose_std":   g_std,
        "glucose_last":  g_last,
        "glucose_slope": g_slope,
        "total_bolus":   total_bolus,
        "total_basal":   total_basal,
        "total_carbs":   total_carbs,
    }


# ── 3. HRV proxy features ──────────────────────────────────────────────────────

def _compute_hrv_features(hr_values):
    """
    Given an array of HR values (bpm) convert to pseudo-RR intervals
    and compute HRV metrics.
    """
    hr  = np.array(hr_values, dtype=float)
    hr  = hr[hr > 0]                  # drop zeros/missing
    if len(hr) < 4:
        return {k: np.nan for k in
                ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hr_mean", "hr_std"]}

    rr  = 60000.0 / hr                # ms

    # successive differences
    diff    = np.diff(rr)
    rmssd   = float(np.sqrt(np.mean(diff ** 2)))
    sdnn    = float(np.std(rr, ddof=1))
    pnn50   = float(np.mean(np.abs(diff) > 50) * 100)   # %
    hr_mean = float(np.mean(hr))
    hr_std  = float(np.std(hr, ddof=1))

    return {
        "hrv_rmssd": rmssd,
        "hrv_sdnn":  sdnn,
        "hrv_pnn50": pnn50,
        "hr_mean":   hr_mean,
        "hr_std":    hr_std,
    }


def extract_hrv_features(patient_data, date):
    """
    HRV proxy features from the 2-hour pre-sleep window.
    """
    from datetime import datetime
    win_end   = datetime.combine(date, time(SLEEP_START_HOUR, 0))
    win_start = win_end - timedelta(hours=PRE_SLEEP_HOURS)

    hr = patient_data["heart_rate"].copy()
    if hr.empty:
        return {k: np.nan for k in
                ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hr_mean", "hr_std"]}

    hr["timestamp"] = pd.to_datetime(hr["timestamp"])
    mask = (hr["timestamp"] >= win_start) & (hr["timestamp"] < win_end)
    hr_win = hr.loc[mask, "value"].values

    return _compute_hrv_features(hr_win)


# ── 4. Build full feature matrix ───────────────────────────────────────────────

def build_feature_matrix(patients_data):
    """
    Iterates over all patients and all nights, extracts features and labels.
    Returns: X_baseline, X_hrv, y, meta_df
    """
    rows = []

    for pid, pdata in patients_data.items():
        labels = label_nights(pdata)

        for date, label in labels.items():
            base = extract_baseline_features(pdata, date)
            hrv  = extract_hrv_features(pdata, date)

            row = {"patient_id": pid, "date": date, "label": label}
            row.update(base)
            row.update(hrv)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["glucose_mean"])    # drop nights with no glucose data

    baseline_cols = ["glucose_mean", "glucose_std", "glucose_last",
                     "glucose_slope", "total_bolus", "total_basal", "total_carbs"]
    hrv_cols      = ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hr_mean", "hr_std"]

    X_baseline = df[baseline_cols].copy()
    X_hrv      = df[baseline_cols + hrv_cols].copy()
    y          = df["label"].copy()
    meta       = df[["patient_id", "date"]].copy()

    print(f"\nFeature matrix built: {len(df)} nights, "
          f"{int(y.sum())} hypoglycemic ({100*y.mean():.1f}%)")

    return X_baseline, X_hrv, y, meta


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all_patients
    synthetic_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    patients = load_all_patients(synthetic_dir)
    Xb, Xh, y, meta = build_feature_matrix(patients)
    print(Xb.head())
    print(Xh.head())
