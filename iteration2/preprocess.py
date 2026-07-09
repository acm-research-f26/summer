"""
preprocess.py
Load HAI CSVs, extract actuator/command columns (our novel twist vs raw
sensor detection), normalize, and produce sliding-window tensors.

The key design choice: we detect anomalies in the ACTUATOR/COMMAND space
rather than the sensor reading space. This maps directly to GridLock's
threat model -- an indirect prompt injection attack manifests as an
unusual command being issued, not necessarily an unusual sensor reading.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))

WINDOW_SIZE = 30  # timesteps per window
STEP_SIZE = 1     # sliding window step

# HAI 21.03 actuator column name patterns -- these are command/setpoint cols.
# Real HAI columns: anything ending in 'Z' (valve position), 'MCP' (pump),
# 'HV' (hand valve), or 'MV' (motor valve). Adjust if using real data.
ACTUATOR_PATTERNS = ["PCV", "MCP", "HV", "MV", "LCV", "B2004", "B2016"]


def load_csv(path):
    df = pd.read_csv(path)
    # drop timestamp if present
    for col in ["timestamp", "time", "Timestamp", "Time", "datetime"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def get_actuator_cols(df):
    """
    Identify actuator/command columns by name pattern.
    On real HAI data, these are the columns that represent valve positions,
    pump states, and setpoint commands -- what an operator (or LLM agent)
    actually controls.
    """
    actuator_cols = [
        c for c in df.columns
        if any(pat in c for pat in ACTUATOR_PATTERNS) and c != "attack"
    ]
    if not actuator_cols:
        # fallback: use all non-attack columns
        print("[warn] no actuator columns matched patterns, using all features")
        actuator_cols = [c for c in df.columns if c != "attack"]
    return actuator_cols


def make_windows(data: np.ndarray, labels: np.ndarray):
    """Sliding windows over time series. Label = 1 if any attack in window."""
    X, y = [], []
    for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        X.append(data[i: i + WINDOW_SIZE])
        y.append(1 if labels[i: i + WINDOW_SIZE].any() else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def load_and_preprocess(train_path=None,
                         test_path=None,
                         scaler_path=None):
    if train_path is None: train_path = _os.path.join(_BASE_DIR, "data", "train1.csv")
    if test_path is None: test_path = _os.path.join(_BASE_DIR, "data", "test1.csv")
    if scaler_path is None: scaler_path = _os.path.join(_BASE_DIR, "data", "scaler.pkl")
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    actuator_cols = get_actuator_cols(train_df)
    print(f"Using {len(actuator_cols)} actuator/command columns: {actuator_cols}")

    # Labels
    train_labels = (train_df["attack"].values != 0).astype(int)
    test_labels = (test_df["attack"].values != 0).astype(int)

    train_feat = train_df[actuator_cols].values.astype(np.float32)
    test_feat = test_df[actuator_cols].values.astype(np.float32)

    # Fit scaler on training data only
    scaler = MinMaxScaler()
    train_feat = scaler.fit_transform(train_feat)
    test_feat = scaler.transform(test_feat)

    os.makedirs("data", exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump((scaler, actuator_cols), f)

    X_train, y_train = make_windows(train_feat, train_labels)
    X_test, y_test = make_windows(test_feat, test_labels)

    print(f"Train windows: {len(X_train)} ({y_train.sum()} attack)")
    print(f"Test windows:  {len(X_test)} ({y_test.sum()} attack)")

    return X_train, y_train, X_test, y_test, len(actuator_cols)


if __name__ == "__main__":
    load_and_preprocess()
