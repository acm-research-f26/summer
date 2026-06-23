"""
data_loader.py
Parses OhioT1DM XML files (real or synthetic) into pandas DataFrames.
Handles both the real dataset format and our synthetic format.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

PATIENT_IDS = [540, 544, 552, 567, 584, 596]
TS_FORMATS  = ["%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]

def _parse_ts(ts_str):
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")

def parse_patient_xml(xml_path):
    """
    Parse one patient XML file.
    Returns a dict of DataFrames keyed by stream name:
      glucose, heart_rate, bolus, basal, meal, sleep
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _extract(tag, ts_attr, val_attr):
        rows = []
        el = root.find(tag)
        if el is None:
            return pd.DataFrame(columns=["timestamp", "value"])
        for event in el.findall("event"):
            ts_raw = event.get(ts_attr)
            val    = event.get(val_attr)
            if ts_raw is not None and val is not None:
                try:
                    rows.append({"timestamp": _parse_ts(ts_raw),
                                 "value":     float(val)})
                except Exception:
                    pass
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _extract_sleep():
        rows = []
        el = root.find("basis_sleep")
        if el is None:
            return pd.DataFrame(columns=["begin", "end"])
        for event in el.findall("event"):
            begin = event.get("begin")
            end   = event.get("end")
            if begin and end:
                try:
                    rows.append({"begin": _parse_ts(begin),
                                 "end":   _parse_ts(end)})
                except Exception:
                    pass
        return pd.DataFrame(rows)

    def _extract_meal():
        rows = []
        el = root.find("meal")
        if el is None:
            return pd.DataFrame(columns=["timestamp", "carbs"])
        for event in el.findall("event"):
            ts_raw = event.get("ts")
            carbs  = event.get("carbs")
            if ts_raw and carbs:
                try:
                    rows.append({"timestamp": _parse_ts(ts_raw),
                                 "carbs":     float(carbs)})
                except Exception:
                    pass
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    return {
        "glucose":    _extract("glucose_level",    "ts",  "value"),
        "heart_rate": _extract("basis_heart_rate", "ts",  "value"),
        "bolus":      _extract("bolus",            "ts",  "dose"),
        "basal":      _extract("basal",            "ts",  "value"),
        "meal":       _extract_meal(),
        "sleep":      _extract_sleep(),
    }


def load_all_patients(data_dir):
    """
    Load all 6 patients from a directory containing XML files.
    Automatically detects training/testing subdirs or flat layout.
    Returns: dict { patient_id: {stream: DataFrame} }
    """
    patients = {}

    # Try subdirectory layout first (train/test split dirs)
    train_dir = os.path.join(data_dir, "train")
    if os.path.isdir(train_dir):
        search_dir = train_dir
    else:
        search_dir = data_dir

    for fname in os.listdir(search_dir):
        if not fname.endswith(".xml"):
            continue
        pid = int(fname.split("-")[0])
        if pid not in PATIENT_IDS:
            continue
        path = os.path.join(search_dir, fname)
        patients[pid] = parse_patient_xml(path)
        print(f"  loaded patient {pid}: "
              f"{len(patients[pid]['glucose'])} glucose readings, "
              f"{len(patients[pid]['heart_rate'])} HR readings")

    return patients


if __name__ == "__main__":
    # Quick smoke test
    synthetic_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    patients = load_all_patients(synthetic_dir)
    print(f"\nLoaded {len(patients)} patients.")
