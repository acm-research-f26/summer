"""
data_loader.py  —  SleepGuard v2
Parses real OhioT1DM 2018 XML files (glucose only).
Handles the real format: DD-MM-YYYY HH:MM:SS timestamps.
Patients: 559, 563, 570, 575, 588, 591
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

PATIENT_IDS = [559, 563, 570, 575, 588, 591]
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
    Parse one patient XML. Returns dict with 'glucose' DataFrame.
    Other streams (HR, bolus, meal) are empty in real 2018 data.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    gluc_el = root.find("glucose_level")
    if gluc_el is not None:
        for event in gluc_el.findall("event"):
            ts_raw = event.get("ts")
            val    = event.get("value")
            if ts_raw and val:
                try:
                    rows.append({
                        "timestamp": _parse_ts(ts_raw),
                        "glucose":   float(val)
                    })
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return {"glucose": df}


def load_all_patients(data_dir):
    """
    Load all patients from a directory.
    Searches train/ subdir first, then root.
    Returns: dict { patient_id: {"glucose": DataFrame} }
    """
    patients = {}

    train_dir = os.path.join(data_dir, "train")
    search_dir = train_dir if os.path.isdir(train_dir) else data_dir

    for fname in sorted(os.listdir(search_dir)):
        if not fname.endswith(".xml"):
            continue
        try:
            pid = int(fname.split("-")[0])
        except ValueError:
            continue
        if pid not in PATIENT_IDS:
            continue

        path = os.path.join(search_dir, fname)
        pdata = parse_patient_xml(path)
        n = len(pdata["glucose"])
        print(f"  patient {pid}: {n} glucose readings")
        if n > 0:
            patients[pid] = pdata

    return patients


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    patients = load_all_patients(data_dir)
    print(f"\nLoaded {len(patients)} patients.")
