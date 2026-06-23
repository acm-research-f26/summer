"""
generate_synthetic_data.py
Generates synthetic OhioT1DM-style XML files for 6 patients.
Mirrors the exact schema of the real 2018 cohort so the pipeline
runs identically on both synthetic and real data.
"""

import numpy as np
import os
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from xml.dom import minidom

PATIENT_IDS = [540, 544, 552, 567, 584, 596]
STUDY_DAYS   = 56
SEED         = 42
np.random.seed(SEED)

def simulate_glucose_day(base=120.0, hypo_night=False):
    readings = []
    g = base + np.random.normal(0, 10)
    for m in range(0, 1440, 5):
        noise = np.random.normal(0, 3)
        trend = np.sin(m / 1440 * 2 * np.pi) * 15
        g += trend * 0.05 + noise
        g = np.clip(g, 40, 400)
        readings.append((m, round(g, 1)))
    if hypo_night:
        dip_start = np.random.randint(0, 72) * 5
        dip_depth = np.random.uniform(45, 69)
        for i, (m, _) in enumerate(readings):
            if dip_start <= m <= dip_start + 90:
                readings[i] = (m, round(dip_depth + np.random.normal(0, 2), 1))
    return readings

def simulate_hr_day(hypo_night=False):
    readings = []
    for m in range(0, 1440, 5):
        if 0 <= m < 360:
            base = 55 if not hypo_night else 70
        elif 360 <= m < 480:
            base = 65
        else:
            base = 72
        hr = base + np.random.normal(0, 4)
        hr = np.clip(hr, 40, 140)
        readings.append((m, round(hr, 1)))
    return readings

def simulate_bolus_events(day_start):
    events = []
    for meal_hour in [7, 12, 18]:
        if np.random.rand() > 0.1:
            t = day_start + timedelta(hours=meal_hour, minutes=int(np.random.randint(-15, 15)))
            dose = round(np.random.uniform(2, 8), 2)
            events.append((t, dose))
    return events

def simulate_basal_events(day_start):
    t = day_start
    rate = round(np.random.uniform(0.5, 1.5), 2)
    return [(t, rate)]

def simulate_meal_events(day_start):
    events = []
    for meal_hour in [7, 12, 18]:
        if np.random.rand() > 0.15:
            t = day_start + timedelta(hours=meal_hour, minutes=int(np.random.randint(-10, 10)))
            carb = int(np.random.randint(20, 80))
            events.append((t, carb))
    return events

def build_patient_xml(patient_id, split="training"):
    days = 42 if split == "training" else 14
    root     = ET.Element("patient", id=str(patient_id))
    gluc_el  = ET.SubElement(root, "glucose_level")
    hr_el    = ET.SubElement(root, "basis_heart_rate")
    bolus_el = ET.SubElement(root, "bolus")
    basal_el = ET.SubElement(root, "basal")
    meal_el  = ET.SubElement(root, "meal")
    sleep_el = ET.SubElement(root, "basis_sleep")

    hypo_nights = set(np.where(np.random.rand(days) < 0.15)[0].tolist())
    study_start = datetime(2018, 1, 1, 0, 0, 0)
    ts_fmt = "%d-%m-%Y %H:%M:%S"

    for day_idx in range(days):
        day_start = study_start + timedelta(days=day_idx)
        is_hypo = day_idx in hypo_nights

        for (m, g) in simulate_glucose_day(hypo_night=is_hypo):
            t = day_start + timedelta(minutes=m)
            ET.SubElement(gluc_el, "event", ts=t.strftime(ts_fmt), value=str(g))

        for (m, hr) in simulate_hr_day(hypo_night=is_hypo):
            t = day_start + timedelta(minutes=m)
            ET.SubElement(hr_el, "event", ts=t.strftime(ts_fmt), value=str(hr))

        for (t, dose) in simulate_bolus_events(day_start):
            ET.SubElement(bolus_el, "event", ts=t.strftime(ts_fmt), dose=str(dose), bwz_carb_input="0")

        for (t, rate) in simulate_basal_events(day_start):
            ET.SubElement(basal_el, "event", ts=t.strftime(ts_fmt), value=str(rate))

        for (t, carb) in simulate_meal_events(day_start):
            ET.SubElement(meal_el, "event", ts=t.strftime(ts_fmt), carbs=str(carb))

        sleep_start = day_start + timedelta(hours=22)
        sleep_end   = day_start + timedelta(hours=30)
        ET.SubElement(sleep_el, "event",
                      begin=sleep_start.strftime(ts_fmt),
                      end=sleep_end.strftime(ts_fmt))

    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines = pretty.split("\n")[1:]
    return "\n".join(lines)

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        label = "training" if split == "train" else "testing"
        for pid in PATIENT_IDS:
            xml_str  = build_patient_xml(pid, split=label)
            filename = f"{pid}-ws-{label}.xml"
            path     = os.path.join(split_dir, filename)
            with open(path, "w") as f:
                f.write(xml_str)
            print(f"  wrote {path}")
    print("\nSynthetic data generation complete.")

if __name__ == "__main__":
    main()
