#!/usr/bin/env python3
"""Debug script to examine hypnogram data."""

import pyedflib
from pathlib import Path
from RAVEN_python.score_results import load_hypnogram

# Get first hypnogram file
hypnogram_dir = Path(__file__).resolve().parent / 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette'
hypnogram_files = list(hypnogram_dir.glob('*Hypnogram.edf'))

if not hypnogram_files:
    print(f"No hypnogram files found in {hypnogram_dir}")
    exit(1)

hypnogram_file = hypnogram_files[0]
print(f"\n=== Examining: {hypnogram_file.name} ===\n")

# Read raw annotations from EDF
edf = pyedflib.EdfReader(str(hypnogram_file))
annotations = edf.readAnnotations()
edf.close()

print(f"Raw annotations count: {len(annotations)}")
print(f"Annotations type: {type(annotations)}")
if annotations:
    print(f"First annotation type: {type(annotations[0])}")
    print(f"First annotation: {annotations[0]}")
print()

# Load using our function
timeline = load_hypnogram(hypnogram_file)
print(f"\nLoaded timeline length: {len(timeline)}")
print("First 20 timeline entries:")
for i, entry in enumerate(timeline[:20]):
    print(f"  {i}: start={entry['start']:.1f}s, end={entry['end']:.1f}s, label='{entry['label']}'")

# Statistics
wake_count = sum(1 for entry in timeline if 'W' in entry['label'] or 'wake' in entry['label'].lower())
sleep_count = sum(1 for entry in timeline if 'Sleep' in entry['label'])
total_wake_seconds = sum(entry['end'] - entry['start'] for entry in timeline if 'W' in entry['label'] or 'wake' in entry['label'].lower())
total_sleep_seconds = sum(entry['end'] - entry['start'] for entry in timeline if 'Sleep' in entry['label'])

print(f"\nStatistics:")
print(f"  Wake stages: {wake_count} entries, {total_wake_seconds:.1f}s")
print(f"  Sleep stages: {sleep_count} entries, {total_sleep_seconds:.1f}s")
print(f"  Total duration: {total_wake_seconds + total_sleep_seconds:.1f}s")
print(f"  Wake percentage: {100*total_wake_seconds/(total_wake_seconds + total_sleep_seconds):.1f}%")
