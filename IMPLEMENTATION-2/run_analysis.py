#!/usr/bin/env python
"""
Simple script to analyze sleep EEG data without a hypnogram.
"""

from sleepeegpy import (
    load_eeg_data, 
    predict_hypnogram,
    clean_sleep_eeg, 
    generate_dashboard,
    save_processed_data,
    save_hypnogram,
    STAGE_NAMES
)

# Load your data
print("Loading EEG data...")
raw = load_eeg_data('/Users/zoebryant/Documents/GitHub/summer/IMPLEMENTATION-2/data/young_adult.fif')

# Automatically predict sleep stages
print("\nPredicting sleep stages...")
hypnogram = predict_hypnogram(raw, epoch_length=30, verbose=True)

# Save predicted hypnogram
save_hypnogram(hypnogram, 'predicted_hypnogram.txt')

# Clean the data
print("\nCleaning data...")
cleaned_raw, info = clean_sleep_eeg(raw, hypnogram, verbose=True)

# Save cleaned data
save_processed_data(cleaned_raw, 'cleaned_data.fif', overwrite=True)

# Generate dashboard
print("\nGenerating dashboard...")
dashboard = generate_dashboard(
    cleaned_raw, 
    hypnogram=hypnogram,
    save_path='dashboard.png'
)

# Print summary
print("\n" + "=" * 50)
print("Sleep Stage Summary:")
epoch_length = 30
total_minutes = len(hypnogram) * epoch_length / 60
for stage_id, stage_name in STAGE_NAMES.items():
    duration = np.sum(hypnogram == stage_id) * epoch_length / 60
    if duration > 0:
        percentage = (duration / total_minutes) * 100
        print(f"  {stage_name}: {duration:.1f} min ({percentage:.1f}%)")

print(f"\nOutput files created:")
print(f"  - predicted_hypnogram.txt")
print(f"  - cleaned_data.fif")
print(f"  - dashboard.png")