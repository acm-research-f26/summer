#!/usr/bin/env python
"""
Process EEG data without a hypnogram using automatic sleep staging.
"""

import sys
from pathlib import Path

from sleepeegpy import (
    load_eeg_data, 
    predict_hypnogram,
    clean_sleep_eeg, 
    generate_dashboard,
    save_processed_data,
    save_hypnogram,
    STAGE_NAMES,
    compute_psd,
    detect_spindles
)

def process_without_hypnogram(fif_path, output_dir='output'):
    """
    Process sleep EEG data without a hypnogram.
    
    Parameters
    ----------
    fif_path : str
        Path to FIF file
    output_dir : str
        Directory for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Processing Sleep EEG Data Without Hypnogram")
    print("=" * 60)
    
    # Step 1: Load data
    print(f"\n1. Loading data from: {fif_path}")
    raw = load_eeg_data(fif_path)
    print(f"   Loaded: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz, "
          f"{raw.times[-1]/60:.1f} minutes")
    
    # Step 2: Automatic sleep staging
    print("\n2. Performing automatic sleep staging...")
    hypnogram = predict_hypnogram(raw, epoch_length=30, method='yasa', verbose=True)
    
    # Save predicted hypnogram
    hypno_path = output_dir / 'predicted_hypnogram.txt'
    save_hypnogram(hypnogram, hypno_path)
    print(f"   Saved predicted hypnogram to: {hypno_path}")
    
    # Step 3: Clean the data
    print("\n3. Cleaning EEG data...")
    cleaned_raw, preproc_info = clean_sleep_eeg(
        raw, 
        hypnogram=hypnogram,
        auto_bad_channels=False,
        auto_bad_epochs=False,
        verbose=True
    )
    
    # Step 4: Save cleaned data
    print("\n4. Saving cleaned data...")
    cleaned_path = output_dir / 'cleaned_data.fif'
    save_processed_data(cleaned_raw, cleaned_path, overwrite=True)
    print(f"   Saved to: {cleaned_path}")
    
    # Step 5: Generate dashboard
    print("\n5. Generating dashboard...")
    dashboard_path = output_dir / 'dashboard.png'
    generate_dashboard(
        cleaned_raw, 
        hypnogram=hypnogram,
        epoch_length=30,
        save_path=str(dashboard_path)
    )
    print(f"   Dashboard saved to: {dashboard_path}")
    
    # Step 6: Compute spectral analysis
    print("\n6. Computing spectral analysis...")
    stage_psds = compute_psd(cleaned_raw, hypnogram=hypnogram, verbose=True)
    
    # Step 7: Detect sleep spindles (if YASA is available)
    print("\n7. Detecting sleep spindles...")
    try:
        spindles_df, tfr = detect_spindles(cleaned_raw, hypnogram=hypnogram, verbose=True)
        if len(spindles_df) > 0:
            spindle_path = output_dir / 'spindles.csv'
            spindles_df.to_csv(spindle_path, index=False)
            print(f"   Spindles saved to: {spindle_path}")
            print(f"   Found {len(spindles_df)} spindles")
    except Exception as e:
        print(f"   Spindle detection skipped: {e}")
    
    # Step 8: Summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    print("\nOutput Files:")
    print(f"  - Predicted hypnogram: {hypno_path}")
    print(f"  - Cleaned data: {cleaned_path}")
    print(f"  - Dashboard: {dashboard_path}")
    
    print("\nSleep Stage Durations:")
    epoch_length = 30
    for stage_id, stage_name in STAGE_NAMES.items():
        duration = np.sum(hypnogram == stage_id) * epoch_length / 60
        if duration > 0:
            percentage = (duration / (len(hypnogram) * epoch_length / 60)) * 100
            print(f"  {stage_name}: {duration:.1f} minutes ({percentage:.1f}%)")
    
    print(f"\nPreprocessing Info:")
    print(f"  Original sampling rate: {preproc_info['original_sfreq']} Hz")
    print(f"  Bad channels: {len(preproc_info.get('bad_channels', []))}")
    print(f"  Interpolated channels: {len(preproc_info.get('interpolated_channels', []))}")
    print(f"  Bad intervals: {len(preproc_info.get('bad_intervals', []))}")

if __name__ == "__main__":
    # Check if file path provided
    if len(sys.argv) > 1:
        fif_path = sys.argv[1]
    else:
        fif_path = '/Users/zoebryant/Documents/GitHub/summer/IMPLEMENTATION-2/sleepeegpy/data/young_adult.fif'  # Default
    
    process_without_hypnogram(fif_path)