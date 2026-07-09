#!/usr/bin/env python
"""
Complete SleepEEGpy pipeline demonstration using .fif files only.
Creates synthetic data in FIF format and runs the full preprocessing and analysis.
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# Import SleepEEGpy
from sleepeegpy import (
    clean_sleep_eeg, generate_dashboard, compute_psd,
    load_eeg_data, save_processed_data, create_fif_from_data,
    load_hypnogram, save_hypnogram, STAGE_NAMES
)

def create_synthetic_fif_data(duration_minutes=10, n_channels=20, sfreq=250):
    """
    Create synthetic sleep EEG data and save as .fif file.
    
    Returns:
        raw: mne.io.Raw object with synthetic EEG data
        hypnogram: np.array with sleep stages
        fif_path: Path to saved .fif file
    """
    duration = duration_minutes * 60
    t = np.arange(0, duration, 1/sfreq)
    
    # Create channel names (simulating a 20-channel montage)
    ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
    
    # Add EOG channels for REM detection
    ch_names.append('LOC')
    ch_names.append('ROC')
    
    # Channel types
    ch_types = ['eeg'] * n_channels + ['eog', 'eog']
    
    # Generate data for each channel
    data = []
    for i, name in enumerate(ch_names):
        if name in ['LOC', 'ROC']:
            # EOG channels - simulate eye movements
            signal = 20 * np.random.randn(len(t))
            # Add occasional eye movements
            for j in range(10):
                pos = int(30 + 120 * j * np.random.rand())
                if pos + 100 < len(t):
                    signal[pos:pos+100] += 50 * np.random.randn(100)
            data.append(signal)
            continue
        
        signal = np.zeros_like(t)
        
        # Add noise
        signal += 5 * np.random.randn(len(t))
        
        # Add specific rhythms based on channel position
        if i < 5:  # Frontal
            signal += 25 * np.sin(2 * np.pi * 1.5 * t + np.random.rand())
        elif i < 10:  # Central
            signal += 10 * np.sin(2 * np.pi * 13 * t + np.random.rand())
        elif i < 15:  # Parietal
            signal += 10 * np.sin(2 * np.pi * 8 * t + np.random.rand())
        else:  # Occipital
            signal += 15 * np.sin(2 * np.pi * 10 * t + np.random.rand())
        
        # Add spindles (randomly distributed)
        if np.random.rand() > 0.7:
            spindle_start = 30 + 120 * np.random.rand()
            spindle_duration = 0.5 + 0.5 * np.random.rand()
            spindle_idx = int(spindle_start * sfreq)
            spindle_len = int(spindle_duration * sfreq)
            if spindle_idx + spindle_len < len(t):
                window = np.hanning(spindle_len)
                signal[spindle_idx:spindle_idx+spindle_len] += (
                    30 * window * np.sin(2 * np.pi * (13 + 2*np.random.rand()) * 
                                        t[spindle_idx:spindle_idx+spindle_len])
                )
        
        data.append(signal)
    
    data = np.array(data)
    
    # Create FIF format using our utility
    raw = create_fif_from_data(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        ch_types=ch_types,
        montage_name='standard_1020'
    )
    
    # Create hypnogram (30-second epochs)
    epoch_length = 30
    n_epochs = int(duration / epoch_length)
    hypnogram = np.zeros(n_epochs, dtype=int)
    
    # Simulate sleep stages: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    for i in range(n_epochs):
        if i < 2:
            hypnogram[i] = 0
        elif i < 4:
            hypnogram[i] = 1
        elif i < 8:
            hypnogram[i] = 2
        elif i < 12:
            hypnogram[i] = 3
        elif i < 14:
            hypnogram[i] = 2
        elif i < 16:
            hypnogram[i] = 4
        else:
            hypnogram[i] = np.random.choice([2, 3, 4, 0], p=[0.3, 0.2, 0.2, 0.3])
    
    # Save as FIF file
    fif_path = Path('synthetic_sleep_data.fif')
    raw.save(fif_path, overwrite=True)
    print(f"Created synthetic FIF file: {fif_path}")
    
    # Save hypnogram
    hypno_path = Path('synthetic_hypnogram.txt')
    save_hypnogram(hypnogram, hypno_path)
    
    return raw, hypnogram, fif_path, hypno_path

def main():
    """Run the complete SleepEEGpy pipeline."""
    print("=" * 60)
    print("SleepEEGpy Pipeline Demonstration (.fif files only)")
    print("=" * 60)
    
    # Step 1: Create synthetic FIF data
    print("\n1. Creating synthetic sleep data in FIF format...")
    raw, hypnogram, fif_path, hypno_path = create_synthetic_fif_data(
        duration_minutes=10, n_channels=20, sfreq=250
    )
    print(f"   Created data with {len(raw.ch_names)} channels, "
          f"{raw.info['sfreq']} Hz, {raw.times[-1]/60:.1f} minutes")
    print(f"   Hypnogram: {len(hypnogram)} epochs, stages: {np.unique(hypnogram)}")
    
    # Step 2: Load from FIF file
    print("\n2. Loading data from FIF file...")
    loaded_raw = load_eeg_data(fif_path)
    loaded_hypno = load_hypnogram(hypno_path)
    print(f"   Loaded {len(loaded_raw.ch_names)} channels from FIF file")
    
    # Step 3: Clean the data
    print("\n3. Cleaning EEG data...")
    cleaned_raw, preproc_info = clean_sleep_eeg(
        loaded_raw, 
        hypnogram=loaded_hypno,
        auto_bad_channels=False,
        auto_bad_epochs=False,
        verbose=True
    )
    
    # Step 4: Save cleaned data as FIF
    print("\n4. Saving cleaned data as FIF...")
    cleaned_path = Path('cleaned_sleep_data.fif')
    save_processed_data(cleaned_raw, cleaned_path, overwrite=True)
    
    # Step 5: Generate dashboard
    print("\n5. Generating dashboard...")
    fig = generate_dashboard(
        cleaned_raw, 
        hypnogram=loaded_hypno,
        epoch_length=30,
        save_path='dashboard.png'
    )
    print("   Dashboard saved as 'dashboard.png'")
    
    # Step 6: Compute spectral analysis
    print("\n6. Computing spectral analysis...")
    stage_psds = compute_psd(cleaned_raw, hypnogram=loaded_hypno, verbose=True)
    print(f"   Computed PSD for {len(stage_psds)} sleep stages")
    
    # Step 7: Display summary of results
    print("\n7. Results summary:")
    print("-" * 40)
    print(f"Preprocessing info:")
    print(f"  Original sampling rate: {preproc_info['original_sfreq']} Hz")
    print(f"  Bad channels: {len(preproc_info.get('bad_channels', []))}")
    print(f"  Bad intervals: {len(preproc_info.get('bad_intervals', []))}")
    
    print("\nSleep stage durations:")
    for stage_id, stage_name in STAGE_NAMES.items():
        duration = np.sum(hypnogram == stage_id) * 30 / 60
        if duration > 0:
            print(f"  {stage_name}: {duration:.1f} minutes")
    
    # Step 8: Export cleaned data in various formats
    print("\n8. Exporting cleaned data...")
    
    # Save as FIF (already done)
    print(f"  FIF: {cleaned_path}")
    
    # Export as EDF if needed
    edf_path = Path('cleaned_sleep_data.edf')
    try:
        cleaned_raw.export(edf_path, overwrite=True)
        print(f"  EDF: {edf_path}")
    except:
        print("  EDF export not available")
    
    # Export channel locations
    if cleaned_raw.info.get('dig'):
        locations_path = Path('channel_locations.txt')
        with open(locations_path, 'w') as f:
            f.write("Channel\tX\tY\tZ\n")
            for ch in cleaned_raw.info['dig']:
                if ch['kind'] == 1:  # EEG channel
                    f.write(f"{ch['ident']}\t{ch['loc'][0]}\t{ch['loc'][1]}\t{ch['loc'][2]}\n")
        print(f"  Channel locations: {locations_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete! Files created:")
    print(f"  - {fif_path} (original data)")
    print(f"  - {hypno_path} (hypnogram)")
    print(f"  - {cleaned_path} (cleaned data)")
    print(f"  - dashboard.png (visualization)")
    print("=" * 60)

if __name__ == "__main__":
    main()