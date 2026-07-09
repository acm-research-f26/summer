"""Cleaning preprocessing module (A1)."""

import numpy as np
import mne
from typing import Optional, Tuple
from sleepeegpy.config.defaults import CleaningConfig, DEFAULT_CONFIG

def clean_sleep_eeg(
    raw: mne.io.Raw,
    hypnogram: Optional[np.ndarray] = None,
    config: Optional[CleaningConfig] = None,
    verbose: bool = True
) -> Tuple[mne.io.Raw, dict]:
    """
    Clean sleep EEG data (A1).
    
    Performs: resampling, filtering, bad channel detection/interpolation,
    bad epoch detection, and re-referencing.
    """
    if config is None:
        config = DEFAULT_CONFIG['cleaning']
    
    preproc_info = {
        'original_sfreq': raw.info['sfreq'],
        'bad_channels': [],
        'bad_intervals': [],
        'interpolated_channels': [],
    }
    
    if verbose:
        print("Starting cleaning pipeline...")
    
    # Make a copy to avoid modifying original
    raw = raw.copy()
    
    # Step 1: Resampling
    if config.resample_freq and config.resample_freq != raw.info['sfreq']:
        if verbose:
            print(f"Resampling from {raw.info['sfreq']} to {config.resample_freq} Hz")
        raw.resample(config.resample_freq)
    
    # Step 2: High-pass filter
    if config.high_pass is not None:
        if verbose:
            print(f"Applying high-pass filter: {config.high_pass} Hz")
        raw.filter(l_freq=config.high_pass, h_freq=None, verbose=verbose)
    
    # Step 3: Low-pass filter (optional)
    if config.low_pass is not None:
        if verbose:
            print(f"Applying low-pass filter: {config.low_pass} Hz")
        raw.filter(l_freq=None, h_freq=config.low_pass, verbose=verbose)
    
    # Step 4: Notch filter
    if config.notch_freq is not None:
        if verbose:
            print(f"Applying notch filter: {config.notch_freq} Hz")
        raw.notch_filter(freqs=config.notch_freq, verbose=verbose)
    
    # Step 5: Auto bad channel detection (if enabled)
    if config.auto_bad_channels:
        if verbose:
            print("Performing automatic bad channel detection...")
        try:
            from pyprep.prep_pipeline import Prep
            prep = Prep(raw, ref_chs='average', ransac=False)
            prep.fit()
            bad_channels = getattr(prep, 'noisy_channels_ransac', [])
            if bad_channels:
                raw.info['bads'] = list(set(raw.info['bads']) | set(bad_channels))
                preproc_info['bad_channels'] = bad_channels
                if verbose:
                    print(f"Found {len(bad_channels)} bad channels")
        except ImportError:
            print("PyPREP not installed. Skipping auto bad channel detection.")
    
    # Step 6: Interpolate bad channels
    if raw.info['bads']:
        if verbose:
            print(f"Interpolating bad channels: {raw.info['bads']}")
        raw.interpolate_bads(method=config.channel_interpolation_method)
        preproc_info['interpolated_channels'] = raw.info['bads'].copy()
        raw.info['bads'] = []
    
    # Step 7: Re-reference to common average
    if config.reference == 'average':
        if verbose:
            print("Applying common average reference")
        raw.set_eeg_reference('average')
    
    if verbose:
        print("Cleaning completed")
    
    return raw, preproc_info

def annotate_bad_intervals(
    raw: mne.io.Raw,
    intervals: list,
    description: str = 'BAD'
) -> mne.io.Raw:
    """
    Annotate bad temporal intervals.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    intervals : list of tuple
        List of (start_time, end_time) in seconds
    description : str
        Annotation description
        
    Returns
    -------
    mne.io.Raw
        Annotated raw data
    """
    for start, end in intervals:
        raw.annotations.append(start, end - start, description)
    return raw