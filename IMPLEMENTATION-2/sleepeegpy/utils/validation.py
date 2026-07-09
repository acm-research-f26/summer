"""Validation utilities for SleepEEGpy."""

import logging
from typing import Optional, Union

import numpy as np
import mne

logger = logging.getLogger(__name__)

def validate_eeg_data(raw: mne.io.Raw, require_preload: bool = True) -> bool:
    """
    Validate EEG data for processing.
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data
    require_preload : bool
        Whether data must be preloaded
        
    Returns
    -------
    bool
        True if valid
    """
    if raw is None:
        raise ValueError("Raw data is None")
    
    if raw.n_channels == 0:
        raise ValueError("No channels found in data")
    
    if raw.info['sfreq'] <= 0:
        raise ValueError("Invalid sampling rate")
    
    if require_preload and not raw.preload:
        logger.warning("Data is not preloaded. This may cause performance issues.")
    
    return True


def validate_hypnogram(hypnogram: np.ndarray, raw: mne.io.Raw, epoch_length: float = 30.0) -> bool:
    """
    Validate hypnogram compatibility with EEG data.
    
    Parameters
    ----------
    hypnogram : np.ndarray
        Hypnogram array
    raw : mne.io.Raw
        EEG data
    epoch_length : float
        Length of each epoch in seconds
        
    Returns
    -------
    bool
        True if valid
    """
    if hypnogram is None:
        raise ValueError("Hypnogram is None")
    
    if len(hypnogram) == 0:
        raise ValueError("Hypnogram is empty")
    
    # Check if hypnogram covers the full recording
    expected_epochs = int(np.ceil(raw.n_times / raw.info['sfreq'] / epoch_length))
    if len(hypnogram) < expected_epochs:
        logger.warning(f"Hypnogram ({len(hypnogram)} epochs) is shorter than "
                      f"expected ({expected_epochs} epochs). Data will be truncated.")
    elif len(hypnogram) > expected_epochs:
        logger.warning(f"Hypnogram ({len(hypnogram)} epochs) is longer than "
                      f"expected ({expected_epochs} epochs). Extra epochs will be ignored.")
    
    return True