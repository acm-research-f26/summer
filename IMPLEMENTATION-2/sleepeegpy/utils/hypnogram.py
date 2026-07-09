"""Hypnogram utilities for SleepEEGpy."""

import numpy as np
import mne
from typing import Optional, Union
from pathlib import Path

STAGE_NAMES = {0: 'wake', 1: 'n1', 2: 'n2', 3: 'n3', 4: 'rem'}
STAGE_ORDER = ['wake', 'n1', 'n2', 'n3', 'rem']

def load_hypnogram(filepath: Union[str, Path], epoch_length: float = 30.0) -> np.ndarray:
    """Load hypnogram from text file (one integer per row)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Hypnogram file not found: {filepath}")
    hypnogram = np.loadtxt(filepath, dtype=int)
    return hypnogram.flatten()

def predict_hypnogram(
    raw: mne.io.Raw, 
    epoch_length: float = 30.0,
    method: str = 'yasa',
    verbose: bool = True
) -> np.ndarray:
    """
    Automatically predict sleep stages from EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data
    epoch_length : float
        Length of each epoch in seconds
    method : str
        Method for sleep staging ('yasa' or 'mne')
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    np.ndarray
        Predicted hypnogram
    """
    if method == 'yasa':
        try:
            import yasa
            if verbose:
                print("Using YASA for automatic sleep staging...")
            
            # YASA requires data to be preloaded
            if not raw.preload:
                raw.load_data()
            
            # Predict sleep stages
            hypnogram = yasa.sleep_staging(raw, epoch_length=epoch_length, verbose=verbose)
            
            if verbose:
                stage_counts = {STAGE_NAMES.get(s, 'unknown'): np.sum(hypnogram == s) 
                              for s in np.unique(hypnogram)}
                print(f"Predicted sleep stages: {stage_counts}")
            
            return hypnogram
            
        except ImportError:
            print("YASA not installed. Falling back to MNE method.")
            return _predict_hypnogram_mne(raw, epoch_length, verbose)
    
    else:
        return _predict_hypnogram_mne(raw, epoch_length, verbose)

def _predict_hypnogram_mne(raw: mne.io.Raw, epoch_length: float = 30.0, verbose: bool = True) -> np.ndarray:
    """
    Simple sleep staging using MNE (fallback method).
    
    This is a simplified version that uses spectral features to estimate sleep stages.
    For better results, install YASA.
    """
    if verbose:
        print("Using MNE-based sleep staging (simplified)...")
    
    # Ensure data is preloaded
    if not raw.preload:
        raw.load_data()
    
    # Calculate duration and epochs
    duration = raw.n_times / raw.info['sfreq']
    n_epochs = int(np.ceil(duration / epoch_length))
    
    # Initialize hypnogram
    hypnogram = np.zeros(n_epochs, dtype=int)
    
    # Get EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        raise ValueError("No EEG channels found in data")
    
    # For each epoch, compute spectral features
    for epoch_idx in range(n_epochs):
        start_time = epoch_idx * epoch_length
        end_time = min((epoch_idx + 1) * epoch_length, duration)
        
        try:
            # Extract epoch
            start_sample = int(start_time * raw.info['sfreq'])
            end_sample = int(end_time * raw.info['sfreq'])
            epoch_data, times = raw[:, start_sample:end_sample]
            
            if epoch_data.shape[1] < raw.info['sfreq'] * 2:  # Need at least 2 seconds
                hypnogram[epoch_idx] = 0  # Wake
                continue
            
            # Compute PSD
            psd, freqs = mne.time_frequency.psd_welch(
                mne.io.RawArray(epoch_data, raw.info),
                fmin=0.5, fmax=30, verbose=False
            )
            
            # Average across channels
            psd_avg = np.mean(psd, axis=0)
            
            # Calculate band powers
            delta_power = np.mean(psd_avg[(freqs >= 0.5) & (freqs < 4)])
            theta_power = np.mean(psd_avg[(freqs >= 4) & (freqs < 8)])
            alpha_power = np.mean(psd_avg[(freqs >= 8) & (freqs < 12)])
            sigma_power = np.mean(psd_avg[(freqs >= 12) & (freqs < 15)])
            beta_power = np.mean(psd_avg[(freqs >= 15) & (freqs < 30)])
            
            total_power = delta_power + theta_power + alpha_power + sigma_power + beta_power
            
            if total_power == 0:
                hypnogram[epoch_idx] = 0
                continue
            
            # Simple heuristic for sleep staging
            # Check for wake (high alpha, high beta, low delta)
            if alpha_power / total_power > 0.3 and beta_power / total_power > 0.1:
                hypnogram[epoch_idx] = 0  # Wake
            # Check for REM (high theta, low delta, low alpha)
            elif theta_power / total_power > 0.25 and delta_power / total_power < 0.3:
                hypnogram[epoch_idx] = 4  # REM
            # Check for deep sleep (high delta)
            elif delta_power / total_power > 0.4:
                hypnogram[epoch_idx] = 3  # N3
            # Check for light sleep (moderate delta, some sigma)
            elif sigma_power / total_power > 0.1:
                hypnogram[epoch_idx] = 2  # N2
            else:
                hypnogram[epoch_idx] = 1  # N1
            
        except Exception as e:
            if verbose:
                print(f"Warning: Error processing epoch {epoch_idx}: {e}")
            hypnogram[epoch_idx] = 0  # Default to wake
    
    if verbose:
        stage_counts = {STAGE_NAMES.get(s, 'unknown'): np.sum(hypnogram == s) 
                       for s in np.unique(hypnogram)}
        print(f"Predicted sleep stages: {stage_counts}")
    
    return hypnogram

def get_stage_intervals(hypnogram: np.ndarray, stage: int, epoch_length: float = 30.0) -> list:
    """Get time intervals for a specific sleep stage."""
    mask = hypnogram == stage
    intervals = []
    start = None
    for i, is_selected in enumerate(mask):
        if is_selected and start is None:
            start = i * epoch_length
        elif not is_selected and start is not None:
            intervals.append((start, i * epoch_length))
            start = None
    if start is not None:
        intervals.append((start, len(hypnogram) * epoch_length))
    return intervals

def save_hypnogram(hypnogram: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save hypnogram to text file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, hypnogram, fmt='%d')