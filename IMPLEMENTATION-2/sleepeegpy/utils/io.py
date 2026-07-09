"""Input/output utilities for .fif files only."""

from pathlib import Path
from typing import Union
import mne
import numpy as np

def load_eeg_data(filepath: Union[str, Path], preload: bool = True) -> mne.io.Raw:
    """
    Load EEG data from .fif or .fif.gz files only.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .fif or .fif.gz file
    preload : bool
        Whether to preload data into memory
        
    Returns
    -------
    mne.io.Raw
        Loaded EEG data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Only allow .fif and .fif.gz files
    suffix = filepath.suffix.lower()
    if suffix == '.gz':
        # Check if it's .fif.gz
        if filepath.suffixes[-2].lower() != '.fif':
            raise ValueError(f"Only .fif and .fif.gz files are supported. Got: {filepath}")
    elif suffix not in ['.fif']:
        raise ValueError(f"Only .fif and .fif.gz files are supported. Got: {suffix}")
    
    try:
        raw = mne.io.read_raw_fif(filepath, preload=preload)
        print(f"Loaded EEG data from {filepath}")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.2f} s")
        return raw
    except Exception as e:
        raise RuntimeError(f"Error loading FIF file: {e}")

def save_processed_data(
    raw: mne.io.Raw,
    filepath: Union[str, Path],
    overwrite: bool = False
) -> None:
    """
    Save processed EEG data as .fif file.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Processed EEG data
    filepath : str or Path
        Output file path (will save as .fif)
    overwrite : bool
        Whether to overwrite existing file
    """
    filepath = Path(filepath)
    
    # Ensure .fif extension
    if filepath.suffix.lower() not in ['.fif', '.fif.gz']:
        filepath = filepath.with_suffix('.fif')
    
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {filepath}")
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    raw.save(filepath, overwrite=overwrite)
    print(f"Saved processed data to {filepath}")

def create_fif_from_data(
    data: np.ndarray,
    sfreq: float,
    ch_names: list,
    ch_types: list = None,
    montage_name: str = 'standard_1020'
) -> mne.io.Raw:
    """
    Create a FIF file from numpy array data.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    ch_names : list
        List of channel names
    ch_types : list, optional
        List of channel types
    montage_name : str
        Montage name for channel positions
        
    Returns
    -------
    mne.io.Raw
        Raw object ready for saving as .fif
    """
    if ch_types is None:
        ch_types = ['eeg'] * len(ch_names)
    
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info)
    
    # Add montage if available
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage)
    except:
        print(f"Warning: Could not set montage {montage_name}")
    
    return raw