"""Spectral analysis module (B2)."""

import numpy as np
import mne
from typing import Dict, List, Optional, Tuple, Union
from sleepeegpy.config.defaults import SpectralConfig, DEFAULT_CONFIG
from sleepeegpy.utils.hypnogram import get_stage_intervals, STAGE_NAMES

def compute_psd(
    raw: mne.io.Raw,
    hypnogram: Optional[np.ndarray] = None,
    config: Optional[SpectralConfig] = None,
    picks: Optional[list] = None,
    fmin: float = 0.5,
    fmax: float = 50.0,
    verbose: bool = True
) -> Dict[int, mne.time_frequency.SpectrumArray]:
    """
    Compute power spectral density per sleep stage.
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data
    hypnogram : np.ndarray, optional
        Sleep scoring vector
    config : SpectralConfig, optional
        Configuration parameters
    picks : list, optional
        Channels to use
    fmin, fmax : float
        Frequency range
        
    Returns
    -------
    dict
        Dictionary mapping sleep stage to SpectrumArray
    """
    if config is None:
        config = DEFAULT_CONFIG['spectral']
    
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True)
    
    if verbose:
        print("Computing PSD per sleep stage...")
    
    stage_psds = {}
    
    # If no hypnogram, compute full PSD
    if hypnogram is None:
        psd, freqs = mne.time_frequency.psd_welch(
            raw, picks=picks, fmin=fmin, fmax=fmax,
            n_fft=config.fft_length,
            n_overlap=config.window_overlap,
            window=config.window_type,
            verbose=verbose
        )
        stage_psds['full'] = mne.time_frequency.SpectrumArray(
            psd, times=[], freqs=freqs, info=raw.info
        )
        return stage_psds
    
    # Compute PSD per sleep stage
    unique_stages = np.unique(hypnogram)
    
    for stage in unique_stages:
        if stage not in STAGE_NAMES:
            continue
        
        intervals = get_stage_intervals(hypnogram, stage=stage)
        if not intervals:
            continue
        
        stage_data = []
        stage_weights = []
        
        for start, end in intervals:
            raw_segment = raw.copy().crop(start, end)
            if raw_segment.n_times < raw.info['sfreq']:
                continue
            
            psd, freqs = mne.time_frequency.psd_welch(
                raw_segment, picks=picks,
                fmin=fmin, fmax=fmax,
                n_fft=config.fft_length,
                n_overlap=config.window_overlap,
                window=config.window_type,
                verbose=False
            )
            stage_data.append(psd)
            stage_weights.append(raw_segment.n_times / raw.info['sfreq'])
        
        if stage_data:
            weights = np.array(stage_weights) / np.sum(stage_weights)
            avg_psd = np.average(stage_data, axis=0, weights=weights)
            stage_psds[stage] = mne.time_frequency.SpectrumArray(
                avg_psd, times=[], freqs=freqs, info=raw.info
            )
            
            if verbose:
                stage_name = STAGE_NAMES[stage]
                duration = np.sum(stage_weights)
                print(f"  {stage_name}: {duration:.1f}s")
    
    return stage_psds

def compute_group_psd(
    stage_psds: List[Dict[int, mne.time_frequency.SpectrumArray]],
    stage_ids: Optional[List[int]] = None
) -> Dict[int, mne.time_frequency.SpectrumArray]:
    """
    Average PSDs across multiple recordings.
    
    Parameters
    ----------
    stage_psds : list
        List of stage PSD dictionaries from compute_psd
    stage_ids : list, optional
        Specific stages to include
        
    Returns
    -------
    dict
        Dictionary mapping sleep stage to averaged SpectrumArray
    """
    if stage_ids is None:
        all_stages = set()
        for psd_dict in stage_psds:
            all_stages.update(psd_dict.keys())
        stage_ids = list(all_stages - {'full'})
    
    group_psds = {}
    
    for stage in stage_ids:
        psd_list = []
        for psd_dict in stage_psds:
            if stage in psd_dict:
                psd_list.append(psd_dict[stage])
        
        if psd_list:
            avg_data = np.mean([psd.data for psd in psd_list], axis=0)
            first_psd = psd_list[0]
            group_psds[stage] = mne.time_frequency.SpectrumArray(
                avg_data, times=[], freqs=first_psd.freqs, info=first_psd.info
            )
            print(f"Stage {STAGE_NAMES[stage]}: averaged {len(psd_list)} recordings")
    
    return group_psds

def get_band_power(
    psd: Union[mne.time_frequency.SpectrumArray, np.ndarray],
    freqs: Optional[np.ndarray] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate power in frequency bands.
    
    Parameters
    ----------
    psd : SpectrumArray or np.ndarray
        PSD data
    freqs : np.ndarray, optional
        Frequency values (if psd is array)
    bands : dict, optional
        Dictionary of band name -> (fmin, fmax)
        
    Returns
    -------
    dict
        Band power values
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'sigma': (12, 15),
            'beta': (15, 30),
            'gamma': (30, 50),
        }
    
    # Extract data
    if isinstance(psd, mne.time_frequency.SpectrumArray):
        data = psd.data
        freqs = psd.freqs
    else:
        data = psd
    
    if data.ndim > 2:
        data = np.mean(data, axis=0)
    
    band_powers = {}
    
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(mask):
            band_power = np.mean(data[..., mask], axis=-1)
            band_powers[band_name] = band_power
    
    return band_powers