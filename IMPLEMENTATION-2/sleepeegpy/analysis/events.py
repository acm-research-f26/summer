"""Event detection and analysis module (B1)."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import mne

from sleepeegpy.config.defaults import SpindleConfig, SlowWaveConfig, REMConfig, DEFAULT_CONFIG
from sleepeegpy.utils.hypnogram import get_stage_intervals, STAGE_NAMES

logger = logging.getLogger(__name__)

def detect_spindles(
    raw: mne.io.Raw,
    hypnogram: Optional[np.ndarray] = None,
    stage: int = 2,  # N2
    config: Optional[SpindleConfig] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[mne.time_frequency.AverageTFR]]:
    """
    Detect sleep spindles using YASA algorithm.
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data
    hypnogram : np.ndarray, optional
        Sleep scoring vector
    stage : int
        Sleep stage for detection (default: N2)
    config : SpindleConfig, optional
        Configuration parameters
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    tuple (pd.DataFrame, mne.time_frequency.AverageTFR)
        Spindle properties and average TFR
    """
    try:
        import yasa
    except ImportError:
        print("YASA is required for event detection. Install with: pip install yasa")
        return pd.DataFrame(), None
    
    if config is None:
        config = DEFAULT_CONFIG['spindle']
    
    if verbose:
        print("Detecting sleep spindles...")
    
    # Use YASA spindle detection
    spindles = yasa.spindles(
        raw,
        hypno=hypnogram,
        include=stage,
        spindle_freq=config.freq_range,
        broadband_freq=config.broadband_range,
        duration=(config.duration_min, config.duration_max),
        min_distance=config.min_interval,
        thresh_relative=config.relative_power_thresh,
        thresh_corr=config.correlation_thresh,
        thresh_rms=config.rms_thresh,
        outlier_rejection=config.outlier_rejection,
        verbose=verbose
    )
    
    # Extract spindle properties
    properties = []
    for ch_name, spindle_data in spindles.items():
        for idx, row in spindle_data.iterrows():
            properties.append({
                'channel': ch_name,
                'start': row['Start'],
                'peak': row['Peak'],
                'end': row['End'],
                'duration': row['Duration'],
                'amplitude': row['Amplitude'],
                'frequency': row['Freq'],
                'power': row['AbsPower'],
                'relative_power': row['RelPower'],
                'rms': row['Rms'],
            })
    
    properties_df = pd.DataFrame(properties)
    
    if verbose:
        print(f"Detected {len(properties_df)} spindles across all channels")
    
    # Compute TFR if spindles were found
    tfr = None
    if len(properties_df) > 0:
        tfr = _compute_event_tfr(raw, properties_df, event='spindle')
    
    return properties_df, tfr

def detect_slow_waves(
    raw: mne.io.Raw,
    hypnogram: Optional[np.ndarray] = None,
    stage: int = 3,  # N3
    config: Optional[SlowWaveConfig] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[mne.time_frequency.AverageTFR]]:
    """
    Detect slow waves using YASA algorithm.
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data
    hypnogram : np.ndarray, optional
        Sleep scoring vector
    stage : int
        Sleep stage for detection (default: N3)
    config : SlowWaveConfig, optional
        Configuration parameters
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    tuple (pd.DataFrame, mne.time_frequency.AverageTFR)
        Slow wave properties and average TFR
    """
    try:
        import yasa
    except ImportError:
        print("YASA is required for event detection. Install with: pip install yasa")
        return pd.DataFrame(), None
    
    if config is None:
        config = DEFAULT_CONFIG['slow_wave']
    
    if verbose:
        print("Detecting slow waves...")
    
    # Use YASA slow wave detection
    sw = yasa.sw_detect(
        raw,
        hypno=hypnogram,
        include=stage,
        sw_freq=config.freq_range,
        sw_transition=config.transition_band,
        sw_neg_min=config.neg_peak_min,
        sw_neg_max=config.neg_peak_max,
        sw_pos_min=config.pos_peak_min,
        sw_pos_max=config.pos_peak_max,
        outlier_rejection=config.outlier_rejection,
        verbose=verbose
    )
    
    # Extract slow wave properties
    properties = []
    for ch_name, sw_data in sw.items():
        for idx, row in sw_data.iterrows():
            properties.append({
                'channel': ch_name,
                'start': row['Start'],
                'negative_peak': row['NegativePeak'],
                'positive_peak': row['PositivePeak'],
                'end': row['End'],
                'duration': row['Duration'],
                'neg_phase_duration': row['NegDuration'],
                'pos_phase_duration': row['PosDuration'],
                'amplitude': row['Amp'],
                'frequency': row['Freq'],
                'slope': row.get('Slope', np.nan),
            })
    
    properties_df = pd.DataFrame(properties)
    
    if verbose:
        print(f"Detected {len(properties_df)} slow waves across all channels")
    
    # Compute TFR
    tfr = None
    if len(properties_df) > 0:
        tfr = _compute_event_tfr(raw, properties_df, event='slow_wave')
    
    return properties_df, tfr

def detect_rem(
    raw: mne.io.Raw,
    hypnogram: Optional[np.ndarray] = None,
    stage: int = 4,  # REM
    config: Optional[REMConfig] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[mne.time_frequency.AverageTFR]]:
    """
    Detect rapid eye movements using YASA algorithm.
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data
    hypnogram : np.ndarray, optional
        Sleep scoring vector
    stage : int
        Sleep stage for detection (default: REM)
    config : REMConfig, optional
        Configuration parameters
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    tuple (pd.DataFrame, mne.time_frequency.AverageTFR)
        REM properties and average TFR
    """
    try:
        import yasa
    except ImportError:
        print("YASA is required for event detection. Install with: pip install yasa")
        return pd.DataFrame(), None
    
    if config is None:
        config = DEFAULT_CONFIG['rem']
    
    if verbose:
        print("Detecting REMs...")
    
    # Find EOG channels
    eog_channels = mne.pick_types(raw.info, eog=True)
    if len(eog_channels) < 2:
        # Try to find LOC and ROC by name
        loc_ch = next((ch for ch in raw.ch_names if 'LOC' in ch.upper()), None)
        roc_ch = next((ch for ch in raw.ch_names if 'ROC' in ch.upper()), None)
        if loc_ch is None or roc_ch is None:
            print("Warning: EOG channels LOC and ROC not found. Using first two available channels.")
            eog_channels = mne.pick_types(raw.info, eeg=True)[:2]
        else:
            eog_channels = [raw.ch_names.index(loc_ch), raw.ch_names.index(roc_ch)]
    
    # Use YASA REM detection
    rem = yasa.rem_detect(
        raw,
        eog_ch=eog_channels[0],
        eog_ch2=eog_channels[1] if len(eog_channels) > 1 else eog_channels[0],
        hypno=hypnogram,
        include=stage,
        rem_freq=config.freq_range,
        rem_amp_min=config.amplitude_min,
        rem_amp_max=config.amplitude_max,
        rem_duration=(config.duration_min, config.duration_max),
        outlier_rejection=config.outlier_rejection,
        verbose=verbose
    )
    
    # Extract REM properties
    properties = []
    for idx, row in rem.iterrows():
        properties.append({
            'start': row['Start'],
            'peak': row['Peak'],
            'end': row['End'],
            'duration': row['Duration'],
            'amplitude': row['Amplitude'],
            'velocity': row.get('Velocity', np.nan),
        })
    
    properties_df = pd.DataFrame(properties)
    
    if verbose:
        print(f"Detected {len(properties_df)} REMs")
    
    # Compute TFR
    tfr = None
    if len(properties_df) > 0:
        tfr = _compute_event_tfr(raw, properties_df, event='rem')
    
    return properties_df, tfr

def _compute_event_tfr(
    raw: mne.io.Raw,
    events_df: pd.DataFrame,
    event: str = 'spindle',
    time_window: Tuple[float, float] = (-1.0, 1.0),
    freq_range: Tuple[float, float] = (1, 30),
    n_cycles: int = 5,
    decim: int = 2
) -> Optional[mne.time_frequency.AverageTFR]:
    """
    Compute average time-frequency representation for events.
    """
    # Determine event center column
    center_col = 'peak'
    if event == 'slow_wave':
        center_col = 'negative_peak'
    elif event == 'rem':
        center_col = 'peak'
    
    if center_col not in events_df.columns:
        center_col = 'start'
    
    # Extract epochs around events
    sfreq = raw.info['sfreq']
    n_before = int(abs(time_window[0]) * sfreq)
    n_after = int(time_window[1] * sfreq)
    
    all_events_data = []
    
    # Process per channel
    for ch_name in events_df['channel'].unique() if 'channel' in events_df.columns else [raw.ch_names[0]]:
        if 'channel' in events_df.columns:
            ch_data = events_df[events_df['channel'] == ch_name]
        else:
            ch_data = events_df
            ch_name = raw.ch_names[0]
        
        ch_idx = raw.ch_names.index(ch_name)
        
        for idx, row in ch_data.iterrows():
            center_sample = int(row[center_col] * sfreq)
            
            # Extract window
            start_sample = max(0, center_sample - n_before)
            end_sample = min(raw.n_times, center_sample + n_after)
            
            # Get data
            data, times = raw[ch_idx, start_sample:end_sample]
            data = data.flatten()
            
            # Pad if necessary
            if len(data) < (n_before + n_after):
                data = np.pad(data, (0, (n_before + n_after) - len(data)))
            
            all_events_data.append(data)
    
    if not all_events_data:
        return None
    
    # Average across events
    avg_data = np.mean(all_events_data, axis=0)
    
    # Compute TFR using Morlet wavelets
    from mne.time_frequency import tfr_morlet
    
    # Create temporary RawArray for TFR computation
    info = mne.create_info([raw.ch_names[0]], sfreq, ch_types='eeg')
    temp_raw = mne.io.RawArray(avg_data.reshape(1, -1), info)
    
    tfr = tfr_morlet(
        temp_raw,
        freqs=np.arange(freq_range[0], freq_range[1], 0.5),
        n_cycles=n_cycles,
        time_bandwidth=0.5 * n_cycles,
        use_fft=True,
        decim=decim,
        return_itc=False
    )
    
    return tfr