"""SleepEEGpy: A Python-based software integration package for sleep EEG analysis."""

from sleepeegpy.version import __version__
from sleepeegpy.preprocessing.cleaning import clean_sleep_eeg
from sleepeegpy.preprocessing.ica import apply_ica, find_artifact_components
from sleepeegpy.preprocessing.dashboard import generate_dashboard
from sleepeegpy.analysis.spectral import compute_psd, compute_group_psd, get_band_power
from sleepeegpy.analysis.events import detect_spindles, detect_slow_waves, detect_rem
from sleepeegpy.utils.io import load_eeg_data, save_processed_data, create_fif_from_data
from sleepeegpy.utils.hypnogram import load_hypnogram, predict_hypnogram, save_hypnogram, STAGE_NAMES

__all__ = [
    '__version__',
    'clean_sleep_eeg',
    'apply_ica',
    'find_artifact_components',
    'generate_dashboard',
    'compute_psd',
    'compute_group_psd',
    'get_band_power',
    'detect_spindles',
    'detect_slow_waves',
    'detect_rem',
    'load_eeg_data',
    'save_processed_data',
    'create_fif_from_data',
    'load_hypnogram',
    'predict_hypnogram',
    'save_hypnogram',
    'STAGE_NAMES',
]