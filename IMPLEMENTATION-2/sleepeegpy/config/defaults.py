"""Default configuration parameters for SleepEEGpy."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class CleaningConfig:
    """Default parameters for cleaning (A1)."""
    resample_freq: int = 250
    high_pass: float = 0.3
    low_pass: Optional[float] = None
    notch_freq: Optional[float] = 50.0
    notch_width: float = 1.0
    reference: str = 'average'
    auto_bad_channels: bool = False
    auto_bad_epochs: bool = False
    peak_to_peak_threshold: float = 500e-6
    channel_interpolation_method: str = 'spherical_spline'
    
@dataclass
class ICAConfig:
    """Default parameters for ICA (A2)."""
    algorithm: str = 'fastica'
    n_components: int = 30
    high_pass_ica: float = 1.0
    random_state: Optional[int] = 42
    
@dataclass
class SpindleConfig:
    """Default parameters for spindle detection."""
    freq_range: tuple = (12, 15)
    broadband_range: tuple = (1, 30)
    duration_min: float = 0.5
    duration_max: float = 2.0
    min_interval: float = 0.5
    relative_power_thresh: float = 0.2
    correlation_thresh: float = 0.65
    rms_thresh: float = 1.5
    outlier_rejection: bool = True
    
@dataclass
class SlowWaveConfig:
    """Default parameters for slow wave detection."""
    freq_range: tuple = (0.3, 1.5)
    transition_band: float = 0.2
    neg_peak_min: float = -200e-6
    neg_peak_max: float = -40e-6
    pos_peak_min: float = 10e-6
    pos_peak_max: float = 150e-6
    outlier_rejection: bool = True
    
@dataclass
class REMConfig:
    """Default parameters for REM detection."""
    freq_range: tuple = (0.5, 5)
    amplitude_min: float = 50e-6
    amplitude_max: float = 325e-6
    duration_min: float = 0.3
    duration_max: float = 1.2
    outlier_rejection: bool = True
    
@dataclass
class SpectralConfig:
    """Default parameters for spectral analysis."""
    fft_length: int = 256
    window_overlap: int = 0
    window_type: str = 'hamming'
    multitaper_bandwidth: Optional[float] = None

DEFAULT_CONFIG = {
    'cleaning': CleaningConfig(),
    'ica': ICAConfig(),
    'spindle': SpindleConfig(),
    'slow_wave': SlowWaveConfig(),
    'rem': REMConfig(),
    'spectral': SpectralConfig(),
}