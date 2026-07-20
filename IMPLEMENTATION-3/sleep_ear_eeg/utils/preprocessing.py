# utils/preprocessing.py
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from config import SAMPLING_RATE, BANDS, EPOCH_SAMPLES

def band_power_features(epoch_data):
    """
    Calculates comprehensive features including band powers, ratios, and statistics.
    """
    # Remove DC offset
    epoch_data = epoch_data - np.mean(epoch_data)
    
    # Apply a bandpass filter (0.5-30 Hz)
    nyquist = SAMPLING_RATE / 2
    b, a = signal.butter(4, [0.5/nyquist, 30.0/nyquist], btype='band')
    
    try:
        filtered = signal.filtfilt(b, a, epoch_data)
    except:
        filtered = epoch_data
    
    features = {}
    
    # 1. Band powers
    for band, (low_freq, high_freq) in BANDS.items():
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        try:
            band_filtered = signal.filtfilt(b, a, filtered)
            features[f'{band}_power'] = np.var(band_filtered)
            features[f'{band}_mean'] = np.mean(np.abs(band_filtered))
        except:
            features[f'{band}_power'] = 0.0
            features[f'{band}_mean'] = 0.0
    
    # 2. Power ratios (important for sleep staging)
    total_power = features.get('total_power', np.var(filtered))
    if total_power > 0:
        for band in BANDS.keys():
            features[f'{band}_ratio'] = features.get(f'{band}_power', 0) / total_power
    else:
        for band in BANDS.keys():
            features[f'{band}_ratio'] = 0.0
    
    # 3. Statistical features
    features['mean'] = np.mean(filtered)
    features['std'] = np.std(filtered)
    features['skew'] = skew(filtered) if len(filtered) > 0 else 0
    features['kurtosis'] = kurtosis(filtered) if len(filtered) > 0 else 0
    features['rms'] = np.sqrt(np.mean(filtered**2))
    features['peak_to_peak'] = np.max(filtered) - np.min(filtered)
    
    # 4. Zero-crossing rate (indicator of frequency)
    zero_crossings = np.where(np.diff(np.sign(filtered)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(filtered) if len(filtered) > 0 else 0
    
    # 5. Hjorth parameters (activity, mobility, complexity)
    if len(filtered) > 0:
        # Activity (variance)
        features['hjorth_activity'] = np.var(filtered)
        
        # Mobility (sqrt of variance of first derivative / variance of signal)
        if features['hjorth_activity'] > 0:
            diff1 = np.diff(filtered)
            mobility = np.sqrt(np.var(diff1) / features['hjorth_activity'])
            features['hjorth_mobility'] = mobility
            
            # Complexity (mobility of first derivative / mobility)
            if mobility > 0:
                diff2 = np.diff(diff1)
                complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
                features['hjorth_complexity'] = complexity
            else:
                features['hjorth_complexity'] = 0
        else:
            features['hjorth_mobility'] = 0
            features['hjorth_complexity'] = 0
    
    # 6. Spectral edge frequency (SEF) - frequency below which 95% of power lies
    try:
        f, Pxx = signal.periodogram(filtered, fs=SAMPLING_RATE)
        cum_power = np.cumsum(Pxx)
        total_power = cum_power[-1]
        if total_power > 0:
            sef_idx = np.where(cum_power >= 0.95 * total_power)[0]
            if len(sef_idx) > 0:
                features['sef_95'] = f[sef_idx[0]]
            else:
                features['sef_95'] = 0
        else:
            features['sef_95'] = 0
    except:
        features['sef_95'] = 0
    
    return features

def segment_and_extract_features(raw, channel_name):
    """
    Segments the data into 30-second epochs and extracts comprehensive features.
    """
    # Get data from the specified channel
    data, times = raw[channel_name]
    data = data.flatten()
    
    # Calculate number of full epochs
    n_epochs = len(data) // EPOCH_SAMPLES
    
    if n_epochs == 0:
        print(f"Warning: Not enough data for a single epoch. Data length: {len(data)} samples")
        return []
    
    features = []
    
    for i in range(n_epochs):
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        epoch_data = data[start:end]
        
        # Check for NaN or invalid data
        if np.isnan(epoch_data).all() or np.isinf(epoch_data).all():
            # Return zeros for all features
            epoch_features = {}
            for band in BANDS.keys():
                epoch_features[f'{band}_power'] = 0.0
                epoch_features[f'{band}_mean'] = 0.0
                epoch_features[f'{band}_ratio'] = 0.0
            epoch_features.update({
                'mean': 0.0, 'std': 0.0, 'skew': 0.0, 'kurtosis': 0.0,
                'rms': 0.0, 'peak_to_peak': 0.0, 'zero_crossing_rate': 0.0,
                'hjorth_activity': 0.0, 'hjorth_mobility': 0.0, 
                'hjorth_complexity': 0.0, 'sef_95': 0.0
            })
        else:
            # Replace any remaining NaN values
            epoch_data = np.nan_to_num(epoch_data)
            epoch_features = band_power_features(epoch_data)
        
        features.append(epoch_features)
    
    return features