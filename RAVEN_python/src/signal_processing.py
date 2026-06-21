"""
Signal processing utilities for RAVEN
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.fft import fft, fftfreq
import pywt
from sklearn.decomposition import NMF

class SignalProcessor:
    """Signal processing utilities for EEG analysis"""
    
    def __init__(self, fs):
        self.fs = fs
        self.frequency_bands = {
            'delta': [0.1, 4.5],
            'theta': [4, 8],
            'alpha': [8, 12],
            'sigma': [11, 16],
            'beta': [16, 30],
            'broadband': [0.1, 30]
        }
    
    def butterworth_bandpass(self, data, lowcut, highcut, order=2):
        """Apply Butterworth bandpass filter"""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def chebyshev_bandpass(self, data, lowcut, highcut, order=2, rp=0.5):
        """Apply Chebyshev Type I bandpass filter"""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.cheby1(order, rp, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def filter_to_band(self, data, band):
        """Filter signal to specific frequency band"""
        if isinstance(band, str):
            band_freq = self.frequency_bands.get(band, [0.1, 30])
        else:
            band_freq = band
        return self.butterworth_bandpass(data, band_freq[0], band_freq[1])
    
    def moving_window_power(self, data, window_size, overlap=0.5):
        """Calculate signal power using moving window"""
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))
        
        num_windows = max(1, (len(data) - window_samples) // step_samples + 1)
        power_values = np.zeros(num_windows)
        time_axis = np.zeros(num_windows)
        
        for i in range(num_windows):
            start_idx = i * step_samples
            end_idx = min(start_idx + window_samples, len(data))
            window_data = data[start_idx:end_idx]
            power_values[i] = np.mean(window_data ** 2)
            time_axis[i] = (start_idx + end_idx) / (2 * self.fs)
        
        return power_values, time_axis
    
    def moving_window_rms(self, data, window_size, overlap=0.5):
        """Calculate RMS using moving window"""
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))
        
        num_windows = max(1, (len(data) - window_samples) // step_samples + 1)
        rms_values = np.zeros(num_windows)
        time_axis = np.zeros(num_windows)
        
        for i in range(num_windows):
            start_idx = i * step_samples
            end_idx = min(start_idx + window_samples, len(data))
            window_data = data[start_idx:end_idx]
            rms_values[i] = np.sqrt(np.mean(window_data ** 2))
            time_axis[i] = (start_idx + end_idx) / (2 * self.fs)
        
        return rms_values, time_axis
    
    def hilbert_envelope(self, data):
        """Extract signal envelope using Hilbert transform"""
        analytic = hilbert(data)
        envelope = np.abs(analytic)
        return envelope, analytic
    
    def continuous_wavelet_transform(self, data, frequencies=None):
        """Compute Continuous Wavelet Transform"""
        if frequencies is None:
            frequencies = np.linspace(0.1, 30, 64)
        
        widths = self.fs / (2 * frequencies * np.pi)
        coeffs, freqs = pywt.cwt(data, widths, 'morl', 1/self.fs)
        return coeffs, freqs
    
    def empirical_mode_decomposition(self, data):
        """Perform Empirical Mode Decomposition"""
        # Simple implementation using scipy's EMD-like approach
        # For proper EMD, consider using PyEMD package
        from scipy.signal import find_peaks
        
        imfs = []
        residual = data.copy()
        
        # Simplified EMD - extract IMFs
        # In practice, use: from PyEMD import EMD
        # emd = EMD()
        # imfs = emd(data)
        
        # Fallback: return signal as single IMF
        imfs = [data]
        residual = np.zeros_like(data)
        
        return imfs, residual
    
    def detect_artifacts(self, data):
        """Detect artifacts using FFT (60-90 Hz band)"""
        nfft = 2 ** int(np.ceil(np.log2(len(data))))
        fft_vals = np.abs(fft(data, nfft))[:nfft//2 + 1]
        freqs = fftfreq(nfft, 1/self.fs)[:nfft//2 + 1]
        
        # Check 60-90 Hz band
        idx_60 = np.where(freqs >= 60)[0]
        idx_90 = np.where(freqs >= 90)[0]
        
        if len(idx_60) == 0 or len(idx_90) == 0:
            return 0
        
        start_idx = idx_60[0]
        end_idx = idx_90[0] if len(idx_90) > 0 else len(freqs)
        
        artifact_mag = np.mean(fft_vals[start_idx:end_idx])
        artifact_threshold = 100
        
        return artifact_mag > artifact_threshold
    
    def compute_short_long_ratios(self, data):
        """Compute short-long term ratios"""
        bands = ['delta', 'theta', 'alpha', 'sigma', 'beta']
        band_freqs = [[0.5, 4], [4, 8], [8, 12], [12, 16], [16, 30]]
        
        num_windows = max(1, len(data) // (2 * self.fs))
        ratios = np.zeros((len(bands), num_windows))
        
        for b, (band_name, band_freq) in enumerate(zip(bands, band_freqs)):
            # Filter signal
            filtered = self.butterworth_bandpass(data, band_freq[0], band_freq[1])
            
            # Apply EMD and get second IMF
            imfs, _ = self.empirical_mode_decomposition(filtered)
            if len(imfs) >= 2:
                second_imf = imfs[1]
            else:
                second_imf = filtered
            
            # Compute short and long means
            short_rms, short_time = self.moving_window_rms(second_imf, 2, 0.5)
            long_rms, long_time = self.moving_window_rms(second_imf, 60, 0.5)
            
            # Align time axes
            min_len = min(len(short_rms), len(long_rms))
            short_rms = short_rms[:min_len]
            long_rms = long_rms[:min_len]
            
            # Compute ratios
            ratios[b, :min_len] = short_rms / (long_rms + 1e-10)
        
        return ratios