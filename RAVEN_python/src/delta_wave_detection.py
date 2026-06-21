# File: src/delta_wave_detection.py
"""
Delta wave detection module for RAVEN
"""
import numpy as np
from scipy.signal import find_peaks
from signal_processing import SignalProcessor

class DeltaWaveDetector:
    """Delta wave sequence detection"""
    
    def __init__(self, fs, thresholds=None):
        self.fs = fs
        self.sp = SignalProcessor(fs)
        self.thresholds = thresholds or self._default_thresholds()
    
    def _default_thresholds(self):
        return {
            'power_delta': 0.4,
            'rms_delta': 15e-6,  # 15 μV
            'cwt_delta': 0.3,
            'hilbert_std_delta': 1.5,
            'artifact_delta': 100,
            'duration_min': 2.0,
            'wavelength_threshold': 0.5
        }
    
    def detect(self, signal_data):
        """Detect delta wave sequences"""
        events = []
        
        # Step 1: Filter to delta band
        delta_signal = self.sp.filter_to_band(signal_data, [0.1, 4.5])
        
        # Step 2: Calculate power with 2s windows
        power_values, power_time = self.sp.moving_window_power(delta_signal, 2, 0.5)
        
        # Step 3: Apply EMD and get second IMF
        imfs, _ = self.sp.empirical_mode_decomposition(delta_signal)
        second_imf = imfs[1] if len(imfs) >= 2 else delta_signal
        
        # Step 4: Calculate RMS
        rms_values, rms_time = self.sp.moving_window_rms(delta_signal, 1, 0.5)
        
        # Step 5: 30-second epochs
        epoch_len = 30
        epoch_overlap = 0.5
        step_samples = int(epoch_len * self.fs * (1 - epoch_overlap))
        epoch_samples = int(epoch_len * self.fs)
        
        for start_sample in range(0, len(signal_data) - epoch_samples, step_samples):
            end_sample = start_sample + epoch_samples
            
            # Extract epoch
            epoch_second_imf = second_imf[start_sample:end_sample]
            
            # Compute CWT
            cwt_coeff, freqs = self.sp.continuous_wavelet_transform(epoch_second_imf)
            
            # Delta band CWT
            delta_freq_mask = (freqs >= 0.1) & (freqs <= 4.5)
            cwt_mag = np.abs(cwt_coeff[delta_freq_mask, :])
            cwt_mask = cwt_mag > self.thresholds['cwt_delta'] * np.max(cwt_mag)
            
            # Find connected regions
            from scipy.ndimage import label
            labeled, num_features = label(cwt_mask)
            
            for i in range(1, num_features + 1):
                region = (labeled == i)
                idx = np.where(region.any(axis=0))[0]
                
                if len(idx) == 0:
                    continue
                
                start_idx = idx[0]
                end_idx = idx[-1]
                duration = (end_idx - start_idx) / self.fs
                
                if duration < self.thresholds['duration_min']:
                    continue
                
                # Evaluate candidate
                abs_start = start_sample + start_idx
                abs_end = start_sample + end_idx
                candidate_segment = delta_signal[abs_start:abs_end]
                raw_segment = signal_data[abs_start:abs_end]
                
                if self._evaluate_delta_wave(candidate_segment, raw_segment):
                    # Count slow waves
                    wave_count = self._count_slow_waves(candidate_segment)
                    if wave_count >= 3:
                        events.append({
                            'start': abs_start / self.fs,
                            'end': abs_end / self.fs,
                            'duration': duration,
                            'wave_count': wave_count,
                            'confidence': self._compute_confidence(candidate_segment)
                        })
        
        return self._consolidate_events(events)
    
    def _evaluate_delta_wave(self, delta_segment, raw_segment):
        """Evaluate delta wave candidate"""
        
        # Power threshold
        power_delta = np.mean(delta_segment ** 2)
        if power_delta < self.thresholds['power_delta']:
            return False
        
        # Hilbert envelope with at least 2 peaks
        envelope, _ = self.sp.hilbert_envelope(delta_segment)
        threshold = np.mean(envelope) + self.thresholds['hilbert_std_delta'] * np.std(envelope)
        peaks, _ = find_peaks(envelope, height=threshold)
        
        if len(peaks) < 2:
            return False
        
        # Check high-frequency band for artifacts
        hf_signal = self.sp.filter_to_band(raw_segment, [20, 30])
        hf_envelope, _ = self.sp.hilbert_envelope(hf_signal)
        hf_threshold = np.mean(hf_envelope) + 2 * np.std(hf_envelope)
        hf_peaks, _ = find_peaks(hf_envelope, height=hf_threshold)
        
        if len(hf_peaks) >= 2:
            return False
        
        # RMS threshold
        rms_delta = np.sqrt(np.mean(delta_segment ** 2))
        if rms_delta < self.thresholds['rms_delta']:
            return False
        
        # Artifact check
        artifact_score = self.sp.detect_artifacts(raw_segment)
        if artifact_score > self.thresholds['artifact_delta']:
            return False
        
        return True
    
    def _count_slow_waves(self, delta_segment):
        """Count number of slow waves"""
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(delta_segment)) != 0)[0]
        
        if len(zero_crossings) < 2:
            return 0
        
        # Calculate wavelengths
        wavelengths = np.diff(zero_crossings) / self.fs
        return np.sum(wavelengths > self.thresholds['wavelength_threshold'])
    
    def _compute_confidence(self, segment):
        """Compute delta wave confidence"""
        envelope, _ = self.sp.hilbert_envelope(segment)
        
        # Based on envelope regularity and amplitude
        envelope_regularity = 1 / (1 + np.std(envelope) / (np.mean(envelope) + 1e-10))
        amplitude_score = min(1.0, np.max(np.abs(segment)) / 100e-6)
        
        return 0.5 * envelope_regularity + 0.5 * amplitude_score
    
    def _consolidate_events(self, events):
        """Consolidate events within 2 seconds"""
        if not events:
            return events
        
        events = sorted(events, key=lambda x: x['start'])
        consolidated = []
        current = events[0].copy()
        
        for event in events[1:]:
            if event['start'] - current['end'] <= 2.0:
                current['end'] = max(current['end'], event['end'])
                current['duration'] = current['end'] - current['start']
                current['wave_count'] += event.get('wave_count', 0)
                current['confidence'] = max(current['confidence'], event['confidence'])
            else:
                consolidated.append(current)
                current = event.copy()
        
        consolidated.append(current)
        return consolidated