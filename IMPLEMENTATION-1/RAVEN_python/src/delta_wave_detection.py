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
            'envelope_std_scale': 0.7,
            'rms_delta': 12.0,
            'artifact_delta': 100,
            'duration_min': 2.0,
            'duration_max': 20.0,
            'peak_to_peak_min': 80.0,
            'wavelength_min': 0.5,
            'wavelength_max': 2.0,
            'wave_count_min': 2
        }
    
    def detect(self, signal_data):
        """Detect delta wave sequences"""
        events = []
        delta_signal = self.sp.filter_to_band(signal_data, [0.1, 4.5])
        envelope, _ = self.sp.hilbert_envelope(delta_signal)
        threshold = np.mean(envelope) + self.thresholds['envelope_std_scale'] * np.std(envelope)
        mask = envelope > threshold

        from scipy.ndimage import label
        labeled, num_features = label(mask)

        for i in range(1, num_features + 1):
            region = (labeled == i)
            idx = np.where(region)[0]
            if len(idx) == 0:
                continue

            start_idx = idx[0]
            end_idx = idx[-1]
            duration = (end_idx - start_idx) / self.fs
            if duration < self.thresholds['duration_min'] or duration > self.thresholds['duration_max']:
                continue

            segment = delta_signal[start_idx:end_idx]
            raw_segment = signal_data[start_idx:end_idx]
            if np.ptp(segment) < self.thresholds['peak_to_peak_min']:
                continue

            if not self._evaluate_delta_wave(segment, raw_segment):
                continue

            wave_count = self._count_slow_waves(segment)
            if wave_count < self.thresholds['wave_count_min']:
                continue

            events.append({
                'start': start_idx / self.fs,
                'end': end_idx / self.fs,
                'duration': duration,
                'wave_count': wave_count,
                'confidence': self._compute_confidence(segment)
            })

        return self._consolidate_events(events)
    
    def _evaluate_delta_wave(self, delta_segment, raw_segment):
        """Evaluate delta wave candidate"""
        rms_delta = np.sqrt(np.mean(delta_segment ** 2))
        if rms_delta < self.thresholds['rms_delta']:
            return False

        hf_signal = self.sp.filter_to_band(raw_segment, [20, 30])
        hf_envelope, _ = self.sp.hilbert_envelope(hf_signal)
        hf_threshold = np.mean(hf_envelope) + 2 * np.std(hf_envelope)
        hf_peaks, _ = find_peaks(hf_envelope, height=hf_threshold)
        if len(hf_peaks) >= 2:
            return False

        artifact_score = self.sp.detect_artifacts(raw_segment)
        if artifact_score > self.thresholds['artifact_delta']:
            return False

        return True
    
    def _count_slow_waves(self, delta_segment):
        """Count number of slow waves"""
        zero_crossings = np.where(np.diff(np.sign(delta_segment)) != 0)[0]
        if len(zero_crossings) < 2:
            return 0

        wavelengths = np.diff(zero_crossings) / self.fs
        valid = (wavelengths >= self.thresholds['wavelength_min']) & (wavelengths <= self.thresholds['wavelength_max'])
        return int(np.sum(valid))
    
    def _compute_confidence(self, segment):
        """Compute delta wave confidence"""
        envelope, _ = self.sp.hilbert_envelope(segment)
        envelope_regularity = 1 / (1 + np.std(envelope) / (np.mean(envelope) + 1e-10))
        amplitude_score = min(1.0, np.max(np.abs(segment)) / 150.0)
        return 0.5 * envelope_regularity + 0.5 * amplitude_score
    
    def _count_slow_waves(self, delta_segment):
        """Count number of slow waves"""
        zero_crossings = np.where(np.diff(np.sign(delta_segment)) != 0)[0]
        if len(zero_crossings) < 2:
            return 0

        wavelengths = np.diff(zero_crossings) / self.fs
        valid = (wavelengths >= self.thresholds['wavelength_min']) & (wavelengths <= self.thresholds['wavelength_max'])
        return int(np.sum(valid))
    
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