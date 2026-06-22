# File: src/k_complex_detection.py
"""
K-Complex detection module for RAVEN
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, correlate

# Use absolute import (no dot)
from signal_processing import SignalProcessor

class KComplexDetector:
    """K-Complex detection algorithm"""
    
    def __init__(self, fs, thresholds=None):
        self.fs = fs
        self.sp = SignalProcessor(fs)
        self.thresholds = thresholds or self._default_thresholds()
    
    def _default_thresholds(self):
        """Return default thresholds from paper"""
        return {
            'power_delta': 0.5,
            'rms_delta': 10.0,
            'hilbert_std': 1.5,
            'correlation': 0.6,
            'cwt': 0.3,
            'artifact': 100,
            'amplitude_min': 40.0,
            'rms_standard_max': 60.0,
            'duration_min': 0.5,
            'duration_max': 3.0
        }
    
    def detect(self, signal_data):
        """Detect K-complexes in the signal"""
        events = []

        # Step 1: Filter to delta band and get envelope
        delta_signal = self.sp.filter_to_band(signal_data, [0.5, 4.5])
        envelope, _ = self.sp.hilbert_envelope(delta_signal)
        threshold = np.mean(envelope) + self.thresholds['hilbert_std'] * np.std(envelope)
        peaks, _ = find_peaks(envelope, height=threshold, distance=int(0.5 * self.fs))

        for peak_idx in peaks:
            start_sample = max(0, peak_idx - int(1.0 * self.fs))
            end_sample = min(len(signal_data), peak_idx + int(1.0 * self.fs))
            duration = (end_sample - start_sample) / self.fs

            if duration < self.thresholds['duration_min']:
                continue
            if duration > self.thresholds['duration_max']:
                continue

            segment = signal_data[start_sample:end_sample]
            delta_segment = delta_signal[start_sample:end_sample]

            if not self._passes_k_complex_checks(segment, delta_segment):
                continue

            events.append({
                'start': start_sample / self.fs,
                'end': end_sample / self.fs,
                'duration': duration,
                'confidence': self._compute_confidence(delta_segment)
            })

        return self._consolidate_events(events)

    def _passes_k_complex_checks(self, segment, delta_segment):
        """Evaluate whether a candidate segment looks like a K-complex"""
        if len(delta_segment) < int(self.fs * self.thresholds['duration_min']):
            return False

        rms_delta = np.sqrt(np.mean(delta_segment ** 2))
        if rms_delta < self.thresholds['rms_delta']:
            return False

        peak_to_peak = np.max(delta_segment) - np.min(delta_segment)
        if peak_to_peak < self.thresholds['amplitude_min']:
            return False

        peaks, _ = find_peaks(delta_segment, height=0.25 * np.max(delta_segment), distance=int(self.fs * 0.2))
        valleys, _ = find_peaks(-delta_segment, height=0.25 * np.max(-delta_segment), distance=int(self.fs * 0.2))
        if len(peaks) == 0 or len(valleys) == 0:
            return False

        if np.min(np.abs(peaks[0] - valleys[0])) > int(1.5 * self.fs):
            return False

        rms_standard = np.sqrt(np.mean(segment ** 2))
        if rms_standard > self.thresholds['rms_standard_max']:
            return False

        artifact_score = self.sp.detect_artifacts(segment)
        if artifact_score > self.thresholds['artifact']:
            return False

        return True

    def _compute_correlation(self, segment):
        """Compute correlation with canonical K-complex template"""
        # Generate template
        duration = 1.0
        t = np.linspace(-duration / 2, duration / 2, int(duration * self.fs))
        template = np.exp(-(t ** 2) / (2 * 0.15 ** 2)) * (-1 + 0.5 * (t / 0.15) ** 2)
        template = template / np.max(np.abs(template))

        segment_norm = segment - np.mean(segment)
        if np.max(np.abs(segment_norm)) > 0:
            segment_norm = segment_norm / np.max(np.abs(segment_norm))

        if len(segment_norm) != len(template):
            from scipy import interpolate
            x_old = np.linspace(0, 1, len(segment_norm))
            x_new = np.linspace(0, 1, len(template))
            f = interpolate.interp1d(x_old, segment_norm)
            segment_resampled = f(x_new)
        else:
            segment_resampled = segment_norm

        return np.corrcoef(segment_resampled, template)[0, 1]
    
    def _find_connected_regions(self, mask):
        """Find connected regions in binary mask"""
        from scipy.ndimage import label
        labeled, num_features = label(mask)
        regions = [np.where(labeled == i) for i in range(1, num_features + 1)]
        return regions
    
    def _compute_confidence(self, segment):
        """Compute confidence score for detection"""
        envelope, _ = self.sp.hilbert_envelope(segment)
        peak_to_peak = np.max(segment) - np.min(segment)
        
        # Confidence based on amplitude and envelope regularity
        amplitude_score = min(1.0, peak_to_peak / 100e-6)
        envelope_score = 1.0 / (1.0 + np.std(envelope) / (np.mean(envelope) + 1e-10))
        
        return 0.5 * amplitude_score + 0.5 * envelope_score
    
    def _consolidate_events(self, events):
        """Consolidate overlapping events"""
        if not events:
            return events
        
        # Sort by start time
        events = sorted(events, key=lambda x: x['start'])
        
        consolidated = []
        current = events[0].copy()
        
        for event in events[1:]:
            if event['start'] - current['end'] <= 0.001:  # 1ms overlap
                current['end'] = max(current['end'], event['end'])
                current['duration'] = current['end'] - current['start']
                current['confidence'] = max(current['confidence'], event['confidence'])
            else:
                consolidated.append(current)
                current = event.copy()
        
        consolidated.append(current)
        return consolidated