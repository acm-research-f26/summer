# File: src/spindle_detection.py
"""
Sleep spindle detection module for RAVEN
"""
import numpy as np
from scipy.signal import find_peaks
from signal_processing import SignalProcessor

class SpindleDetector:
    """Sleep spindle detection algorithm"""
    
    def __init__(self, fs, thresholds=None):
        self.fs = fs
        self.sp = SignalProcessor(fs)
        self.thresholds = thresholds or self._default_thresholds()
    
    def _default_thresholds(self):
        return {
            'envelope_std_scale': 1.0,
            'rms_spindle': 1.0,
            'amplitude_max': 120.0,
            'rms_standard_max': 50.0,
            'duration_min': 0.5,
            'duration_max': 3.0,
            'frequency_min': 11.0,
            'frequency_max': 16.0,
            'artifact': 100
        }
    
    def detect(self, signal_data):
        """Detect sleep spindles"""
        events = []
        sigma_signal = self.sp.filter_to_band(signal_data, [11, 16])
        broadband_signal = self.sp.filter_to_band(signal_data, [4.5, 30])

        envelope, _ = self.sp.hilbert_envelope(sigma_signal)
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

            segment = sigma_signal[start_idx:end_idx]
            raw_segment = signal_data[start_idx:end_idx]
            sigma_power = np.mean(segment ** 2)
            broad_power = np.mean(broadband_signal[start_idx:end_idx] ** 2)

            if sigma_power / (broad_power + 1e-10) < 0.1:
                continue

            rms_sigma = np.sqrt(np.mean(segment ** 2))
            if rms_sigma < self.thresholds['rms_spindle']:
                continue

            peak_to_peak = np.max(segment) - np.min(segment)
            if peak_to_peak > self.thresholds['amplitude_max']:
                continue

            peaks, _ = find_peaks(segment, height=0.2 * np.max(np.abs(segment)), distance=int(self.fs / self.thresholds['frequency_max']))
            if len(peaks) < int(self.thresholds['frequency_min'] * duration / 2):
                continue

            rms_standard = np.sqrt(np.mean(raw_segment ** 2))
            if rms_standard > self.thresholds['rms_standard_max']:
                continue

            artifact_score = self.sp.detect_artifacts(raw_segment)
            if artifact_score > self.thresholds['artifact']:
                continue

            events.append({
                'start': start_idx / self.fs,
                'end': end_idx / self.fs,
                'duration': duration,
                'confidence': self._compute_confidence(segment)
            })

        return self._consolidate_events(events)
    
    def _compute_confidence(self, segment):
        """Compute spindle confidence score"""
        peaks, _ = find_peaks(segment, height=0.2 * np.max(np.abs(segment)), distance=int(self.fs / self.thresholds['frequency_max']))
        if len(peaks) == 0:
            return 0.0

        intervals = np.diff(peaks) / self.fs
        if len(intervals) == 0:
            return 0.0

        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        regularity = 1 - (std_interval / (mean_interval + 1e-10))
        regularity = max(0.0, min(1.0, regularity))

        envelope, _ = self.sp.hilbert_envelope(segment)
        envelope_ratio = np.max(envelope) / (np.mean(envelope) + 1e-10)
        amplitude_score = max(0.0, min(1.0, 1 - (envelope_ratio - 1) / 4))

        return 0.5 * regularity + 0.5 * amplitude_score
    
    def _compute_confidence(self, segment):
        """Compute spindle confidence score"""
        # Check frequency stability
        peaks, _ = find_peaks(segment, height=0.1 * np.max(np.abs(segment)))
        if len(peaks) < 3:
            return 0
        
        intervals = np.diff(peaks) / self.fs
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Frequency should be 11-16 Hz (period 0.0625-0.091s)
        if mean_interval < 0.0625 or mean_interval > 0.091:
            return 0
        
        # Regularity score
        regularity = 1 - (std_interval / (mean_interval + 1e-10))
        regularity = max(0, min(1, regularity))
        
        # Amplitude modulation
        envelope, _ = self.sp.hilbert_envelope(segment)
        envelope_ratio = np.max(envelope) / (np.mean(envelope) + 1e-10)
        if envelope_ratio > 5:
            amplitude_score = 0.5
        else:
            amplitude_score = 1 - (envelope_ratio - 1) / 4
        amplitude_score = max(0, min(1, amplitude_score))
        
        return (regularity + amplitude_score) / 2
    
    def _consolidate_events(self, events):
        """Consolidate overlapping events"""
        if not events:
            return events
        
        events = sorted(events, key=lambda x: x['start'])
        consolidated = []
        current = events[0].copy()
        
        for event in events[1:]:
            if event['start'] - current['end'] <= 0.5:  # 500ms overlap
                current['end'] = max(current['end'], event['end'])
                current['duration'] = current['end'] - current['start']
                current['confidence'] = max(current['confidence'], event['confidence'])
            else:
                consolidated.append(current)
                current = event.copy()
        
        consolidated.append(current)
        return consolidated