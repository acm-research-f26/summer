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
            'power_sigma': 0.3,
            'rms_spindle': 10e-6,  # 10 μV
            'cwt_spindle': 0.4,
            'hilbert_std_spindle': 1.0,
            'artifact': 100,
            'amplitude_max': 100e-6,  # 100 μV
            'rms_standard_max': 20e-6,  # 20 μV
            'duration_min': 0.5,
            'duration_max': 10.0
        }
    
    def detect(self, signal_data):
        """Detect sleep spindles"""
        events = []
        
        # Step 1: Clip signal
        signal_clipped = np.clip(signal_data, -100e-6, 100e-6)
        
        # Step 2: Filter to sigma band
        sigma_signal = self.sp.filter_to_band(signal_clipped, [11, 16])
        
        # Step 3: Filter to 4.5-30 Hz for power ratio
        broadband_signal = self.sp.filter_to_band(signal_clipped, [4.5, 30])
        
        # Step 4: Calculate power
        sigma_power, sigma_time = self.sp.moving_window_power(sigma_signal, 0.3, 2/3)
        broad_power, broad_time = self.sp.moving_window_power(broadband_signal, 0.3, 2/3)
        
        # Step 5: Power ratio
        power_ratio = sigma_power / (broad_power + 1e-10)
        
        # Step 6: 30-second epochs with 50% overlap
        epoch_len = 30
        epoch_overlap = 0.5
        step_samples = int(epoch_len * self.fs * (1 - epoch_overlap))
        epoch_samples = int(epoch_len * self.fs)
        
        for start_sample in range(0, len(signal_data) - epoch_samples, step_samples):
            end_sample = start_sample + epoch_samples
            epoch_data = sigma_signal[start_sample:end_sample]
            
            # Check if power exceeds threshold
            epoch_power = np.mean(epoch_data ** 2)
            if epoch_power < self.thresholds['power_sigma']:
                continue
            
            # Detect spindles in epoch
            candidates = self._detect_in_epoch(epoch_data)
            
            for cand in candidates:
                abs_start = (start_sample + cand['start']) / self.fs
                abs_end = (start_sample + cand['end']) / self.fs
                duration = (cand['end'] - cand['start']) / self.fs
                
                if self.thresholds['duration_min'] <= duration <= self.thresholds['duration_max']:
                    events.append({
                        'start': abs_start,
                        'end': abs_end,
                        'duration': duration,
                        'confidence': cand.get('confidence', 0.5)
                    })
        
        return self._consolidate_events(events)
    
    def _detect_in_epoch(self, epoch_data):
        """Detect spindles within a single epoch"""
        # Apply EMD and get first IMF
        imfs, _ = self.sp.empirical_mode_decomposition(epoch_data)
        first_imf = imfs[0] if imfs else epoch_data
        
        # Compute CWT
        cwt_coeff, freqs = self.sp.continuous_wavelet_transform(first_imf)
        
        # Focus on spindle band
        freq_mask = (freqs >= 11) & (freqs <= 16)
        spindle_cwt = np.abs(cwt_coeff[freq_mask, :])
        
        # Threshold CWT
        cwt_mask = spindle_cwt > self.thresholds['cwt_spindle'] * np.max(spindle_cwt)
        
        # Find connected regions
        from scipy.ndimage import label
        labeled, num_features = label(cwt_mask)
        
        candidates = []
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            idx = np.where(mask.any(axis=0))[0]
            
            if len(idx) == 0:
                continue
            
            start_idx = idx[0]
            end_idx = idx[-1]
            duration = (end_idx - start_idx) / self.fs
            
            if duration >= 0.5:
                segment = epoch_data[start_idx:end_idx]
                sigma_segment = self.sp.filter_to_band(segment, [11, 16])
                
                # RMS threshold
                rms_sigma = np.sqrt(np.mean(sigma_segment ** 2))
                if rms_sigma < self.thresholds['rms_spindle']:
                    continue
                
                # Hilbert transform evaluation
                envelope, _ = self.sp.hilbert_envelope(segment)
                threshold = np.mean(envelope) + self.thresholds['hilbert_std_spindle'] * np.std(envelope)
                peaks, _ = find_peaks(envelope, height=threshold)
                
                if len(peaks) == 0:
                    continue
                
                # Peak-to-peak validation
                peak_to_peak = np.max(sigma_segment) - np.min(sigma_segment)
                if peak_to_peak > self.thresholds['amplitude_max']:
                    continue
                
                # RMS standard check
                rms_standard = np.sqrt(np.mean(segment ** 2))
                if rms_standard > self.thresholds['rms_standard_max']:
                    continue
                
                confidence = self._compute_confidence(segment)
                candidates.append({
                    'start': start_idx,
                    'end': end_idx,
                    'confidence': confidence
                })
        
        return candidates
    
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