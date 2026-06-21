# create_sample_data.py
import numpy as np

def generate_sample_eeg(fs=256, duration=60):
    """Generate sample EEG data with simulated events"""
    t = np.arange(0, duration, 1/fs)
    
    # Base signal (delta + theta + alpha + beta)
    signal = (
        # Delta (0.5-4 Hz)
        10 * np.sin(2 * np.pi * 1 * t) + 
        5 * np.sin(2 * np.pi * 2.5 * t) +
        # Theta (4-8 Hz)
        8 * np.sin(2 * np.pi * 6 * t) +
        # Alpha (8-12 Hz)
        6 * np.sin(2 * np.pi * 10 * t) +
        # Beta (16-30 Hz)
        3 * np.sin(2 * np.pi * 20 * t)
    )
    
    # Add K-complex at 10s
    kc_start = int(10 * fs)
    kc_end = int(11 * fs)
    kc_time = t[kc_start:kc_end]
    kc = -30 * np.exp(-((kc_time - 10.5)**2) / (2 * 0.3**2))
    signal[kc_start:kc_end] += kc
    
    # Add spindle at 20s
    sp_start = int(20 * fs)
    sp_end = int(22 * fs)
    sp_time = t[sp_start:sp_end]
    spindle = 15 * np.sin(2 * np.pi * 13 * sp_time) * np.exp(-((sp_time - 21)**2) / (2 * 0.5**2))
    signal[sp_start:sp_end] += spindle
    
    # Add delta wave sequence at 30-35s
    delta_start = int(30 * fs)
    delta_end = int(35 * fs)
    delta_time = t[delta_start:delta_end]
    delta_wave = 20 * np.sin(2 * np.pi * 1.5 * delta_time) * np.exp(-((delta_time - 32.5)**2) / (2 * 2**2))
    signal[delta_start:delta_end] += delta_wave
    
    # Add noise
    signal += 2 * np.random.randn(len(signal))
    
    return signal, t

# Generate and save
signal, t = generate_sample_eeg()
import pandas as pd
df = pd.DataFrame({'time': t, 'signal': signal})
df.to_csv('data/sample_eeg.csv', index=False)
print("Sample data saved to data/sample_eeg.csv")