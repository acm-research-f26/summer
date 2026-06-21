# create_sample_data.py
import numpy as np
import pandas as pd


def k_complex_template(t, center, duration=1.2, amplitude=80.0):
    """Generate a K-complex-like transient."""
    sigma = duration / 5
    return -amplitude * np.exp(-((t - center) ** 2) / (2 * sigma ** 2)) * (1 - ((t - center) / (2 * sigma)) ** 2)


def spindle_template(t, center, duration=6.0, amplitude=60.0, freq=13.0):
    """Generate a sleep spindle waveform."""
    envelope = np.exp(-((t - center) ** 2) / (2 * (duration / 4) ** 2))
    return amplitude * np.sin(2 * np.pi * freq * t) * envelope


def delta_wave_template(t, center, duration=8.0, amplitude=100.0, freq=1.5):
    """Generate a slow delta wave burst."""
    envelope = np.exp(-((t - center) ** 2) / (2 * (duration / 4) ** 2))
    return amplitude * np.sin(2 * np.pi * freq * t) * envelope


def generate_sample_eeg(fs=256, duration=60, noise_std=2.0):
    """Generate sample EEG data with simulated events."""
    t = np.arange(0, duration, 1 / fs)

    signal = (
        10 * np.sin(2 * np.pi * 1 * t)
        + 5 * np.sin(2 * np.pi * 2.5 * t)
        + 8 * np.sin(2 * np.pi * 6 * t)
        + 6 * np.sin(2 * np.pi * 10 * t)
        + 3 * np.sin(2 * np.pi * 20 * t)
    )

    events = []

    kc_center = 10.5
    signal += k_complex_template(t, center=kc_center, duration=1.2, amplitude=80.0)
    events.append({'label': 'kcomplex', 'start': 10.0, 'end': 11.0})

    sp_center = 23.0
    signal += spindle_template(t, center=sp_center, duration=6.0, amplitude=60.0)
    events.append({'label': 'spindle', 'start': 20.0, 'end': 26.0})

    delta_center = 36.0
    signal += delta_wave_template(t, center=delta_center, duration=8.0, amplitude=100.0)
    events.append({'label': 'deltawave', 'start': 32.0, 'end': 40.0})

    signal += noise_std * np.random.randn(len(signal))
    return signal, t, events

def save_sample_eeg(output_path='data/sample_eeg.csv', fs=256, duration=60):
    signal, t, events = generate_sample_eeg(fs=fs, duration=duration)
    df = pd.DataFrame({'time': t, 'signal': signal})
    df.to_csv(output_path, index=False)
    print(f'Sample data saved to {output_path}')
    print('Generated events:')
    for event in events:
        print(f"  - {event['label']} from {event['start']}s to {event['end']}s")


if __name__ == '__main__':
    save_sample_eeg()