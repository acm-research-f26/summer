# create_sample_data.py
import json
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


def _get_by_path(data, path):
    if path is None:
        return None
    value = data
    for key in path.split('.'):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def _find_numeric_array(data):
    if isinstance(data, list):
        if data and all(isinstance(x, (int, float)) for x in data):
            return data
        for item in data:
            result = _find_numeric_array(item)
            if result is not None:
                return result
        return None
    if isinstance(data, dict):
        for key in ['signal', 'eeg', 'data', 'values', 'samples', 'amplitude', 'wave']:
            if key in data:
                result = _find_numeric_array(data[key])
                if result is not None:
                    return result
        for value in data.values():
            result = _find_numeric_array(value)
            if result is not None:
                return result
    return None


def load_sample_eeg_from_json(input_path, fs=256, signal_key=None, time_key=None):
    with open(input_path, 'r') as f:
        data = json.load(f)

    signal_data = _get_by_path(data, signal_key) if signal_key else None
    if signal_data is None:
        signal_data = _find_numeric_array(data)

    if signal_data is None:
        raise ValueError(f'No numeric signal array found in JSON: {input_path}')

    signal = np.asarray(signal_data, dtype=float)
    if time_key:
        time_data = _get_by_path(data, time_key)
        if time_data is None:
            raise ValueError(f'No time array found at JSON path: {time_key}')
        t = np.asarray(time_data, dtype=float)
        if len(t) != len(signal):
            raise ValueError('Time array length does not match signal length')
    else:
        t = np.arange(len(signal)) / fs

    return signal, t, []


def save_sample_eeg(output_path='data/sample_eeg.csv', fs=256, duration=60, input_json=None, signal_key=None, time_key=None):
    if input_json:
        signal, t, events = load_sample_eeg_from_json(
            input_json,
            fs=fs,
            signal_key=signal_key,
            time_key=time_key,
        )
    else:
        signal, t, events = generate_sample_eeg(fs=fs, duration=duration)
    df = pd.DataFrame({'time': t, 'signal': signal})
    df.to_csv(output_path, index=False)
    print(f'Sample data saved to {output_path}')
    if events:
        print('Generated events:')
        for event in events:
            print(f"  - {event['label']} from {event['start']}s to {event['end']}s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create sample EEG dataset or convert JSON dataset to CSV')
    parser.add_argument('--output', '-o', default='data/sample_eeg.csv', help='CSV output path')
    parser.add_argument('--fs', type=int, default=256, help='Sampling frequency for generated or JSON data')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds for generated data')
    parser.add_argument('--input-json', help='Path to JSON dataset containing EEG signal data')
    parser.add_argument('--signal-key', help='Dot-separated JSON path to the signal array')
    parser.add_argument('--time-key', help='Dot-separated JSON path to the time array')
    args = parser.parse_args()

    save_sample_eeg(
        output_path=args.output,
        fs=args.fs,
        duration=args.duration,
        input_json=args.input_json,
        signal_key=args.signal_key,
        time_key=args.time_key,
    )