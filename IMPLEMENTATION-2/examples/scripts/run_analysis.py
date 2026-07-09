"""
Example script for EEG analysis.

This script demonstrates the analysis workflow:
1. Load preprocessed EEG data
2. Detect sleep events (spindles, slow waves)
3. Perform spectral analysis
4. Generate visualizations
"""

import mne
from sleepeegpy.analysis import events, spectral
from sleepeegpy.visualization import event_plots, spectrogram
from sleepeegpy.utils import io


def main():
    """Run analysis pipeline."""
    # Load preprocessed EEG data
    # raw = mne.io.read_raw_fif('preprocessed_data.fif', preload=True)

    # Detect sleep spindles
    # spindles = events.detect_spindles(raw, ch_names=['C3', 'C4'])

    # Detect slow waves
    # slow_waves = events.detect_slow_waves(raw, ch_names=['F3', 'F4', 'C3', 'C4'])

    # Perform spectral analysis
    # analyzer = spectral.SpectralAnalyzer(raw.info['sfreq'])
    # psd_results = analyzer.analyze(raw.get_data())

    # Visualize events
    # event_plots.plot_events(raw, spindles=spindles, slow_waves=slow_waves)

    # Plot spectrogram
    # spectrogram.plot_spectrogram(raw)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
