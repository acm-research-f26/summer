"""
Example script for EEG preprocessing.

This script demonstrates the basic preprocessing workflow:
1. Load EEG data
2. Resample and filter
3. Detect and remove bad channels/epochs
4. Apply ICA decomposition
5. Save preprocessed data
"""

import mne
from sleepeegpy.preprocessing import cleaning, ica
from sleepeegpy.utils import io


def main():
    """Run preprocessing pipeline."""
    # Load raw EEG data
    # raw = io.load_eeg('data/sample_data/sample.edf')

    # Resample to 100 Hz
    # raw.resample(100)

    # Apply bandpass filter (0.5 - 30 Hz)
    # raw.filter(0.5, 30)

    # Detect bad channels (example: using standard deviation threshold)
    # std_threshold = 5
    # for ch in raw.ch_names:
    #     if raw[ch][0].std() > std_threshold:
    #         raw.info['bads'].append(ch)

    # Fit ICA
    # ica_model = ica.fit_ica(raw, n_components=30)

    # Detect artifact components
    # artifact_comps = ica.detect_artifact_components(ica_model, raw)
    # ica_model.exclude = artifact_comps

    # Apply ICA
    # raw_ica = ica_model.apply(raw)

    # Save preprocessed data
    # raw_ica.save('preprocessed_data.fif')

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
