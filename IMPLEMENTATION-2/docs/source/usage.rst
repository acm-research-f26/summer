Usage
=====

Basic Workflow
--------------

Here's a simple example of using SleepEEGPy:

.. code-block:: python

    import mne
    from sleepeegpy.preprocessing import cleaning, ica
    from sleepeegpy.analysis import events, spectral

    # Load EEG data
    raw = mne.io.read_raw_edf('data.edf', preload=True)

    # Preprocessing
    raw.filter(0.5, 30)  # Bandpass filter
    raw.resample(100)     # Resample to 100 Hz

    # ICA artifact removal
    ica_model = ica.fit_ica(raw)
    raw_clean = ica_model.apply(raw)

    # Detect sleep events
    spindles = events.detect_spindles(raw_clean)
    slow_waves = events.detect_slow_waves(raw_clean)

    # Spectral analysis
    analyzer = spectral.SpectralAnalyzer(raw_clean.info['sfreq'])
    psd_results = analyzer.analyze(raw_clean.get_data())

Notebooks
---------

Detailed tutorials are available in the `notebooks` directory:

- `01_cleaning.ipynb` - Data cleaning walkthrough
- `02_ica.ipynb` - ICA decomposition and artifact removal
- `03_dashboard.ipynb` - Interactive visualization
- `04_events.ipynb` - Sleep event detection
- `05_spectral.ipynb` - Spectral analysis
- `06_complete_pipeline.ipynb` - End-to-end example
