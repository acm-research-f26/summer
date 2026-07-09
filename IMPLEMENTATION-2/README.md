![acm research banner light](https://github.com/acm-research/paperimplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# fall 2026 paper implementation

---

**paper title:** SleepEEGpy: a Python-based software integration package to organize preprocessing, analysis, and visualization of sleep EEG data

## ⚲ project summary
SleepEEGpy is a Python-based open-source software package that simplifies sleep EEG preprocessing, analysis, and visualization by integrating multiple established tools into a unified pipeline. Built on MNE-Python, PyPREP, YASA, and SpecParam, it offers an all-in-one solution for sleep EEG research, including cleaning, independent component analysis, sleep event detection, spectral feature analysis, and visualization tools.

## ⌖ motivation
Current EEG software packages are either implemented in MATLAB and behind a paywall, optimized for sleep EEG but restricted to either preprocessing, sleep scoring, or specific analyses, or based on Python but not necessarily optimized for sleep research. SleepEEGpy addresses this unmet need by providing a comprehensive package that goes beyond the typical configuration in many labs and combines multiple software environments to work with sleep EEG data.

## ✎ᝰ novelty
- **unified python-based pipeline**: SleepEEGpy integrates preprocessing, analysis, and visualization for general sleep EEG data into a single framework, eliminating the need to combine multiple software environments.

- **comprehensive event detection**: Integration of YASA-based detection algorithms for sleep spindles, slow waves, and rapid eye movements, providing features of detected events and average time-frequency representations.

- **sleep-specific preprocessing**: Tailored preprocessing for sleep data including resampling, filtering, bad channel detection and interpolation, artifact rejection, and ICA decomposition with sleep-specific considerations.

## ✰ methodology
1. **dataset**: uses the [Zenodo Sleep EEG dataset](10.5281/zenodo.10362190) dataset of overnight sleep EEG data from 44 healthy young adult participants (25 females, age 25.86 ± 3.14 years, ranging from 21 to 36 years) who participated in a research study on sleep and memory consolidation
2. **architecture**: python-based software pipeline w/ multiple signal processing techniques 
   
   - **cleaning (A1)**: Chebyshev Type I bandpass filter (0.3-40 Hz)
   
   - **independent component analysis (A2)**: EEG signal decomposition using FastICA
   
   - **event detection (B1)**: YASA-based detection algorithms
   
   - **spectral analysis (B2)**: Power spectral density per sleep stage using Welch's method (FFT length 256, Hamming window)
3. **evaluation**:
   - characteristic activity signatures typical of each vigilance state revealed: alpha oscillations in wakefulness, spindles and slow waves in NREM sleep, theta activity in REM sleep

   - comparison with EEGLAB using the same sleep EEG dataset validated consistency of spectral outputs; topographic distributions of key frequency bands across vigilance states showed near-identical spatial patterns between both platforms

   - event detection demonstrated with 48,057 spindles detected across all channels in N2 sleep (~187 per channel, ~0.97 spindles/minute) revealing established phenomena such as slower frontal spindles versus faster centroparietal spindles
4. **metrics**: 
   - **event detection metrics**: Number of events per channel, amplitude, frequency, duration, and topographical distribution of event features

   - **sleep stage duration**: Total minutes and percentage of recording in each sleep stage derived from hypnogram vectors (Wake, N1, N2, N3, REM)

   - **spectral parameterization**: Periodic component parameters including center frequency, power, and bandwidth; aperiodic component parameters including offset and exponent
   - **power spectral density**: PSD computed per sleep stage across frequency bands (delta 1-4 Hz, theta 4-8 Hz, alpha 8-12.5 Hz, sigma 12-15 Hz, beta 12.5-30 Hz) with mean ± standard deviation values reported for frontal, central, and occipital regions

## ⛰︎ impact
1. **lowering entry barriers**: Providing an accessible tool for beginners in sleep EEG research, reducing the frustrating initial encounters with sleep EEG data and enabling new researchers to focus on understanding high-level steps rather than coding errors and arbitrary parameter definitions.

2. **standardizing workflows**: Streamlining preprocessing, analysis, and visualization through a simple, script-based API that promotes reproducible research practices critical for large-scale and collaborative projects.

3. **enabling comprehensive analysis**: Offering a complete sleep EEG processing pipeline that includes artifact removal, ICA, event detection, spectral analyses, and visualization in a single framework.

#### future work
- integration of machine learning and deep learning algorithms for prediction and classification tasks

- identification of pathological events during sleep