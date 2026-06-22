![acm research banner light](https://github.com/acm-research/paperimplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# fall 2026 paper implementation

---

**paper:** RAVEN: Software for automated sleep microstructure analysis from electroencephalography

## ⚲ paper summary
RAVEN (Rhythmic Analysis of Variations in EEG Neural activity) is a novel MATLAB-based algorithm that uses identification of K-complexes, sleep spindles, slow oscillating delta waves, and CAP cycles in EEG signals to create a more effective & reliable way to analyze multiple sleep microstructure characteristics at once.

## ⌖ motivation
Current wrist actigraphy-based screening for sleep apnea suffers from two critical limitations: (1) inability to detect snoring, a key acoustic biomarker for apnea, and (2) poor estimation of apnea event duration due to reliance on limb movement rather than respiratory effort. This project proposes a novel ear-EEG and audio-based wearable system that overcomes these limitations by capturing cortical arousal signatures through EEG while simultaneously analyzing tracheal breathing sounds. By monitoring ear-based neural activity and respiratory acoustics, this approach detects pure apnea events—where patients gasp or snort without significant limb movement—and precisely measures respiratory event durations, enabling accurate sleep apnea diagnosis without requiring in-lab polysomnography (PSG).

## ✎ᝰ novelty
- **Ear-based actigraphy with reduced motion artifact**: Unlike wrist-based actigraphy, the ear moves with significantly less erratic motion, providing more stable and reliable movement data for filtering out wake periods and body-position changes without the noise introduced by hand gestures or arm movement.
- **Integrated audio-EEG apnea event detection**: By synchronizing ear-EEG cortical arousal detection with tracheal audio analysis (snoring, gasping, snorting), this system directly identifies pure apnea events that wrist actigraphy misses, and uses the combined signal to precisely timestamp respiratory event onset and duration.
- **PSG-level metrics without PSG**: The system yields apnea-hypopnea indices (AHI), oxygen desaturation proxies (via audio-derived respiratory effort), and arousal-based event duration data—offering near-PSG diagnostic accuracy in a wearable device suitable for longitudinal home monitoring.

## ✰ methodology
1. **Dataset**: Uses the [PSG-Audio](https://www.scidb.cn/en/detail?dataSetId=778740145531650048) dataset containing synchronized polysomnography (PSG) and tracheal/ambient audio recordings from patients with OSA. The paper's RAVEN software was developed using high-quality PSG data from 10 healthy volunteers and validated in three independent datasets: the Montreal Archive of Sleep Studies (MASS; n=19), the CAP Database (n=8), and the SmartSleep Lab (SSL; n=11).

2. **Architecture**: RAVEN is a MATLAB-based software pipeline employing multiple complementary signal processing techniques:
   - **Signal filtering**: Chebyshev Type I bandpass filter (0.1-30 Hz) for raw EEG, with second-order Butterworth filters for specific bands: delta (0.1-4.5 Hz) for K-complexes and delta waves, sigma (11-16 Hz) for sleep spindles, and five bands (delta, theta, alpha, sigma, beta) for CAP analysis
   - **Detection algorithms**: Power thresholding with moving windows (50% overlap), Hilbert transform for envelope analysis, continuous wavelet transform (CWT) with Morlet wavelet for time-frequency decomposition, empirical mode decomposition (EMD) for intrinsic mode functions, and short-long term ratios for amplitude variation assessment
   - **Modular detector pipeline**: Separate detection algorithms for each microstructure type (K-complexes, spindles, delta waves, CAP cycles), with CAP using dual-channel detection for improved specificity

3. **Evaluation**:
   - Event-by-event matching between RAVEN detections and expert annotations; events with any temporal overlap classified as true positives (TP)
   - Expert-annotated events not detected by RAVEN classified as false negatives (FN); RAVEN detections absent from expert scoring classified as false positives (FP)
   - Performance assessed across three independent datasets with varying annotation standards

4. **Metrics**:
   - **Sensitivity**: TP/(TP+FN) — proportion of events detected
   - **Precision**: TP/(TP+FP) — proportion of detected events considered real
   - **F1-score**: Harmonic mean of precision and sensitivity — 2*(precision*sensitivity)/(precision+sensitivity)
   - These metrics were selected because they are unaffected by the large number of true negatives (non-event segments) that would artificially inflate specificity

#### additional methodology:
- **Artifact rejection**: FFT analysis of higher frequencies (60-90 Hz) to identify powerline interference and movement artifacts; RMS thresholds for EEG amplitude; event consolidation to remove duplicate detections across overlapping epochs
- **Threshold derivation**: Default thresholds calculated as median - 0.5×IQR from expert-annotated validation datasets to favor sensitivity over specificity

## ⛰︎ impact
This project has the potential to democratize sleep apnea screening by replacing expensive, sleep-lab-bound PSG with a consumer-grade, in-ear wearable. With near-PSG diagnostic accuracy, it can enable early detection of OSA in at-risk populations, reduce healthcare system burdens, and provide longitudinal data for tracking treatment efficacy (e.g., CPAP adherence). It also paves the way for personalized sleep medicine by enabling granular, night-by-night tracking of apnea severity and event characteristics in the patient's natural sleep environment.

#### future work
- **Real-time algorithm optimization**: Port the signal processing pipeline from MATLAB to embedded C/C++ for on-device inference, enabling real-time event alerts and closed-loop stimulation (e.g., auditory tone to prompt positional change)
- **Multi-day longitudinal studies**: Validate the device across multiple nights and diverse patient populations (including those with central apnea and mixed apnea) to assess robustness and reproducibility
- **Integration with oximetry**: Add a reflective pulse oximeter to the ear-worn device to directly capture SpO₂ desaturation events, further reducing reliance on PSG and adding a critical clinical metric for OSA severity grading
- **Clinical threshold validation**: Extend the ear-EEG/audio system to pathological datasets to validate detection thresholds across different patient populations and OSA severities

**additional sources:**
- Arolaakso, M., Pitkanen, H., Pitkanen, M., et al. (2025). RAVEN: Software for automated sleep microstructure analysis from electroencephalography. doi: 10.5281/zenodo.15386929
- PSG-Audio Dataset: A Synchronized Polysomnography and Audio Dataset for Sleep Apnea Research