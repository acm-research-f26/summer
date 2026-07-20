![acm research banner light](https://github.com/acm-research/paperimplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# fall 2026 paper implementation

---

**paper title:** novel, wearable, in-ear eeg technology to assess sleep and daytime sleepiness

## ⚲ project summary
this project implements and validates a novel in-ear eeg system for sleep staging and daytime sleepiness assessment. the implementation reproduces the key findings from the paper, demonstrating that in-ear eeg can achieve comparable performance to traditional polysomnography (psg) for sleep monitoring. the system uses custom-fitted eeg earbuds with dry-contact electrodes positioned in the external auditory canals, providing a comfortable and practical alternative to traditional sleep studies.

## ⌖ motivation
traditional polysomnography is the gold standard for sleep assessment but is costly, burdensome, and limited to laboratory settings. this creates significant barriers for longitudinal sleep monitoring and limits access to sleep healthcare. the motivation for this implementation is to validate a wearable in-ear eeg system that can:
- enable comfortable, at-home sleep monitoring
- reduce the cost and complexity of sleep studies
- provide clinical-grade sleep staging accuracy
- enable longitudinal tracking of sleep patterns and daytime sleepiness
- make sleep healthcare more accessible to diverse populations

## ✎ᝰ novelty
- **hardware innovation**: custom-fit eeg earbuds with dry-contact electrodes that maintain stable skin contact without gel or paste, using a proprietary conductive polymer coating for optimal signal quality
- **cross-head configuration**: interaural (cross-head) bipolar channel configuration that maximizes dynamic range and delta-band power, improving sleep stage discrimination compared to single-ear derivations
- **clinical validation**: first validation of in-ear eeg for maintenance of wakefulness test (mwt) sleep onset latency, including both healthy controls and participants with central disorders of hypersomnolence (cdh)
- **comprehensive feature set**: extraction of over 40 features including spectral powers, ratios, statistical measures, hjorth parameters, and spectral edge frequency for robust sleep staging

## ✰ methodology
1. **dataset**: uses the eesm23 dataset from openneuro (ds005178) containing 320 nights of recordings from 30 healthy subjects, including simultaneous in-ear eeg and full scalp psg recordings in bids format
2. **architecture**: random forest classifier with balanced class weights and hyperparameter tuning
   - feature extraction includes delta, theta, alpha, beta band powers and ratios
   - statistical features (mean, std, skew, kurtosis, rms, peak-to-peak)
   - hjorth parameters (activity, mobility, complexity)
   - zero-crossing rate and spectral edge frequency
   - standard scaling for feature normalization
3. **evaluation**: stratified train-test split (80-20) with cross-validation
   - epoch-level agreement analysis (30-second epochs)
   - sleep onset latency comparison for mwt trials
   - overnight sleep architecture metrics (tst, se, waso)
   - confusion matrix analysis for stage-specific performance
4. **metrics**: accuracy, cohen's kappa, f1 score (weighted and per-class), precision, recall, confusion matrix

#### additional methodology:
- **data preprocessing**: bandpass filtering (0.5-30 hz), resampling to 250 hz, 30-second epoch segmentation
- **missing data handling**: robust handling of artefacts and missing labels with fallback strategies
- **datalad integration**: efficient data management with on-demand data retrieval for large-scale datasets

## ⛰︎ impact
this implementation demonstrates that in-ear eeg technology can provide clinical-grade sleep staging in a comfortable, wearable form factor. the ability to monitor sleep architecture and daytime sleepiness outside the laboratory has significant implications for:
- **clinical practice**: enables remote sleep monitoring for patients with narcolepsy, idiopathic hypersomnia, and other sleep disorders
- **research**: facilitates large-scale longitudinal sleep studies with reduced burden on participants
- **treatment monitoring**: allows objective assessment of medication effects on sleep and alertness over extended periods
- **accessibility**: reduces barriers to sleep healthcare by providing a more comfortable and accessible alternative to traditional psg
- **personalized medicine**: enables continuous monitoring of sleep patterns for personalized treatment optimization

#### future work
- **deep learning models**: implement convolutional neural networks (cnns) or recurrent neural networks (rnns) for end-to-end sleep staging
- **multimodal integration**: incorporate inertial measurement unit (imu) data for motion artifact detection and sleep/wake classification
- **real-time processing**: develop streaming capabilities for real-time sleep stage classification
- **cross-dataset validation**: validate the model on multiple datasets to ensure generalizability
- **transfer learning**: leverage large-scale scalp eeg datasets to improve in-ear eeg classification
- **mobile deployment**: deploy the model on edge devices for standalone sleep monitoring applications

**additional sources:**
- paper: berent, j. et al. (2026). "a novel, wearable, in-ear eeg technology to assess sleep and daytime sleepiness." bioelectronic medicine.
- dataset: eesm23 (openneuro ds005178) - ear-eeg sleep monitoring data sets
- mne: python library for eeg/meg/ecg analysis
- scikit-learn: machine learning library used for classification