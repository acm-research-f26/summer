![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

---

# Reducing redshift-dependent bias in CNN-based gravitational lens detection through physically motivated data augmentation.

## 📌 Project Summary
Gravitational lensing is a phenomena that occurs when light bends around massive objects because of gravity, causing warped images, reflections, or even rings around. As of now, current CNN models exhibit some bias in redshift (high-z) sourced graviational lensing. CNNs have emerged as a leading architecture in detecting this phenomenon. However, they exhibit a systemic bias against where performance degrades significantly for lenses whose background source galaxies are at high redshift, where they appear fainter and redder. This project explores physically based data augmentation to reduce this bias and therefore imporve the detection of valuable gravitational learning phenomenon. 

## 🎯 Motivation
With the rise of new telescopic missions like Euclid and LSST, more data is available, and therefore the logical next step is to automate the recognition of such phenomena. High-z lenses are extremely useful for cosmological measurements like the Hubble constant. Improving high-redshift lens detection directly expands the sample of lenses available for time-delay cosmography, which can be used to derive the hubble constant 

## 🧩 Novelty
- As of now, current CNN models exhibit some bias in redshift sourced graviational lensing.
- This project aims to reduce that bias to allow for seamless observation of new redshift based systems. 

## 🧠 Methodology
1. **Dataset**: 
Uses simulated data created via Lenstronomy, a simulation tool used frequently in the field. Source galaxy brightness and size are physically linked to redshift — higher redshift sources appear fainter and more compact, mimicking the real cosmological dimming effect. Training data is deliberately skewed toward low-redshift sources (z = 0.2–0.6) to replicate the known bias in real survey training sets (e.g. SLACS). Test data spans the full redshift range (z = 0.2–2.0) to enable fair evaluation across all redshift bins
2. **Architecture**: Resnet-18
   - Body frozen — preserves generic visual features (edge, curve, texture detectors) learned from ImageNet
   - Classification head replaced with Linear(512, 1) for binary lens/non-lens output
   - Loss function: BCEWithLogitsLoss (binary cross-entropy with logits)
   - Optimizer: Adam, lr=0.0001
3. **Evaluation**:
   -Detection recall stratified by source redshift bin: (0.2–0.6], (0.6–1.0], (1.0–1.4], (1.4–1.8], (1.8–2.2]
   - AUC-ROC on held-out test set
   - Non-lens classification accuracy as a sanity check (confirms any performance gap is redshift-specific, not general model weakness)
5. **Metrics**:
   - 

#### Additional Methodology:
- **Something optional**: Sentence

## 🌍 Impact
By being able to better identify instances of redshift gravitational lensing, we can eventually use this data to get closer estimates of constants like the hubble constant

#### Future Work
- **Something optional**: Sentence

**Additional Sources:**
- Wilde et al. 2022, MNRAS — CNN interpretability and color bias in lens detection
