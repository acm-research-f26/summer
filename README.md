# pHGFN — pH-Conditioned GFlowNet for RNA-Targeted Drug Design

## 📌 Project Summary

pHGFN generates diverse, drug-like molecules that selectively bind the KRAS i-motif at tumor pH (~6.7) while avoiding the unfolded RNA conformation at healthy pH (~7.4). By exploiting the pH-driven conformational switch of the KRAS promoter i-motif — a four-stranded RNA structure that folds only under acidic conditions — the model learns structural selectivity grounded in real 3D atomic coordinates, not hand-crafted pH-sensing features.

---

## 🎯 Motivation

KRAS is mutated in roughly 25% of all human cancers and ~90% of pancreatic adenocarcinoma cases, yet the KRAS protein has historically resisted direct small-molecule inhibition. The KRAS promoter contains a C-rich sequence that folds into an i-motif under the acidic microenvironment of solid tumors (Warburg effect, ~pH 6.7), suppressing transcription when stabilized. At healthy tissue pH (~7.4) this structure unfolds, making selective binding a built-in tumor-targeting handle — no engineered pH sensor required. Designing molecules that exploit this conformational switch could open a new class of transcription-suppressing cancer therapeutics.

---

## 🧩 Novelty

**First GFlowNet for RNA-targeted, pH-conditioned drug design:** Prior GFlowNet drug-design work targets proteins; this work targets a structurally dynamic RNA element where conformational selectivity matters more than raw affinity.

**Structure-grounded selectivity:** pH is implemented as a conformer switch — two real PDB files (folded i-motif vs. unfolded ssRNA) — so all structural signal comes from GNINA docking on atomic coordinates, not a scalar pH feature.

**RNA-FM + ChemBERTa + pH-conditioned cross-attention:** Frozen 640-d RNA-FM embeddings and frozen 768-d ChemBERTa molecular embeddings are fused by a trainable cross-attention head; the pH scalar routes the head to predict either the acidic-fold or neutral-fold GNINA score.

**SELFIES vocabulary with graded ADMET gating:** The 34-token SELFIES alphabet makes invalid molecules structurally impossible (100% syntactic validity), and lower-bound ADMET filters reject trivial fragments before any reward is assigned.

---

## 🧠 Methodology

**Dataset:** Uses the [HARIBOSS](https://hariboss.pasteur.cloud/) dataset of 98 RNA-ligand PDB complexes for auxiliary oracle training, augmented with ~2,000 SELFIES-sampled and HARIBOSS-ligand molecules docked with GNINA; behavior-clone pretraining uses ZINC250k.

**Architecture:** Four-stage pipeline — (1) HARIBOSS preprocessing for contact-based binding labels, (2) offline GNINA docking of ~2,000 molecules against both i-motif conformers, (3) proxy oracle (RNA-FM + ChemBERTa + pH-conditioned cross-attention, ~10k trainable params) trained on docking labels, (4) SELFIES causal transformer GFlowNet (6-layer, 8-head, 512-d) fine-tuned with Trajectory Balance loss and a graded, ADMET-gated reward.

**Evaluation:** Top-200 GFlowNet candidates independently re-docked with GNINA against both conformers to validate proxy predictions; Pareto frontier computed over selectivity (GNINA differential) vs. drug-likeness (QED).

**Metrics:** GNINA differential (score\_acidic − 1.5 × score\_neutral), QED, scaffold diversity, internal diversity, novelty vs. ZINC250k training set.

**Reward shaping:** Three-tier graded reward — invalid SMILES → 0.001, valid but not drug-like → 0.02 + 0.08 × QED, drug-like → 0.10 + exp(clamp(differential) / T) — so the policy climbs toward drug-likeness before optimizing selectivity.

**Why offline proxy + GFlowNet:** GNINA takes 5–10 min per molecule; GFlowNet requires millions of reward queries. Docking ~2,000 molecules once and distilling a fast proxy makes the loop tractable. GFlowNet is preferred over RL because it samples proportional to reward, preserving molecular diversity rather than mode-collapsing to a single optimized scaffold.

---

## 🌍 Impact

pHGFN demonstrates that the pH-driven conformational switch of oncogene promoter i-motifs can be directly embedded into a generative model's reward signal using real structural biology — no separate pH-sensing pharmacophore needed. All 198 independently re-docked top candidates showed confirmed acidic selectivity (100% sign accuracy, mean GNINA differential −1.827 kcal/mol, best −3.065 kcal/mol), with a scaffold diversity of 0.924 and complete novelty relative to the ZINC250k training set. This establishes a proof-of-concept framework for structure-selective, conformation-aware RNA-targeted drug design that could generalize to other pH-sensitive nucleic acid targets in the tumor microenvironment.

---

## Future Work

**Improve proxy ranking:** With ~2,000 docking labels the proxy achieves correct binary sign but low ranking correlation (r = −0.005); active learning to expand the docking set should close this gap.

**Generalize to other i-motifs:** The C-rich promoter regions of MYC, BCL2, and VEGF fold into similar i-motifs and could be targeted with the same pH-conditioned framework.

**Wet-lab validation:** Computational selectivity predictions should be validated with in vitro binding assays (e.g., circular dichroism, ITC) against folded and unfolded KRAS i-motif constructs.

---

## Additional Sources

- RNA-FM: [github.com/ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM)
- ChemBERTa: [github.com/seyonechithrananda/bert-loves-chemistry](https://github.com/seyonechithrananda/bert-loves-chemistry) · [Chithrananda et al., 2020](https://arxiv.org/abs/2010.09885)
- GNINA: [github.com/gnina/gnina](https://github.com/gnina/gnina)
- GFlowNet Foundations: [Bengio et al., 2021](https://arxiv.org/abs/2111.09266)
- Target-conditioned GFlowNet (TacoGFN): [Shen et al., 2024](https://arxiv.org/abs/2310.03223)
- SELFIES: [github.com/aspuru-guzik-group/selfies](https://github.com/aspuru-guzik-group/selfies) · [Krenn et al., 2020](https://arxiv.org/abs/1905.13741)
- HARIBOSS: [Panei et al., 2022](https://doi.org/10.1093/bioinformatics/btac483)
- GerNA-Bind: [Xia et al., 2025](https://doi.org/10.1038/s42256-025-01154-z)
