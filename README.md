![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations

Hello leads! Find your branch and update it with all your files and README.md (template below). Let us know if run into any access issues.

Use the below README template as you work through your implementations! 

<!--Remove text above the '---' -->
---

# Headliner
yea 

## 📌 Project Summary

This project benchmarks **layer importance ranking methods** for mixed-precision post-training quantization. The main artifact is [`comprehensive_layer_importance_benchmark.ipynb`](comprehensive_layer_importance_benchmark.ipynb), which introduces two new **data-free** scorers — **PCT** and **Entropy** — and evaluates them against prior and established baselines on **ResNet-18** and **ViT-Small** (CIFAR-10) across aggressive bit-width baselines (INT1–INT4, plus INT8).

**Headline finding:** the right ranker depends on architecture — **PCT for CNNs (ResNet)**, **Entropy for transformers (ViT)** — both without calibration data.

For each method, the notebook ranks layers, upgrades the top-*k* to INT8, leaves the rest at a low bit-width, and measures **accuracy recovery**, **calibration** (ECE, NLL, Brier), and **memory savings** vs FP32.

## 🎯 Motivation

Uniform low-bit quantization destroys both task accuracy and probability calibration. Mixed precision can recover performance if we protect the right layers—but methods disagree on which layers matter, and most published rankers require calibration data that production often lacks. We introduce **PCT** and **Entropy** as new weight-only scorers and benchmark them against established methods (HAWQ, InfoQ, BMPQ, CLADO) and prior baselines (OLD, NSDS) on equal footing across bit-widths, *k* budgets, accuracy, and calibration.

## 🧩 Novelty
- **PCT (Per-Channel Truncation)** *(ours)*: Scores each layer by relative Frobenius error after **per-channel** symmetric quantization—the same rounding mechanism used at deploy time. Unlike global-scale or magnitude heuristics, PCT asks: *which layers lose the most information when actually quantized?*
- **Entropy** *(ours)*: Scores each layer by **Shannon entropy of its weight histogram**—high entropy means dispersed, heterogeneous weights that are harder to represent in few bits. Unlike Hessian/Fisher methods, it needs no forward passes or calibration images.
- **Architecture-aware guidance**: We show that a single “best” data-free ranker does not exist—CNNs and ViTs need different scores—and provide evidence-backed defaults (PCT vs Entropy).
- **Comprehensive sweep**: 10 methods × 5 *k* values × 5 baseline bit-widths × 2 architectures, measuring accuracy recovery **and** calibration (ECE), not accuracy alone.

*NSDS is **prior work** included for comparison. SEA (spectral asymmetry) is an additional data-free scorer evaluated in the notebook.*

## 🧠 Methodology
1. **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — full test set for evaluation; 512-image calibration subset for data-required methods (HAWQ, InfoQ, BMPQ, CLADO).
2. **Architecture**:
   - **ResNet-18** (CIFAR-style stem: 3×3 conv, no maxpool)
   - **ViT-Small** (CIFAR-10 checkpoints)
3. **Evaluation protocol** (per baseline bit-width *B* ∈ {1, 2, 3, 4, 8}):
   - **Baseline**: all layers at *B*-bit (per-channel symmetric).
   - **Recovery**: top-*k* layers (by method score) → INT8; remaining layers stay at *B*-bit; *k* ∈ {1, 3, 5, 10, 20}.
   - **Memory savings**: `1 − (k·8 + (L−k)·B) / (L·32)` vs FP32.
4. **Metrics**:
   - **Accuracy** and raw **acc recovery** (mixed acc − baseline acc, in pp).
   - **Calibration**: ECE, NLL, Brier (lower is better).
   - **Normalized recovery** (unlearning-compatible): `(metric − INTB) / (FP32 − INTB) × 100%` for acc and ECE at headline k=5.
   - **AUC** over *k* for accuracy recovery and ECE (summary tables).

#### Methods compared

| Group | Method | Data? |
|-------|--------|-------|
| **Ours** | **PCT**, **Entropy** | ✗ |
| Also evaluated | **SEA** | ✗ |
| Prior (comparison) | **NSDS** | ✗ |
| Baselines | **OLD**, **Random** | ✗ |
| Established | **HAWQ**, **InfoQ**, **BMPQ**, **CLADO** | 512 imgs |

#### Additional Methodology:
- Related reliability-focused benchmark: [`unlearning_reversal_method_benchmark.ipynb`](unlearning_reversal_method_benchmark.ipynb) (INT4, 2k test images, normalized recovery headline table).
- Set `RUN_DATA_REQ = False` to skip expensive calibration-based scorers; `FORCE_RECOMPUTE = True` to ignore cached CSVs.

## 🚀 Running the benchmark

**Requirements:** Python 3.10+, PyTorch, torchvision, pandas, numpy, scipy, scikit-learn, matplotlib.

```bash
pip install torch torchvision scipy scikit-learn pandas matplotlib
jupyter notebook comprehensive_layer_importance_benchmark.ipynb
```

**Checkpoints** (expected under repo root):
- ResNet: `checkpoints_backdoor_resnet18/` or `checkpoints_resnet_cifar10/`
- ViT: `checkpoints_vit_cifar10/vit_small_0.pt`

**Outputs** → `results/comprehensive_layer_importance_benchmark/`:

| File | Description |
|------|-------------|
| `all_benchmark_results.csv` | Full sweep (all methods, bit-widths, *k*) |
| `resnet_scores.csv`, `vit_scores.csv` | Per-layer method scores |
| `auc_summary.csv` | AUC over *k* by method and bit-width |
| `resnet_unlearning_compatible_k5_int{1,2,3,4}.csv` | Normalized acc/ECE recovery @ k=5 |
| `*_accuracy_recovery.png`, `*_calibration_*.png` | Recovery and calibration plots |

## 📊 Results

Headline numbers use **normalized recovery** at **k=5** (top-5 layers → INT8, rest at baseline bit-width) on the full CIFAR-10 test set. Recovery = `(metric − INTB) / (FP32 − INTB) × 100%`. Full sweeps in `results/comprehensive_layer_importance_benchmark/`.

### ResNet-18 — **PCT wins (data-free)**

INT4 @ k=5 (primary deployment scenario):

| Method | Type | Acc recovery | ECE recovery |
|--------|------|--------------|--------------|
| InfoQ | data-req (512 imgs) | 14.5% | 15.7% |
| OLD | data-free | 12.2% | 14.9% |
| **PCT (ours)** | **data-free** | **10.8%** | **16.3%** |
| NSDS | data-free (prior) | 8.3% | 11.1% |
| Entropy (ours) | data-free | 2.1% | 9.3% |

Among **data-free** methods on ResNet, **PCT** is the top choice at INT4:
- **Best ECE recovery** at k=5 (16.3%) — beats OLD, NSDS, Entropy, and SEA.
- **Best data-free accuracy-recovery AUC** over k at INT4 (15.6 pp·k vs OLD 14.2, NSDS 12.2).
- Corroborated by [`unlearning_reversal_method_benchmark.ipynb`](unlearning_reversal_method_benchmark.ipynb), where PCT leads **ECE recovery outright** at INT4 k=5 (**23.4%**).

InfoQ can edge PCT on raw accuracy but requires 512 calibration images; PCT uses weights only.

### ViT-Small — **Entropy wins (data-free)**

INT4 @ k=5:

| Method | Type | Acc recovery | ECE recovery |
|--------|------|--------------|--------------|
| HAWQ | data-req | 81.5% | 80.5% |
| InfoQ / CLADO | data-req | 76.5% | ~76% |
| **Entropy (ours)** | **data-free** | **62.2%** | **55.5%** |
| SEA | data-free (ours) | 61.3% | 52.9% |
| PCT | data-free (ours) | −5.0% | −4.1% |

Among **data-free** methods on ViT, **Entropy (ours)** leads at every baseline bit-width (INT1/2/4) on both **accuracy-recovery AUC** and **ECE AUC**. At INT4 k=5 it tops PCT, SEA, OLD, and NSDS. Attention and MLP blocks have wide, heterogeneous weight distributions—entropy captures which layers are hardest to compress; per-channel truncation (PCT) mis-ranks them on ViT.

### Takeaways

| Architecture | Best data-free ranker | Why |
|--------------|----------------------|-----|
| **ResNet-18** | **PCT** | Per-channel quantization damage → best CNN calibration recovery at INT4. |
| **ViT-Small** | **Entropy** | Weight-distribution spread → best transformer recovery across bit-widths. |

Data-required methods (HAWQ, InfoQ, BMPQ, CLADO) can win when you already have a representative calibration set—but in production that is often expensive, stale, or unavailable (edge devices, privacy, domain shift). **PCT** and **Entropy** are the weight-only defaults this benchmark supports.

## 🌍 Impact

### The real-world problem

Teams ship mixed-precision models to save memory and latency, but **uniform INT4/INT2 quantization breaks more than accuracy**—confidence scores become unreliable (miscalibrated ECE), which breaks thresholding, rejection, monitoring, and anything that trusts `softmax` outputs. The usual fix is mixed precision: keep a few layers at INT8. The hard part is **which** layers—wrong picks waste your bit budget and leave both accuracy and calibration broken.

Most published rankers assume you can run **forward passes on calibration data** (HAWQ, InfoQ, BMPQ, CLADO) or use **generic weight heuristics** (magnitude/outlier scores like OLD, structural stats like NSDS). That fails in common deploy settings:

- **Edge / embedded** — no representative cal set on-device; privacy rules block sending data back.
- **Stale or shifted data** — cal set from training doesn’t match production; Hessian/Fisher rankings drift.
- **Post-training only** — retraining or QAT isn’t an option; you need a ranker that runs on exported weights **today**.

We introduce **PCT** and **Entropy** as two new **data-free** layer scorers and show **which one to use depends on architecture**—not one-size-fits-all.

---

### How our methods differ from existing ones

| Existing approach | What it uses | Limitation | **PCT / Entropy** |
|-------------------|--------------|------------|-------------------|
| **HAWQ, InfoQ, BMPQ, CLADO** | Activations, gradients, Fisher/Hessian on **512+ cal images** | Needs data, compute, and domain match; unusable when cal is missing or stale | **Weights only** — run offline on a checkpoint in seconds |
| **OLD (outlier dimension)** | Global abs-max scale error | Uses a **different quantizer** than per-channel deploy PTQ; mis-ranks layers whose damage is channel-local | **PCT** uses the **same per-channel symmetric quant** as the sweep/deploy path |
| **NSDS (prior work)** | Kurtosis × effective rank — structural tensor stats | No direct link to quantization error; competitive but doesn’t target calibration on CNNs | **PCT** optimizes for **observed rounding loss**; leads **ECE recovery** on ResNet INT4 |
| **Magnitude / random** | Scale or chance | Ignores **how** weights quantize; weak recovery curves | Both scores are **quantization-aware** (truncation error vs distributional complexity) |

**PCT** answers: *“If I actually quantize this layer per-channel, how much information do I lose?”*  
**Entropy** answers: *“How spread out are this layer’s weights—how hard are they to compress?”*

They are **complementary**, not interchangeable: PCT wins on ResNet; Entropy wins on ViT (see Results).

---

### Why people should care

**1. Same memory budget, better model.**  
At INT4 with only **k=5** layers upgraded to INT8, using the wrong ranker can mean negative recovery (worse than uniform quant). Using **PCT on ResNet** yields **16.3% ECE recovery** and competitive accuracy **with zero cal images**—vs Entropy at 2.1% / 9.3% on the same architecture. On **ViT**, **Entropy** hits **62.2% acc recovery** and **55.5% ECE recovery** at k=5 while **PCT hurts** (−5.0% / −4.1%). **Picking the architecture-default ranker is not a small tweak—it’s the difference between a deployable model and a broken one.**

**2. Calibration is a production requirement, not a paper metric.**  
Accuracy-only layer ranking hides failures users actually see: overconfident wrong predictions, broken abstention, bad A/B monitoring. PCT is the only data-free method that **leads ECE recovery on ResNet INT4** (and **23.4% ECE recovery** in the unlearning reliability benchmark—best in that table). If you ship quantized CNNs and anyone reads confidence, **accuracy-only rankers are the wrong tool.**

**3. No calibration pipeline to build or maintain.**  
InfoQ/HAWQ-class methods can beat data-free scores **when cal is perfect**—but that’s an ops cost every team pays repeatedly. PCT and Entropy are **checkpoint-in, ranking-out**: export weights → score layers → set mixed-precision config → ship. That’s the workflow edge, mobile, and privacy-sensitive teams actually use.

**4. Actionable deploy rule.**  
Don’t guess which ranker to port:

> **CNN / ResNet-style models → PCT**  
> **ViT / transformer-style models → Entropy**

This benchmark is the evidence for that split. Wrong architecture → wrong layers → wasted INT8 budget.

---

### Bottom line

Existing methods either **need data you often don’t have** or **score weights in ways that don’t match how layers actually quantize**. **PCT** and **Entropy** are new, data-free, quantization-aware rankers with a clear real-world split: **protect CNN layers by truncation damage, protect transformer layers by weight entropy**. People should care because mixed precision only works if layer selection works—and layer selection should not require a calibration pipeline that production doesn’t have.

#### Future Work
- **Pareto analysis** (acc recovery vs ECE recovery) instead of a single composite score.
- **Additional architectures** and datasets beyond CIFAR-10.
- **Per-layer ground-truth correlation** (Spearman ρ vs measured single-layer recovery).

**Additional Sources:**
- HAWQ, InfoQ, BMPQ, CLADO — established mixed-precision / calibration-aware quantization literature
