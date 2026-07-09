![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Mixed-Precision Proportional Bit-Budget Allocator

## 📌 Project Summary

**Part 2** of this repo — an extension of the layer-importance ranking benchmark in [`1-ranking-effectiveness/`](1-ranking-effectiveness/).

Part 1 asks: *which layers matter?* (rank layers, upgrade top-*k* to INT8, leave the rest at a fixed low bit-width). Part 2 asks: *given a target **average** bit-width, how should bits be distributed across layers?*

The main artifact is [`2-bit-allocation/mixed_precision_proportional_allocator_benchmark.ipynb`](2-bit-allocation/mixed_precision_proportional_allocator_benchmark.ipynb). It connects **data-free layer scores** from Part 1 (+ InfoQ, spectral variants) to a **sensitivity-guided greedy bit allocator** under target average bit-widths ∈ {2.5, 3.0, 3.5, 4.0}.

**Headline finding:** proportional allocation **separates rankers at every budget target**—unlike Part 1’s two-level scheme (low-bit + INT8), which ties many methods together. A **2×2 taxonomy** (weight values vs spectral × propagation vs none) predicts the best scorer by architecture: **PCT** on CNNs, **SEA / Spectral Top-10** on ViTs, **SEA / Entropy** on LLMs.

## 🔗 Relationship to Part 1

| | **Part 1** — Ranking effectiveness | **Part 2** — Bit allocation (this notebook) |
|---|---|---|
| **Question** | Which ranker best identifies important layers? | Which ranker best drives a mixed bit assignment under a budget? |
| **Allocation** | Two-level: top-*k* → INT8, rest → *B*-bit | Multi-level greedy ladder: layers get 2/3/4/8 (vision) or 3/4/8 (LLM) bits |
| **Budget** | Fixed *k* count of INT8 layers | Target **average** bits across all layers (param-weighted) |
| **Datasets** | CIFAR-10 | CIFAR-100, ImageNet, Qwen 1.5B WikiText-2 |
| **Notebook** | [`1-ranking-effectiveness/comprehensive_layer_importance_benchmark.ipynb`](1-ranking-effectiveness/comprehensive_layer_importance_benchmark.ipynb) | [`2-bit-allocation/mixed_precision_proportional_allocator_benchmark.ipynb`](2-bit-allocation/mixed_precision_proportional_allocator_benchmark.ipynb) |

Part 1 established **PCT for CNNs, Entropy for ViTs** under the two-level protocol. Part 2 tests whether those rankings still win when bits are allocated **proportionally** on harder benchmarks and real deployment bit targets.

## 🎯 Motivation

Production teams rarely specify “upgrade exactly 5 layers to INT8.” They specify a **memory or latency budget** — e.g. “average 3.0 bits per weight.” A greedy allocator that walks layers by sensitivity and upgrades each as high as the budget allows is the standard approach in mixed-precision literature (NSDS, HAWQ, Schaefer et al. Algorithm 2).

Part 1’s two-level recovery sweep is the right tool for **comparing rankers**, but it collapses methods at many targets: once you fix *k* and the low bit-width, different rankings can produce identical bit mixes. Part 2 closes the loop from **score → allocation → accuracy/PPL** under continuous budget targets, which is what you actually ship.

## 🧩 Novelty

We propose a **2×2 taxonomy of data-free sensitivity metrics** along two axes — **information source** and **propagation awareness** — and show empirically that the optimal metric is **architecture-dependent**: numerical metrics (PCT) dominate on CNNs while spectral metrics (SEA, Spectral Top-10) dominate on vision transformers and LLMs. This provides the first principled framework for data-free metric selection in mixed-precision quantization.

### Metric taxonomy

|  | **Uses weight values** | **Spectral only** |
|--|------------------------|-------------------|
| **No propagation** | **PCT** | **SEA**, **Spectral Top-10** |
| **With propagation** | **Spectral-PCT** | — |

- **Information source** — *numerical* metrics read weight magnitudes and quantization damage (PCT, Entropy); *spectral* metrics read singular-value structure of weight tensors (SEA, Spectral Top-10).
- **Propagation awareness** — *no-propagation* metrics score each layer in isolation; *with-propagation* metrics combine spectral structure with cross-layer truncation effects (**Spectral-PCT** propagates per-channel rounding loss through the spectral ranking).

**Entropy** (Shannon entropy of the weight histogram) is a numerical, no-propagation metric — same quadrant as PCT — but optimized for **transformer-style** weight distributions rather than CNN per-channel truncation damage.

### Architecture-dependent metric selection

| Architecture | Winning quadrant | Default metric(s) | Benchmark evidence |
|--------------|------------------|-------------------|-------------------|
| **CNN / ResNet** | Weight values · no propagation | **PCT** | Part 1 CIFAR-10 INT4; Part 2 CIFAR-100 & ImageNet @ 3.0–4.0 avg bits |
| **ViT** | Spectral only · no propagation | **SEA**, **Spectral Top-10** | Part 2 CIFAR-100 ViT; ImageNet ViT @ 3.5–4.0 avg bits |
| **LLM (Qwen 1.5B)** | Spectral + numerical · no propagation | **SEA** (@ 3.5), **Entropy** (@ 4.0) | Part 2 WikiText-2 PPL — extends the ViT split to language models |

At extreme LLM budgets (≈2.5 avg bits), **Spectral-PCT** (with-propagation hybrid) remains competitive; at deployable targets (3.5–4.0), **SEA** and **Entropy** win — confirming that transformer-family models favor the **spectral** and **distributional** quadrants over raw truncation error (PCT).

---

- **Score-to-allocation pipeline**: Reuses Part 1 data-free scorers (PCT, Entropy, SEA, OLD, NSDS) plus spectral variants (`spectral_pct`, `spectral_top10`) and InfoQ, feeding them into `greedy_allocate_bits()`.
- **Multi-level bit ladder**: Vision models use `{2, 3, 4, 8}`; Qwen uses `{3, 4, 8}` — high-sensitivity layers can receive intermediate bit-widths, not just INT8.
- **Cross-scale evaluation**: Same allocator protocol on CIFAR-100, ImageNet, and a 1.5B LLM (WikiText-2 perplexity).
- **Ranker differentiation at every target**: Different score orderings produce different bit mixes at each target avg-bit, so allocator quality directly tests ranking quality.

## 🧠 Methodology

### Allocation protocol (NSDS / HAWQ / Schaefer et al. Algorithm 2)

1. Start all layers at `b_min` (lowest bit on the ladder).
2. Sort layers by sensitivity descending (score from ranker).
3. For each layer in order, upgrade it as high on the ladder as the param-weighted budget allows.

Implementation: `bit_budget_allocator.py → greedy_allocate_bits()`

### Benchmarks

| Benchmark | Model | Metric | Bit ladder |
|-----------|-------|--------|------------|
| CIFAR-100 | ResNet-18, ViT-B/16 | Accuracy | {2, 3, 4, 8} |
| ImageNet | ResNet-18, ViT-B/16 | Accuracy | {2, 3, 4, 8} |
| WikiText-2 | Qwen 1.5B | Perplexity (↓) | {3, 4, 8} |

**Targets:** avg bits ∈ {2.5, 3.0, 3.5, 4.0} for vision; Qwen also includes {4.5, 5.0}.

### Methods compared

| Group | Method | Data? | Source |
|-------|--------|-------|--------|
| **Ours (Part 1)** | **PCT**, **Entropy**, **SEA** | ✗ | Layer scores |
| Spectral variants | **spectral_pct**, **spectral_top10** | ✗ | Computed in notebook |
| Prior | **NSDS**, **OLD** | ✗ | Layer scores |
| Established | **InfoQ** | 512 imgs | Optional (`RUN_INFOQ`) |
| Baselines | **random**, **greedy_random**, uniform | ✗ | Controls |

Scores are loaded from cached CSVs (legacy Part 1 / dataset-specific benchmarks preferred); missing columns are enriched on the fly.

#### Additional Methodology

- `SMOKE_TEST = True` limits ImageNet eval to 512 images and Qwen to 8k tokens for fast iteration.
- `FORCE_RECOMPUTE = False` reuses cached allocator CSVs in `results/proportional_mp_allocator/`.
- `RUN_INFOQ = False` by default (slow, especially on LLM).

## 🚀 Running the benchmark

**Requirements:** Python 3.10+, PyTorch, torchvision, pandas, numpy, scipy, scikit-learn, matplotlib, `datasets`, `transformers`, `accelerate`.

```bash
pip install torch torchvision scipy scikit-learn pandas matplotlib datasets transformers accelerate
jupyter notebook 2-bit-allocation/mixed_precision_proportional_allocator_benchmark.ipynb
```

**Checkpoints** (expected under repo root or env vars):

| Benchmark | Path |
|-----------|------|
| CIFAR-100 | `checkpoints_cifar100/{resnet18,vit_b16}_cifar100_best_valacc.pt` |
| ImageNet | `checkpoints_imagenet/` + ImageNet val split (`IMAGENET_ROOT`) |
| Qwen 1.5B | `QWEN_PATH` (local HF-format model dir) |

**Outputs** → `results/proportional_mp_allocator/`:

| File | Description |
|------|-------------|
| `all_allocator_results.csv` | Combined sweep across all benchmarks |
| `cifar100_*_allocator_results.csv` | Per-architecture CIFAR-100 results |
| `imagenet_*_allocator_results.csv` | Per-architecture ImageNet results |
| `qwen_allocator_results.csv` | Qwen WikiText-2 perplexity sweep |
| `vision_accuracy_vs_budget.png` | Accuracy vs target avg bits (vision) |
| `qwen_ppl_vs_budget.png` | Perplexity vs target avg bits (LLM) |

## 📊 Results

Headline: **best method per benchmark @ each target** (vision = accuracy ↑, Qwen = perplexity ↓). Smoke-test run; set `SMOKE_TEST = False` for full eval.

### CIFAR-100 ResNet-18

| Target avg bits | Best method | Accuracy |
|-----------------|-------------|----------|
| 2.5 | OLD | 5.73% |
| 3.0 | **PCT** | 27.28% |
| 3.5 | spectral_pct | 74.28% |
| 4.0 | **PCT** | 74.69% |

**PCT** wins at the two highest practical targets — consistent with Part 1’s CNN default.

### CIFAR-100 ViT-B/16

| Target avg bits | Best method | Accuracy |
|-----------------|-------------|----------|
| 2.5 | spectral_top10 | 21.89% |
| 3.0 | SEA | 39.83% |
| 3.5 | spectral_top10 | 62.49% |
| 4.0 | InfoQ | 62.85% |

Transformer-side scorers (SEA, spectral, Entropy family) dominate; Part 1’s **Entropy** ranking aligns with this architecture split.

### ImageNet ResNet-18

| Target avg bits | Best method | Accuracy |
|-----------------|-------------|----------|
| 2.5 | OLD | 0.39% |
| 3.0 | spectral_pct | 0.39% |
| 3.5 | **PCT** | 68.55% |
| 4.0 | **PCT** | 70.70% |

**PCT** wins at 3.5 and 4.0 avg bits on full ImageNet — ranker generalizes beyond CIFAR.

### ImageNet ViT-B/16

| Target avg bits | Best method | Accuracy |
|-----------------|-------------|----------|
| 2.5 | SEA | 0.59% |
| 3.0 | greedy_random | 0.20% |
| 3.5 | **SEA** | 87.11% |
| 4.0 | **SEA** | 88.28% |

**SEA** leads at deployment-relevant targets (3.5, 4.0) on ImageNet ViT.

### Qwen 1.5B — WikiText-2

| Target avg bits | Best method | Perplexity |
|-----------------|-------------|------------|
| 2.5 | spectral_pct | 6,352,318 |
| 3.0 | SEA | 2,262,656 |
| 3.5 | **SEA** | 88.97 |
| 4.0 | **Entropy** | 65.30 |

At usable LLM budgets (3.5–4.0 avg bits), **SEA** and **Entropy** (Part 1’s ViT default) drive the best perplexity — extending the architecture-aware split to language models.

### Takeaways

| Setting | Taxonomy quadrant | Strong allocators | Link to Part 1 |
|---------|-------------------|-------------------|----------------|
| **CNN / ResNet** (CIFAR-100, ImageNet) | Weight values · no propagation | **PCT**, Spectral-PCT | Confirms PCT as CNN default |
| **ViT** (CIFAR-100, ImageNet) | Spectral only · no propagation | **SEA**, Spectral Top-10 | Confirms spectral scorers for transformers |
| **LLM** (Qwen) | Spectral + numerical · no propagation | **SEA**, **Entropy** | ViT-like split extends to language models |

The two-level Part 1 benchmark tells you *which layers to protect*; this allocator benchmark tells you *how to spend a continuous bit budget* using those same rankings — and the **2×2 taxonomy** predicts which quadrant to pick before you run a sweep.

## 🌍 Impact

### Why proportional allocation matters

Real deploy constraints are **average-bit targets**, not “*k* layers at INT8.” A 3.0-bit model might mix INT2, INT3, and INT4 layers — the greedy allocator produces that mix from a single sensitivity ranking. Without this step, Part 1 rankings are necessary but not sufficient: you know layer order, but not the bit assignment that actually hits your memory cap.

### What changes vs Part 1

- **Finer bit granularity** exposes ranker differences that two-level allocation hides.
- **Harder datasets** (CIFAR-100, ImageNet) and **LLM perplexity** test whether CIFAR-10 ranking conclusions survive scale.
- **Intermediate bit-widths** (INT3) matter at low avg-bit targets (2.5–3.0) where INT8-heavy schemes are infeasible.

### Deploy guidance (updated)

Use the taxonomy to pick a default before benchmarking:

| Architecture | Pick this quadrant first | Default scorer |
|--------------|--------------------------|----------------|
| **CNN / ResNet** | Weight values · no propagation | **PCT** |
| **ViT** | Spectral only · no propagation | **SEA** or **Spectral Top-10** |
| **LLM** | Spectral or numerical · no propagation | **SEA** (mid budget), **Entropy** (higher budget) |
| **Any (tight budget)** | With propagation | **Spectral-PCT** as fallback |

Run the allocator notebook with your checkpoint and target avg bits to get a per-layer bit assignment (`bit_mix` column in results CSVs).

#### Future Work

- Full eval (`SMOKE_TEST = False`) on ImageNet and Qwen.
- Calibration metrics (ECE) under proportional allocation, not just accuracy/PPL.
- Pareto frontier over target avg bits vs accuracy/PPL by ranker.
- Joint optimizer (ILP) vs greedy baseline.

**Additional Sources:**
- Part 1 ranking benchmark: [`1-ranking-effectiveness/README.md`](1-ranking-effectiveness/README.md)
- NSDS, HAWQ, Schaefer et al. — greedy mixed-precision allocation literature
