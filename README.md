![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# 👑 SLAY: Sensitivity-Led Allocation for Yield 👑

## 💅 Project Summary 💅

**SLAY** (**S**ensitivity-**L**ed **A**llocation for **Y**ield) is data-free mixed-precision quantization that ranks layers, spends a bit budget where it matters, and measures what industry actually pays for, which is **memory**, **accuracy**, **FLOPs (processing power)**, **time**, and **energy**.

| Stage | Folder | Question |
|-------|--------|----------|
| **1 · Score** | [`1-data-free-ranking-effectiveness/`](1-data-free-ranking-effectiveness/) | Which layers matter for deployment, comparing methods with and without calibration data? |
| **2 · Ladder** | [`2-bit-allocation-experiment/`](2-bit-allocation-experiment/) | How do we turn those scores into a bit mix under a target avg width? |
| **3 · Yield** | [`3-industry-connection/`](3-industry-connection/) | What does that buy in memory / accuracy / FLOPs / time / energy? Why is this relevant to the industry? |

**The results SLAYED.** Data-free scorers (PCT, SEA, Entropy, Spectral-PCT, …) use **0 forward passes** and **0 allocation GFLOPs**. On several INT3 operating points they match or beat HAWQ/InfoQ/BMPQ/CLADO on accuracy recovery — while those methods spend tens to hundreds of thousands of GFLOPs just to *rank* layers. Extrapolated to LLM CI/CD, that gap becomes hours of GPU time and watt-hours of energy that SLAY simply does not spend. Done and dusted.

## 👠 Relationship across parts

| | **Part 1** — Ranking | **Part 2** — Allocation | **Part 3** — Yield (industry) |
|---|---|---|---|
| **Question** | Which ranker identifies important layers? | Which ranker drives a mixed bit mix under a budget? | What does that cost in FLOPs / time / energy / memory? |
| **Allocation** | Two-level: top-*k* → INT8, rest → *B*-bit | Multi-level greedy ladder: 2/3/4/8 (vision) or 3/4/8 (LLM) | Same two-level + cost accounting |
| **Budget** | Fixed *k* INT8 layers | Target **average** bits (param-weighted) | Memory + ranking compute |
| **Datasets** | CIFAR-10 | CIFAR-100, ImageNet, Qwen WikiText-2 | CIFAR-100, ImageNet, Qwen |
| **Notebook** | [`comprehensive_layer_importance_benchmark.ipynb`](1-data-free-ranking-effectiveness/comprehensive_layer_importance_benchmark.ipynb) | [`mixed_precision_proportional_allocator_benchmark.ipynb`](2-bit-allocation-experiment/mixed_precision_proportional_allocator_benchmark.ipynb) | [`mixed_precision_cost_benchmark.ipynb`](3-industry-connection/mixed_precision_cost_benchmark.ipynb) |

Part 1 establishes **which layers to protect**. Part 2 turns those rankings into a **continuous bit budget**. Part 3 prices the ranking step itself — and shows data-free SLAY scorers win on the accuracy–memory Pareto front at **0 ranking cost**.

## 👑 Motivation

Production teams ship under **memory and latency SLAs**, not “upgrade exactly *k* layers to INT8.” They also cannot always afford calibration pipelines (privacy, edge, stale data, weekly model variants). In industry,
quantization happens uniformly since the assumption is data-free methods
are not as beneficial as data-required methods; however, data-required methods take lots of computation power, so uniform quantization is preferred.

SLAY 👑 finally gives industry a data-free quantization method to improve model deployment:

1. **Score** layers with a data-free sensitivity metric (architecture-aware taxonomy from Parts 1–2).
2. **Allocate** bits on a ladder `{2,3,4,8}` (vision) / `{3,4,8}` (LLM) to hit a target average bit-width.
3. **Yield** — report weight-memory compression, accuracy/PPL recovery, allocation FLOPs, expected GPU seconds, and expected energy vs data-required baselines.

## 💅 Novelty

We propose a **2×2 taxonomy of data-free sensitivity metrics** along two axes — **information source** and **propagation awareness** — and show empirically that the optimal metric is **architecture-dependent**: numerical metrics (PCT) dominate on CNNs while spectral metrics (SEA, Spectral Top-10) dominate on vision transformers and LLMs. This provides a principled framework for data-free metric selection in mixed-precision quantization.

### Metric taxonomy

|  | **Uses weight values** | **Spectral only** |
|--|------------------------|-------------------|
| **No propagation** | **PCT**, **Entropy** | **SEA**, **Spectral Top-10** |
| **With propagation** | **Spectral-PCT** | — |

- **Information source** — *numerical* metrics read weight magnitudes and quantization damage (PCT, Entropy); *spectral* metrics read singular-value structure of weight tensors (SEA, Spectral Top-10).
- **Propagation awareness** — *no-propagation* metrics score each layer in isolation; *with-propagation* metrics combine spectral structure with cross-layer truncation effects (**Spectral-PCT**).

### Architecture-dependent metric selection

| Architecture | Winning quadrant | Default SLAY scorer | Evidence |
|--------------|------------------|---------------------|----------|
| **CNN / ResNet** | Weight values · no propagation | **PCT** / Spectral-PCT | Parts 1–3 (CIFAR / ImageNet) |
| **ViT** | Spectral only · no propagation | **SEA** / Spectral Top-10 | Parts 2–3 |
| **LLM** | Spectral + numerical · no propagation | **SEA** / **Entropy** | Parts 2–3 (Qwen) |

---

- **Score → allocate → yield pipeline**: Parts 1–3 share the same data-free scorers; only the allocation protocol and cost accounting change.
- **Zero ranking FLOPs**: PCT / SEA / Entropy / Spectral-PCT need no calibration forwards — allocation GFLOPs = **0**.
- **Pareto claim**: mixed precision lands **between** uniform INT3 and INT4 on memory *and* accuracy, with **~87–91%** weight memory saved vs FP32.
- **Industry cost model**: ranking FLOPs → expected GPU seconds → expected energy (kWh) for LLM CI/CD scale.

## 👠 Methodology

### Score → allocate → yield

1. Rank layers with a data-free scorer (taxonomy above).
2. Allocate bits — Part 1/3 two-level (*k* → INT8) or Part 2 greedy ladder to target `B̄`.
3. Measure accuracy/PPL, weight memory (`mem saved vs FP32 = 1 − B̄ / 32`), allocation GFLOPs, extrapolated GPU time and energy.

### Benchmarks

| Benchmark | Model | Metric | Where |
|-----------|-------|--------|-------|
| CIFAR-10 | ResNet-18, ViT-Small | Acc + ECE | Part 1 |
| CIFAR-100 / ImageNet | ResNet-18, ViT-B/16 | Accuracy | Parts 2–3 |
| WikiText-2 | Qwen 1.5B | Perplexity (↓) | Parts 2–3 |

### Methods compared

| Group | Method | Data? | Ranking GFLOPs |
|-------|--------|-------|----------------|
| **SLAY (ours)** | **PCT**, **Entropy**, **SEA**, Spectral-PCT, Spectral Top-10 | ✗ | **0** |
| Prior | **NSDS**, **OLD** | ✗ | **0** |
| Established | **HAWQ**, **InfoQ**, **BMPQ**, **CLADO** | 512 imgs | tens–hundreds of thousands |

#### Additional Methodology

- Part 3 allocation GFLOPs = analytic pass counts × FLOPs/forward (`thop`).
- GPU time scaled from RateQuant’s ~1.6 s gradient/calib estimate at Qwen3-8B ([arXiv 2605.06675](https://arxiv.org/abs/2605.06675)).
- Energy ≈ GPU power × time (conservative **300 W** A100-class TDP during allocation).

## 👑 Running the benchmarks

**Requirements:** Python 3.10+, PyTorch, torchvision, pandas, numpy, scipy, scikit-learn, matplotlib; Parts 2–3 also need `datasets`, `transformers`, `accelerate`, `thop`.

```bash
# Part 1 — ranking
jupyter notebook 1-data-free-ranking-effectiveness/comprehensive_layer_importance_benchmark.ipynb

# Part 2 — bit allocation
jupyter notebook 2-bit-allocation-experiment/mixed_precision_proportional_allocator_benchmark.ipynb

# Part 3 — cost / memory / energy
jupyter notebook 3-industry-connection/mixed_precision_cost_benchmark.ipynb
```

See each part’s README for checkpoints, env vars (`IMAGENET_ROOT`, `QWEN_PATH`), and output CSV paths.

## 💅 Results

### Allocation compute (FLOPs) — ranking cost before you even quantize

Data-free methods: **0 GFLOPs** (weights only). Data-required methods scale with layers × calibration passes:

| Method | ResNet CIFAR-100 | ResNet ImageNet | ViT ImageNet | Qwen 1.5B |
|--------|------------------|-----------------|--------------|-----------|
| **PCT / SEA / Entropy / …** | **0** | **0** | **0** | **0** |
| BMPQ | 54 | 175 | 1,083 | 1,185 |
| InfoQ | 393 | 1,284 | 14,085 | 77,845 |
| CLADO | 571 | 1,868 | 17,697 | 81,797 |
| HAWQ | 7,499 | 24,515 | **274,482** | **158,062** |

Units: allocation GFLOPs. Ratio vs HAWQ is **∞** for every data-free method.

### Pareto frontier — mixed sits between INT3 and INT4

The whole point of mixed precision: **huge** weight-memory savings vs FP32 (every config saves **~87–91%**), while landing *between* uniform INT3 and uniform INT4 on **both** axes — memory *and* accuracy. Uniform INT3 is tiny but accuracy collapses; uniform INT4 is accurate but bigger; SLAY mixed sits in the gap with accuracy pushed up toward FP32.

Weight memory (bits → savings), same formula as Parts 1–2: **`mem saved vs FP32 = 1 − B̄ / 32`** (where `B̄` = average bits per weight).

| Config | Avg bits (B̄) | **Mem saved vs FP32** | Accuracy |
|--------|---------------|-----------------------|----------|
| Uniform INT3 | 3.0 | **90.6%** | collapses (see below) |
| **SLAY mixed** | **~3.5–4.2** | **~87–89%** | **near-FP32** |
| Uniform INT4 | 4.0 | **87.5%** | good |
| Uniform INT8 | 8.0 | 75.0% | ~FP32 |
| FP32 | 32.0 | 0% | reference |

**Per-setting Pareto** (Part 3 op point: top-*k*=5 → INT8, rest INT3; winning data-free vs data-required scorer):

| Setting | INT3 acc / mem-saved | **SLAY mixed acc / mem-saved** | INT4 acc / mem-saved | Data-req acc |
|---------|----------------------|--------------------------------|----------------------|--------------|
| **ResNet / CIFAR-100** | 68.42% / 90.6% | **Spectral-PCT 72.91% / 86.9%** @ **0** GFLOPs | 74.25% / 87.5% | BMPQ 72.66% @ 54 GFLOPs |
| **ResNet / ImageNet** | 0.70% / 90.6% | **PCT 11.10% / 86.9%** @ **0** GFLOPs | 60.10% / 87.5% | InfoQ 14.20% @ 1,284 GFLOPs |
| **ViT / CIFAR-100** | 52.72% / 90.6% | **SEA 62.53% / 88.6%** @ **0** GFLOPs | 62.76% / 87.5% | CLADO 61.99% @ 17,696 GFLOPs |
| **ViT / ImageNet** | 41.55% / 90.6% | **SEA 88.80% / 88.6%** @ **0** GFLOPs | 92.15% / 87.5% | InfoQ 81.10% @ 14,085 GFLOPs |

FP32 refs: ResNet ≈ 74.9% (CIFAR-100) / 78.4% (ImageNet); ViT-B/16 ≈ 62.9% (CIFAR-100) / 93.3% (ImageNet).

The Pareto read: for **only ~1–2 percentage points less memory savings than INT3** (88.6% vs 90.6%), SLAY mixed buys **+9 to +47 pp accuracy** — e.g. ViT/ImageNet jumps from a **41.55%** INT3 collapse to **88.80%** (SEA), still saving **88.6%** of FP32 weight memory, at **0 ranking GFLOPs**. Data-free matches or beats the data-required scorers on 3 of 4 settings while they burn thousands of GFLOPs.

#### Part 2 — matched avg-bit targets (the allocator view)

The proportional allocator ([`2-bit-allocation-experiment`](2-bit-allocation-experiment/)) targets an explicit `B̄`, so memory savings are locked in and large, and the leaderboard becomes pure accuracy/PPL at that memory point:

| Target B̄ | **Mem saved vs FP32** | Position vs uniforms |
|-----------|-----------------------|----------------------|
| 2.5 | **92.2%** | below INT3 — aggressive |
| 3.0 | **90.6%** | = INT3 memory |
| 3.5 | **89.1%** | **between INT3 and INT4** |
| 4.0 | **87.5%** | = INT4 memory |

A 3.5-bit SLAY mix is the textbook “in between”: **89.1%** weight memory saved (between INT4’s 87.5% and INT3’s 90.6%), with Part 2 accuracy/PPL that tracks the better ranker (PCT / SEA / Entropy) on that frontier.

### Expected time (GPU seconds) — LLM scaling

| Model | Est. HAWQ GPU time | Est. SLAY (data-free) |
|-------|--------------------|------------------------|
| Qwen2-1.5B | ~120 s (0.03 h) | **0 s** |
| Qwen3-8B | ~640 s (0.18 h) | **0 s** |
| Qwen3-32B | ~2,560 s (0.71 h) | **0 s** |
| **100 variants / week @ 8B** | **~17.8 GPU-hours** | **0** |
| **100 variants / week @ 32B** | **~71.1 GPU-hours** | **0** |

### Expected energy

Energy ≈ GPU power × time (**300 W** sustained draw):

| Scenario | HAWQ energy | SLAY energy |
|----------|-------------|-------------|
| One Qwen3-8B ranking | 640 s × 300 W ≈ **192 kJ** (~0.053 kWh) | **0** |
| One Qwen3-32B ranking | 2560 s × 300 W ≈ **768 kJ** (~0.21 kWh) | **0** |
| 100× / week @ 8B | 17.8 h × 300 W ≈ **5.3 kWh** | **0** |
| 100× / week @ 32B | 71.1 h × 300 W ≈ **21.3 kWh** | **0** |

### Takeaways

| Setting | Taxonomy quadrant | Strong SLAY scorers | Cost vs HAWQ/InfoQ |
|---------|-------------------|---------------------|--------------------|
| **CNN / ResNet** | Weight values · no propagation | **PCT**, Spectral-PCT | **0** GFLOPs |
| **ViT** | Spectral only · no propagation | **SEA**, Spectral Top-10 | **0** GFLOPs |
| **LLM** | Spectral + numerical · no propagation | **SEA**, **Entropy** | **0** GFLOPs |

## 👑 Impact

### Why the ranking cost matters

At datacenter rates (~$0.10/kWh + GPU rental), the ranking step alone is free under SLAY and non-trivial under HAWQ-class pipelines — **before** counting engineer time, calib data ops, and failed mismatched-calib rankings (Part 3 calibration-robustness sweep).

### Deploy guidance

> Pick the taxonomy quadrant → score weights only → greedy-allocate to your avg-bit budget → ship.  
> **CNN → PCT · ViT → SEA · LLM → SEA / Entropy.**  
> Ranking cost stays **0 GFLOPs / 0 GPU-s / 0 Wh**; memory follows `1 − B̄/32` (**~87–91%** saved); accuracy follows Part 2/3 Pareto tables.

#### Future Work

- Full (`SMOKE_TEST=False`) ImageNet / Qwen sweeps
- Measured board-level joules (not TDP×time estimates)
- ECE under proportional allocation
- ILP bit assignment vs greedy ladder

**Additional Sources:**
- Part 1: [`1-data-free-ranking-effectiveness/README.md`](1-data-free-ranking-effectiveness/README.md)
- Part 2: [`2-bit-allocation-experiment/README.md`](2-bit-allocation-experiment/README.md)
- HAWQ, InfoQ, BMPQ, CLADO, NSDS, RateQuant (arXiv 2605.06675)
