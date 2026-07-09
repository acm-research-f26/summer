# Implementation 2 — Action Relevance Prior (ARP)

This folder builds on [Implementation 1](../implementation_1/) by adding a lightweight **ML layer** that ranks interactive page elements by task relevance, then injects top-ranked hints into the agent's observation before the LLM chooses an action.

## Research question

Can a semantic relevance prior reduce wasted clicks and total steps in browser-agent trajectories **without replacing the LLM**?

## What ARP does

1. **Score** — TF-IDF + cosine similarity between the task prompt and each interactive element (text, tag, aria-label).
2. **Rank** — Keep top-k elements per page.
3. **Inject** — Prepend a short relevance summary to the agent's state message so the LLM prioritizes high-scoring targets.

ARP does not block or override LLM decisions. It only augments context.

## Project layout

```
implementation_2/
├── benchmark/
│   ├── tasks.json          # 5 fixed evaluation tasks
│   └── sample_pages.py     # Offline DOM fixtures + simulated trajectories
├── src/
│   ├── action_relevance.py # Core ML scorer
│   ├── observation_enhancer.py
│   └── trajectory_logger.py
├── eval/
│   ├── run_offline_eval.py # Baseline vs ARP comparison
│   └── compare_results.py
├── tests/
│   └── test_action_relevance.py
├── integrate_arp.py        # Optional hook into browser-use
├── run_eval.sh
└── results/
    ├── comparison.json
    └── comparison.md
```

## Prerequisites

- Python 3.11+ (same venv as web-ui is fine)
- `scikit-learn` and `numpy` (see `requirements.txt`)

```bash
pip install -r implementation_2/requirements.txt
```

## Run offline evaluation

No browser or API keys required — uses sample page fixtures from Implementation 1 tasks.

```bash
cd implementation_2
chmod +x run_eval.sh
./run_eval.sh
```

Or directly:

```bash
python eval/run_offline_eval.py
python tests/test_action_relevance.py
```

Results are written to `results/comparison.json`.

## Enable ARP in live web-ui (optional)

From the web-ui repo root, after installing deps:

```python
from implementation_2.integrate_arp import enable_arp_layer
enable_arp_layer(top_k=5)
# then run the agent as usual via webui.py
```

## Results summary (offline eval, 3 tasks)

| Metric | Baseline | ARP | Change |
|--------|----------|-----|--------|
| Avg steps | 4.3 | 1.7 | **−61.5%** |
| Irrelevant clicks | 3.0 | 0.7 | **−77.8%** |
| Success rate | 100% | 100% | same |

See `results/comparison.md` for per-task breakdown and notes.

## Next steps

- [ ] Run live eval on all 5 benchmark tasks once API keys + Playwright are working
- [ ] Try sentence-transformers upgrade path for cross-page generalization
- [ ] Log real trajectories from web-ui into `results/runs/`
