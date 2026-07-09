# ARP vs Baseline — Offline Evaluation Results

**Date:** 2026-07-08  
**Eval type:** Offline sample pages (3 of 5 benchmark tasks)  
**Model:** TF-IDF + cosine similarity, top_k=3

## Summary

| Metric | Baseline | ARP | Improvement |
|--------|----------|-----|-------------|
| Success rate | 100% (3/3) | 100% (3/3) | — |
| Avg steps | 4.3 | 1.7 | **−61.5%** |
| Avg irrelevant clicks | 3.0 | 0.7 | **−77.8%** |

ARP reaches the same task goals with far fewer steps and almost no wasted navigation clicks.

## Per-task breakdown

### utd_admissions
- **Task:** Find when UTD admissions open for fall semester
- Baseline: 5 steps, 4 irrelevant clicks (wandered Home → Academics → About → Admissions)
- ARP: 2 steps, 1 irrelevant click (Admissions → Apply Now directly)

### wiki_python
- **Task:** Find when Python was first released on Wikipedia
- Baseline: 4 steps, 2 irrelevant clicks
- ARP: 2 steps, 0 irrelevant clicks (search box → Python article)

### hn_top_story
- **Task:** Read the title of the top Hacker News story
- Baseline: 4 steps, 3 irrelevant clicks (header, nav, login before story)
- ARP: 1 step, 1 irrelevant click (top story link immediately)

## Interpretation

The ARP layer surfaces task-relevant elements in a compact hint block before the LLM sees the full DOM. In these fixtures, that is enough to skip most exploratory navigation.

This is **offline** evaluation on simplified DOM snapshots. Live browser runs may differ due to dynamic content, larger pages, and LLM non-determinism. Next step: rerun on live web-ui once API keys and Playwright are working from Implementation 1.

## How to reproduce

```bash
cd implementation_2
./run_eval.sh
```

Raw data: `results/comparison.json`, per-run logs in `results/runs/` (gitignored).
