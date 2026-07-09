# Working notes — Implementation 2

## Goal

Add an ML feature layer on top of the Implementation 1 baseline that **quantitatively improves** agent efficiency — fewer steps, fewer irrelevant clicks — without changing the underlying LLM.

## Hypothesis

Browser agents waste steps because every interactive element looks equally important in the flat DOM list the LLM sees. A lightweight relevance scorer can rank elements by task similarity and guide the LLM toward better first clicks.

## Approach: Action Relevance Prior (ARP)

- **Model:** TF-IDF + cosine similarity (no GPU, no extra API calls)
- **Features per element:** tag, visible text, aria-label, placeholder, title
- **Integration:** prepend top-k ranked elements to the observation string
- **Evaluation:** compare baseline click paths vs ARP-guided paths on fixed tasks

Chose TF-IDF over embeddings first because it's fast, interpretable, and runs offline — good for iterating before investing in heavier models.

## Design decisions

| Decision | Rationale |
|----------|-----------|
| Page-local TF-IDF fit | IDF adapts to each page's vocabulary (e.g. "Apply" matters more on admissions pages) |
| top_k = 3–5 | Enough signal without flooding the context window |
| Inject hints, don't filter | Safer — LLM can still pick anything; we just bias attention |
| Offline fixtures first | Impl 1 showed API keys + Playwright were still blocked; offline eval unblocks ML work |

## Progress

- [x] Define 5 benchmark tasks from Impl 1 observations (`benchmark/tasks.json`)
- [x] Build trajectory logger for step-level metrics
- [x] Implement TF-IDF relevance scorer (`src/action_relevance.py`)
- [x] Add observation enhancer + `integrate_arp.py` hook
- [x] Create offline eval harness with 3 sample pages
- [x] Unit tests for scorer ranking
- [x] First offline eval run — see results below
- [ ] Live browser eval on all 5 tasks (blocked on API quota + Playwright from Impl 1)

## Results — offline eval (2026-07-08)

Ran `./run_eval.sh` on 3 sample pages (UTD admissions, Wikipedia Python, HN top story).

| Task | Baseline steps | ARP steps | Irrelevant clicks saved |
|------|---------------|-----------|-------------------------|
| utd_admissions | 5 | 2 | 4 → 1 |
| wiki_python | 4 | 2 | 2 → 0 |
| hn_top_story | 4 | 1 | 3 → 1 |

**Aggregate:** 61.5% fewer steps, 77.8% fewer irrelevant clicks, 100% success both modes.

ARP paths are shorter because the scorer surfaces "Admissions", "Apply Now", "Python (programming language)", and the top HN story link in the injected hint block — the simulated agent follows those instead of wandering through nav chrome.

## Bugs fixed

- **Success rate was 0% for ARP** in first eval run — success was incorrectly tied to `len(clicks) >= success_after` (baseline step count). Fixed to check whether `goal_element` appears in the click sequence.

## Open questions

- Does ARP help on pages with very sparse text (icon-only buttons)?
- Will sentence-transformers generalize better than TF-IDF on paraphrased tasks?
- How much does the hint block add to token cost per step?

## Next

Run live eval once Impl 1 blockers are cleared. Compare token usage and wall-clock time in addition to steps.
