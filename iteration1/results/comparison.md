# Model comparison (test split)

| task       | model                 |   accuracy |   macro_f1 |   weighted_f1 |
|:-----------|:----------------------|-----------:|-----------:|--------------:|
| binary     | baseline              |     0.9377 |     0.9377 |        0.9377 |
| multiclass | baseline              |     0.9718 |     0.9513 |        0.9719 |
| binary     | distilbert            |     0.9688 |     0.9688 |        0.9688 |
| multiclass | distilbert_unweighted |     0.9718 |     0.9468 |        0.9696 |
| multiclass | distilbert_weighted   |     0.9831 |     0.9708 |        0.9829 |

## Validity caveats — read before quoting numbers

1. **`is_dark` is an upper bound — register-contaminated.** dataset.tsv
   negatives are page-chrome HTML fragments ("Pillowcases & Shams", "Write a
   review") while positives are marketing-style copy, so high binary accuracy
   may reflect a register/style detector, not dark-pattern understanding. The
   category model is the more meaningful artifact; treat `is_dark` as a
   coarse gate, not as evidence the model "understands manipulation".

2. **Social Proof F1 is likely optimistic via template memorization.**
   Exact-string dedup does NOT catch templated near-duplicates of the form
   *"Name from City just bought Product about N hours ago"*. The model can
   memorize the template skeleton, inflating Social Proof precision/recall.
   Treated as a known limitation for iter 2 (page-id-level holdout).
