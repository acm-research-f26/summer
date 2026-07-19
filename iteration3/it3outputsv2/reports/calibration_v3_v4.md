# VLM calibration: v3 vs v4

Generated 2026-07-19 19:33 · 546 screens scored by both prompts (0 scored only by v4, excluded here) · same model, same images, same parser — only the prompt's calibration block differs.

## Verdict

- **Hedging DOWN**: category-score mass in [0.45, 0.55] went 10.0% → 0.1% (exactly 0.5: 10.0% → 0.1%).
- **Severity spread WIDER**: std 0.121 → 0.191, range 0.499 → 0.688, max 0.499 → 0.688.
- **Detection guardrail HELD**: macro-F1 vs ground truth 0.345 → 0.368 at threshold 0.5 (calibration should move confidence, not correctness; a drop beyond 0.05 would mean the model got confidently wrong).

## Hedge mass (share of category scores at/near 0.5)

| category | v3_at_0.5 | v3_in_band | v3_std | v4_at_0.5 | v4_in_band | v4_std |
|---|---|---|---|---|---|---|
| urgency | 0.051 | 0.051 | 0.337 | 0.000 | 0.000 | 0.341 |
| scarcity | 0.117 | 0.117 | 0.214 | 0.000 | 0.000 | 0.292 |
| social_proof | 0.101 | 0.101 | 0.165 | 0.000 | 0.000 | 0.225 |
| guilt_wording | 0.068 | 0.068 | 0.131 | 0.000 | 0.000 | 0.215 |
| other | 0.161 | 0.161 | 0.191 | 0.004 | 0.004 | 0.245 |
| ALL | 0.100 | 0.100 | 0.226 | 0.001 | 0.001 | 0.270 |

## Severity_vision distribution

| score | count | mean | std | min | 25% | 50% | 75% | max | range |
|---|---|---|---|---|---|---|---|---|---|
| severity_v3 | 546.000 | 0.164 | 0.121 | 0.000 | 0.000 | 0.190 | 0.250 | 0.499 | 0.499 |
| severity_v4 | 546.000 | 0.198 | 0.191 | 0.000 | 0.000 | 0.160 | 0.400 | 0.688 | 0.688 |

## Detection vs ground truth at threshold 0.5 (guardrail)

| category | n_gt | v3_precision | v3_recall | v3_f1 | v4_precision | v4_recall | v4_f1 |
|---|---|---|---|---|---|---|---|
| urgency | 51 | 0.333 | 0.961 | 0.495 | 0.389 | 0.863 | 0.537 |
| scarcity | 27 | 0.267 | 0.889 | 0.410 | 0.256 | 0.815 | 0.389 |
| social_proof | 16 | 0.111 | 0.438 | 0.177 | 0.170 | 0.500 | 0.254 |
| guilt_wording | 12 | 0.211 | 0.667 | 0.320 | 0.256 | 0.917 | 0.400 |
| other | 205 | 0.522 | 0.234 | 0.323 | 0.607 | 0.166 | 0.261 |

![calibration](calibration_v3_v4.png)
