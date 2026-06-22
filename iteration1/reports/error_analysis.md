# Error analysis — multi-class DistilBERT (test split)

## Top-3 most-confused class pairs

- **Urgency ↔ Other** — 1 off-diagonal (0 true Urgency → Other, 1 true Other → Urgency)
- **Scarcity ↔ Social Proof** — 1 off-diagonal (0 true Scarcity → Social Proof, 1 true Social Proof → Scarcity)
- **Scarcity ↔ Other** — 1 off-diagonal (1 true Scarcity → Other, 0 true Other → Scarcity)

## Sample misclassifications per pair (up to 10 each)

### Urgency ↔ Other

| true | pred | text | probs |
|---|---|---|---|
| Other | Urgency | LIMITED TIME OFFER : Grab a Matching Pair & save up 60%! | Urgency=0.79, Scarcity=0.04, Social Proof=0.03, Guilt-wording=0.04, Other=0.10 |

### Scarcity ↔ Social Proof

| true | pred | text | probs |
|---|---|---|---|
| Social Proof | Scarcity | 143 BOUGHT | Urgency=0.05, Scarcity=0.43, Social Proof=0.38, Guilt-wording=0.05, Other=0.08 |

### Scarcity ↔ Other

| true | pred | text | probs |
|---|---|---|---|
| Scarcity | Other | An item you ordered is in high demand. No worries, we have reserved your order. | Urgency=0.08, Scarcity=0.20, Social Proof=0.15, Guilt-wording=0.17, Other=0.39 |
