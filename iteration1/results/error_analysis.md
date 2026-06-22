# Error analysis — multi-class DistilBERT (test split)

## Top-3 most-confused class pairs

- **Urgency ↔ Other** — 1 off-diagonal (0 true Urgency → Other, 1 true Other → Urgency)
- **Scarcity ↔ Other** — 1 off-diagonal (1 true Scarcity → Other, 0 true Other → Scarcity)
- **Guilt-wording ↔ Other** — 1 off-diagonal (0 true Guilt-wording → Other, 1 true Other → Guilt-wording)

## Sample misclassifications per pair (up to 10 each)

### Urgency ↔ Other

| true | pred | text | probs |
|---|---|---|---|
| Other | Urgency | LIMITED TIME OFFER : Grab a Matching Pair & save up 60%! | Urgency=0.79, Scarcity=0.04, Social Proof=0.03, Guilt-wording=0.04, Other=0.10 |

### Scarcity ↔ Other

| true | pred | text | probs |
|---|---|---|---|
| Scarcity | Other | An item you ordered is in high demand. No worries, we have reserved your order. | Urgency=0.08, Scarcity=0.20, Social Proof=0.14, Guilt-wording=0.18, Other=0.40 |

### Guilt-wording ↔ Other

| true | pred | text | probs |
|---|---|---|---|
| Other | Guilt-wording | I wish to Start Shopping but I don't want to receive daily email alerts about your amazing discounts on luxury brands. | Urgency=0.06, Scarcity=0.06, Social Proof=0.07, Guilt-wording=0.50, Other=0.32 |
