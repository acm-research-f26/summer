# Stimulus set for the behavioral experiment

10 screens · selected 2026-07-19 05:43 · seed 42

## Selection rule

Selected on **severity_vision** (the score the experiment validates), over the 546 screens with cached vision predictions. Hard constraints: at least one ground-truth guilt_wording screen and at least one purely-visual own-set screen (FIVE_*) — without them, two of the five categories would be absent from the experiment entirely (guilt_wording exists on 12 screens of the corpus, the visual-only tricks on 6). Remaining slots: 5 equal-width severity bins, 2 per bin, seeded, preferring unrepresented vision top-categories and platforms; shortfalls fill by max-min severity distance.

**Why the model chooses the stimuli:** severity for every stimulus is fixed by the frozen iteration-1/2 detectors plus the literature-derived weights *before* any behavior is measured. The fall regression of behavioral delta on severity is therefore a genuine out-of-sample prediction test of the score, not a fit to hand-picked examples.

## Chosen screens (low -> high severity)

| screen_id | dataset | platform | severity_vision | severity_text | top vision category | reason |
|---|---|---|---|---|---|---|
| communication_27--boss-revolution-0-37_221f.jpg | contextdp | mobile | 0.000 | 0.005 | urgency | severity bin 1/5 |
| 1089.jpg | contextdp | mobile | 0.000 | 0.250 | urgency | severity bin 1/5 |
| americanapprel.png | contextdp | web | 0.150 | 0.567 | scarcity | severity bin 2/5 |
| urbanedcsupply.com.png | contextdp | web | 0.150 | 0.593 | scarcity | severity bin 2/5 |
| bokksu.com.png | contextdp | web | 0.250 | 0.602 | social_proof | severity bin 3/5 |
| FIVE_1.png | own | own | 0.250 | 0.478 | other | visual-only (hard constraint) |
| THREE_7.png | own | own | 0.300 | 0.336 | guilt_wording | severity bin 4/5 |
| 13761_lRqlNk0.jpg | contextdp | mobile | 0.300 | 0.606 | guilt_wording | severity bin 4/5 |
| booking.com_.png | contextdp | web | 0.430 | 0.636 | scarcity | severity bin 5/5 |
| ONE_8.png | own | own | 0.480 | 0.732 | guilt_wording | guilt_wording (hard constraint) |

Severity spread: min 0.000, max 0.480, range 0.480, 6 distinct values (rounded to 3 dp).

Neutralized counterparts are NOT yet produced; `stimuli.csv` carries placeholder paths under `data/neutralized/` and the instrument renders a gray placeholder until the real images exist.
