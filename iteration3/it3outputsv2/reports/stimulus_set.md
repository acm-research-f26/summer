# Stimulus set for the behavioral experiment

10 screens · selected 2026-07-19 19:33 · seed 42

## Selection rule

Selected on **severity_vision** (the score the experiment validates), over the 546 screens with cached vision predictions. Hard constraints: at least one ground-truth guilt_wording screen and at least one purely-visual own-set screen (FIVE_*) — without them, two of the five categories would be absent from the experiment entirely (guilt_wording exists on 12 screens of the corpus, the visual-only tricks on 6). Remaining slots: 5 equal-width severity bins, 2 per bin, seeded, preferring unrepresented vision top-categories and platforms; shortfalls fill by max-min severity distance.

**Why the model chooses the stimuli:** severity for every stimulus is fixed by the frozen iteration-1/2 detectors plus the literature-derived weights *before* any behavior is measured. The fall regression of behavioral delta on severity is therefore a genuine out-of-sample prediction test of the score, not a fit to hand-picked examples.

## Chosen screens (low -> high severity)

| screen_id | dataset | platform | severity_vision | severity_text | top vision category | reason |
|---|---|---|---|---|---|---|
| music_1--Spotify-1-6_55ff.jpg | contextdp | mobile | 0.000 | 0.061 | urgency | severity bin 1/5 |
| 1972.jpg | contextdp | mobile | 0.000 | 0.019 | urgency | severity bin 1/5 |
| (921)www.wolfandbadger.com_c9c9.jpg | contextdp | web | 0.240 | 0.604 | scarcity | severity bin 2/5 |
| herringshoes.co.uk.png | contextdp | web | 0.240 | 0.589 | scarcity | severity bin 2/5 |
| 10433.jpg | contextdp | mobile | 0.400 | 0.268 | social_proof | severity bin 3/5 |
| 1368.jpg | contextdp | mobile | 0.400 | 0.179 | other | severity bin 3/5 |
| ONE_8.png | own | own | 0.480 | 0.732 | guilt_wording | guilt_wording (hard constraint) |
| FIVE_2.png | own | own | 0.480 | 0.364 | guilt_wording | visual-only (hard constraint) |
| (273)shapermint.com_35e5.jpg | contextdp | web | 0.631 | 0.530 | urgency | severity bin 5/5 |
| dipyourcar.com.png | contextdp | web | 0.688 | 0.677 | guilt_wording | severity bin 5/5 |

Severity spread: min 0.000, max 0.688, range 0.688, 6 distinct values (rounded to 3 dp).

Neutralized counterparts are NOT yet produced; `stimuli.csv` carries placeholder paths under `data/neutralized/` and the instrument renders a gray placeholder until the real images exist.
