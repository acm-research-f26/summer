![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations

# Lighthouse - Catching Dark Patterns in UIs

Iteration 2 README below, Iteration 1 README can be found in the iteration1 folder.

## 📌 Project Summary
Dark patterns are the shady design tricks that nudge users to do things that benefit the web designer, but that the user did not intend to do. These include manipulative "only 2 left!" warnings, loud countdown timers, guilt-trip buttons like "No thanks, I hate saving money," and boxes that are already checked for you.

Most research just counts how often these show up. What's missing is the aspect of how much each trick actually changes what people do, and whether you can guess that just by looking at a screen. The primary goal is to build a model that first identifies what dark pattern is occuring without human analysis, then scores how manipulative a screen is. After, a real experiment will be run to check if that score predicts how much people's choices actually change, with that experiment being used to test and tune the model.

This iteration goes from reading the words on a screen to seeing the screen itself. A lot of the strongest dark patterns aren't in the text, but rather in the design, like a greyed-out "decline" next to a giant "accept," or a box already ticked. This iteration puts a vision model in front of the  screenshot and adds another layer to identifying dark patterns and how strong they are.

## 🎯 Motivation
Dark patterns are everywhere and I notice them constantly. Humans are good at spotting and naming them, and the taxonomies are well researched, but the gap is that this can't be done well automatically yet. One set of papers tells you a trick is there, another tells you tricks work in general, but there isn't a model that detects them efficiently. They also do not provide variance in how strong these patterns are. The protection that would arise from being able to identify what dirty tricks are occuring on a UI, and how strong they are, is the motivation for this research.

## 🧩 Novelty
- **Score-to-behavior link:** Detection tools like AidUI and UIGuard stop at "a dark pattern is here." Behavioral work like Luguri & Strahilevitz measures effect but never ties it to a model. There is not research connecting a model's per-screen manipulation score to a measured shift in real decisions. That predictive link, and the R² behind it, is the novelty of this research.
- **Prevalence vs. potency:** Mathur counted how often each trick appears, and Luguri measured how hard each one hits. Running this detector over many real screens lets us ask whether the web's most common manipulations are also its most effective, or whether sites lean on weak tricks and under-use strong ones.
- **Text-vs-vision blind spots:** Because we run a reading model and a seeing model side by side, we can isolate the dark patterns that are invisible to text and only show up visually, providing novelty compared to previous research that solely exists in one camp or another.

## 🧠 Methodology
1. **Dataset**: [CONTEXTDP](https://github.com/SageSELab/AidUI) (the AidUI evaluation set), 501 labeled screenshots (162 web, 339 mobile) with real dark-pattern instances and clean controls. On top of that, a hand-curated set of 45 of my own screenshots to cover the gaps the public data leaves: 20 random, 10 guilt-wording buttons, 9 salesy-but-clean controls, and 6 purely visual tricks.
2. **Architecture**: two systems go head-to-head on the same screenshots, everything frozen (no training, no threshold tuning, cutoff fixed at 0.50).
   - System A, reading: OCR (EasyOCR) pulls every readable string off the screenshot, then the frozen text classifier scores it. It only knows what it can spell out. *(easyOCR → filter → DistilBERT → max-product aggregate)*
   - System B, seeing: [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), run on Colab. This works to read the pixels directly to identify colors, sizes, preselected controls, and more to find patterns that sneak past the text model *(screenshot → Qwen2-VL-7B, prompt v3 → JSON scores)*.
   - Both output the same 5 presence scores (urgency, scarcity, social proof, guilt wording, other) plus an overall is_dark verdict.
3. **Evaluation**: per-category precision, recall, and F1 (multi-label), is_dark accuracy, web-only and mobile-only cuts, and a threshold check at 0.4 / 0.5 / 0.6. The 45 own screenshots run as a separate, harder test set.
4. **Metrics**: precision, recall, F1 per category, micro/macro-F1, is_dark accuracy.

#### Additional Methodology:
- **The visual-only test**: six screenshots of purely structural tricks (preselected checkboxes, false hierarchy) with no manipulative words on them at all. This isolates exactly what vision can catch that text just can't, and it's the cleanest test of the whole iteration.

### Results

TLDR: The vision model has been largely successful in detecting dark patterns compared to the text-only model as it catches tricks text is blind to. Two datasets back that up, the public CONTEXTDP set (501) and my own set (45). This vision model shall be tightened up over the cohort with more participants supporting, but iteration 2 proves its feasibility.

**Curated Dataset of 45 Screens:**

| Metric | Text | Vision | Δ |
|---|---|---|---|
| is_dark F1 | 0.831 | 0.904 | +0.073 |
| is_dark accuracy | 0.711 | 0.844 | +0.133 |
| micro-F1 (all flags) | 0.534 | 0.577 | +0.043 |
| visual-only tricks named | — | 5 / 6 | — |

**Full per-category scorecard (own set):**

| Category | Sup | Text P | Text R | Text F1 | Vision P | Vision R | Vision F1 | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| urgency | 6 | 0.240 | 1.000 | 0.387 | 0.353 | 1.000 | 0.522 | +0.135 |
| scarcity | 5 | 0.312 | 1.000 | 0.476 | 0.333 | 0.800 | 0.471 | −0.006 |
| social_proof | 6 | 0.294 | 0.833 | 0.435 | 0.500 | 0.167 | 0.250 | −0.185 |
| guilt_wording | 12 | 1.000 | 0.667 | 0.800 | 1.000 | 0.667 | 0.800 | 0.000 |
| other | 18 | 0.611 | 0.611 | 0.611 | 0.818 | 0.500 | 0.621 | +0.010 |
| micro | | 0.417 | 0.745 | 0.534 | 0.560 | 0.596 | 0.577 | +0.043 |
| is_dark | | 0.727 | 0.970 | 0.831 | 0.825 | 1.000 | 0.904 | +0.073 |

The two detectors make opposite mistakes. Across categories, text over-fires: recall 0.82 but precision 0.49, so it flags almost everything real plus a lot that isn't. Vision is the pickier one: precision 0.60, recall 0.63. It says less, but what it says is a lot more trustworthy.

guilt_wording got a real number for the first time. No public dataset has any guilt-wording labels, so the 12 guilt screens I collected are the first actual measurement of it anywhere. Both systems land on the same line, F1 0.80 with precision 1.00, meaning every screen either one flagged as guilt-wording was right.

**On web pages (CONTEXTDP), vision wins across the board, and urgency is the big one:**

| Category (web) | Text F1 | Vision F1 | Δ |
|---|---|---|---|
| urgency | 0.573 | 0.819 | +0.25 |
| scarcity | 0.514 | 0.541 | +0.03 |
| micro-F1 | 0.406 | 0.559 | +0.15 |
| is_dark acc | 0.500 | 0.648 | +0.15 |

The threshold check holds the same ordering at 0.4 and 0.5, so it's not a lucky cutoff. On the six purely visual screens, the model named the exact trick in its own words on 5 of 6, something OCR can't pull from the image at all:

> *"A preselected checkbox for a 2-year protection plan."*
> *"A cookie consent banner with options preselected in the site's favor."*
> *"A large, blue 'Accept all' button placed front and center."*

## 🌍 Impact
A good manipulation score is step one toward tools that warn users, help regulators check sites quickly, and let researchers say which tricks actually matter instead of just counting them. It also speeds up dark-pattern research in general, since a lightweight open-source model that both reads and sees a UI means nobody has to sit and grade every screen by hand.

#### Future Work
Iterations 1 and 2 build the detector, but detecting dark patterns isn't new, as AidUI and UIGuard already spot them in a screenshot.
 
- **Iteration 3, tying the score to real behavior:** take the model's manipulation score for a screen and run an experiment to see if that score predicts how much the screen actually changes someone's decision. This means future work for iteration 3 will involve preparing the questions and experiments required.
- **Model Improvement:** sharpen the model's confidence (it currently hedges visual catches at a flat 0.50), and break up the messy "other" bucket and grow the labeled set so the scores are solid enough to trust.


**Additional Sources:**
- Mansur et al., *AidUI: Toward Automated Recognition of Dark Patterns in User Interfaces* (2023), CONTEXTDP dataset
- Mathur et al., *Dark Patterns at Scale* (2019)
- Luguri & Strahilevitz, *Shining a Light on Dark Patterns* (2021)
- Yada et al., *Dark Patterns in E-commerce: a dataset and baselines* (2022)
