![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations

# Lighthouse - Catching Dark Patterns in UIs


## 📌 Project Summary
Dark patterns are the shady design tricks that nudge users to do things that benefit the web designer, but that the user did not intend to do. These include manipulative "only 2 left!" warnings, loud countdown timers, guilt-trip buttons like "No thanks, I hate saving money," boxes that are already checked for you.

Most research just counts how often these show up. What's missing is the aspect of how much each trick actually changes what people do, and whether you can guess that just by looking at a screen. The primary goal is to build a model that first identifies what dark pattern is occuring without human analysis, then scores how manipulative a screen is. After, a real experiment will be run to check if that score predicts how much people's choices actually change.

Iteration 1 was focused on designing a model that can identify which dark pattern is occuring in a UI, and if there even is one, using the text present. Iteration 2 will jump to a vision language model, as well as incorporating the scoring aspect.

## 🎯 Motivation
Dark patterns are everywhere and something that I've noticed constantly. I feel that they're taxonomies and identification when humans are doing it are well researched, but the hole in the research is that this cannot be done well automously. One group of papers tells you a trick is ther and another tells you tricks work in general, but there aren't models that are efficient in detecting these things. To design that, you first need a model that can read a screen, so iteration 1 builds the text half.

## 🧩 Novelty
- **Score-to-behavior link**: Prior work either detects dark patterns or measures their effect on people, but not both. The eventual contribution is connecting a model's per-screen manipulation score to a measured change in real decisions.
- **Burdensome requirement of human analysis**: The experienced eye of a human is necessary to classify and analyze these UI decisions, so the novelty here is to automate this process while maintaining high accuracy, which is to be tested and fine-tuned with the help of study groups.

## 🧠 Methodology
1. **Dataset**: [Mathur "Dark Patterns at Scale"](https://github.com/aruneshmathur/dark-patterns) (1,818 labeled lines, 1,178 after cleaning) for the category task, plus [ec-darkpattern](https://github.com/yamanalab/ec-darkpattern) (a balanced set) for the yes/no task. I also made a bigger training set by having an LLM reword each line a bunch of ways to diversify the training set.
2. **Architecture**: fine-tuned **DistilBERT**, with a simple TF-IDF + logistic regression baseline to compare against.
   - Two models: one for "is this dark," one for the 5 categories (urgency, scarcity, social proof, guilt wording, other).
   - Trained the category one with and without class weights. Weighted won.
   - The model name is one line of config, so swapping to RoBERTa later is easy.
3. **Evaluation**:
   - 70/15/15 split, fixed seed, so both models get tested on the same rows.
   - A normal test set.
   - A harder test was also used; 40 lines written by hand in different wording and non-shopping spots (apps, subscriptions, cookie banners), to see if it holds up on stuff it's never seen.
4. **Metrics**: accuracy, macro-F1, per-class F1, confusion matrices.

#### Additional Methodology:
- **The augmentation jump**: our first model looked great on the normal test (~0.97) but did not perform on the reworded test (~0.70), and urgency tanked (1 out of 6) as it had just memorized shopping wording. It knew "limited time only" but not "act fast, this won't come back", so I had an LLM reword every training line in various ways and trained on all of it. Result was a deeper understanding of each trick rather than learning exact words. Only the training set was reworded and never the test, so the comparison stays fair.

### Results
 
Normal test (same kind of data it trained on):
 
| Task | Model | Macro-F1 |
|---|---|---|
| Binary | Baseline | 0.94 |
| Binary | DistilBERT | **0.97** |
| Categories | DistilBERT | **0.97** |
 
This looked good but was suspicious. Here's before and after reworded training data was added:
 
| Test | Binary | Categories | Urgency |
|---|---|---|---|
| Reworded, before | ~0.71 | ~0.69 | 1 / 6 |
| Reworded, after | **~0.88** | **~0.87** | **5 / 6** |
 
On brand-new wording it now catches 26 of 30 dark patterns and gets the category right 87% of the time, increased from basically guessing. Scarcity is perfect now (6/6), and urgency went from a not performing to 5/6. The one weak spot left is "other," which currently stands as a sort of 'junk-drawer' bucket.

## 🌍 Impact
A good manipulation score is step one toward tools that warn users, help regulators check sites fast, and let researchers say tricks actually matter instead of just counting them. The future work of researchers looking into dark patterns can be greatly sped up as well with the existence of a lightweight, open-source model that detects and scores dark patterns in a UI.

#### Future Work
- **Iteration 2 - going visual and scoring**: A lot of the worst dark patterns aren't in the words used, but rather in the visual elements of the UI. Examples include a greyed-out "decline" next to a giant "accept," a box already being ticked, a fake timer counting down, important text shrunk into a corner, the cheap option hidden at the bottom, etc. Iteration 2 swaps in a *vision-language model that looks at the actual screenshot and it's attributes, like colors, layout, button shapes and sizes, and highlighted areas. Since labeled screenshots barely exist, it'll mostly prompt a vision model instead of training one. An OCR step (screenshot to text to this model) connects the two to continue using the model developed in iteration 1.

**Additional Sources:**
- Mathur et al., *Dark Patterns at Scale* (2019)
- Luguri & Strahilevitz, *Shining a Light on Dark Patterns* (2021)
- Yada et al., *Dark Patterns in E-commerce: a dataset and baselines* (2022)
