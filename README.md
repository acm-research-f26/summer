![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

---

# Autonomous Browser Agents for Real-World Web Tasks

## 📌 Project Summary

This project studies how large language model (LLM)–driven browser agents perform on everyday web navigation tasks — things like finding information on a university site, filling forms, or clicking through multi-step flows. Rather than building a system from scratch right away, I'm starting with a reproducible baseline using [browser-use/web-ui](https://github.com/browser-use/web-ui), an open-source Gradio interface on top of the [browser-use](https://github.com/browser-use/browser-use) agent framework. The goal is to understand default agent behavior, failure modes, and what to measure before proposing changes.

**Current status:** Implementation 1 (baseline setup) is in [`implementation_1/`](implementation_1/). The Gradio UI launches locally; end-to-end task runs are in progress while I finish Playwright and API key setup.

## 🎯 Motivation

A lot of "AI agent" demos look impressive in short clips, but it's unclear how reliably they work on real websites with dynamic DOMs, login walls, and noisy layouts. I want to move past hype and actually observe an agent step-by-step: where it succeeds, where it gets stuck, and whether failures come from the LLM, the browser layer, or the task itself.

I'm especially interested in tasks that matter to students and researchers — looking up deadlines, navigating official pages, extracting structured facts from unstructured web content. A solid baseline lets me compare future improvements (better prompts, planners, verifiers, etc.) against something concrete.

## 🧩 Novelty

- **Baseline-first approach**: Establish a trustworthy, repeatable setup before modifying the agent — so later results are actually comparable.
- **Real-site task focus**: Planned evaluation on fixed, real-world websites (not just sandboxed toy pages).
- **Failure analysis**: Track not just pass/fail but *why* tasks fail (LLM errors, browser/Playwright issues, site layout changes, etc.).

## 🧠 Methodology

1. **Dataset**: No standard benchmark dataset yet. For the baseline phase I'm using a small set of hand-written tasks on real websites (e.g. "open the UTD website and find when admissions open"). Longer term I may adopt or adapt benchmarks like [WebArena](https://webarena.dev/) or [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) once the local pipeline is stable.

2. **Architecture**: [browser-use/web-ui](https://github.com/browser-use/web-ui) as the front end, running the browser-use agent with:
   - **LLM backbone**: Configurable via `.env` (testing Gemini 2.0 Flash and GPT-4o)
   - **Browser automation**: Playwright (Chromium)
   - **UI**: Gradio web interface for task input and live step logs
   - **Launcher**: Custom `implementation_1/run_baseline.sh` script for one-command local startup

3. **Evaluation**:
   - Run each task multiple times once the baseline is fully working
   - Log agent steps, final answer, and whether the task completed
   - Note failure category when a run doesn't succeed
   - Save run logs under `implementation_1/results/`

4. **Metrics**:
   - **Task success rate** — did the agent produce a correct/useful answer?
   - **Step count** — how many actions before completion or failure?
   - **Time to completion** — wall-clock time per task
   - **Failure type breakdown** — LLM / browser / site / timeout

#### Additional Methodology:
- **Reproducibility**: All setup steps, working notes, and run instructions live in [`implementation_1/`](implementation_1/) so someone else can clone web-ui and rerun the baseline.

## 🌍 Impact

Reliable browser agents could reduce repetitive web research — checking deadlines, gathering info across sites, or automating simple workflows. But that only matters if we understand failure modes and can measure improvement honestly. This project aims to build that foundation: a baseline you can run locally, a small task suite on real sites, and clear metrics before claiming anything works "autonomously."

#### Future Work
- Complete end-to-end baseline runs after fixing Playwright install and API keys
- Define a fixed task suite (3–5 sites, 1–2 tasks each) and log results in `results/`
- Compare LLM providers (Gemini vs GPT-4o) on the same tasks
- Explore additions from the literature: planning modules, action verifiers, memory improvements
- Optionally integrate a formal benchmark (WebArena, Mind2Web) for standardized comparison

**Additional Sources:**
- [browser-use](https://github.com/browser-use/browser-use) — core browser agent framework
- [browser-use/web-ui](https://github.com/browser-use/web-ui) — Gradio UI used as baseline
- [WebArena](https://webarena.dev/) — realistic web environment benchmark
- [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) — generalist web agent dataset

## Repository layout

```
implementation_1/
├── README.md          # Setup and run instructions for the baseline
├── notes.md           # Working research notes and observations
├── run_baseline.sh    # Launcher for the web-ui Gradio app
└── results/           # Run logs and experiment output (gitignored except .gitkeep)
```

## Quick start

```bash
# 1. Clone browser-use/web-ui separately (sibling to this repo or anywhere you prefer)
git clone https://github.com/browser-use/web-ui.git

# 2. Follow web-ui README for venv, requirements, Playwright, .env

# 3. From this repo's implementation_1 folder:
chmod +x run_baseline.sh
./run_baseline.sh
# Open http://127.0.0.1:7788
```

See [`implementation_1/README.md`](implementation_1/README.md) for full details.
