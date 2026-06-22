# Working notes — Implementation 1

## Implementation goal

Establish a reproducible baseline using browser-use/web-ui. I want to launch the Gradio UI, run a couple of manual agent tasks, and understand the default flow before designing experiments.

## Baseline selected

[browser-use/web-ui](https://github.com/browser-use/web-ui) — Gradio front-end on top of browser-use.

## Why I chose it

- Already integrates browser-use with a usable UI (less glue code for week one)
- Supports multiple LLM providers via `.env`
- Active repo with clear install steps
- Enough to study agent behavior without building a UI from scratch

## Setup progress

- [x] Clone upstream repo locally (`../web-ui`)
- [x] Read README (venv, requirements, Playwright, `.env`)
- [ ] Add this `implementation_1` folder with run script and notes
- [ ] Finish upstream install (`.venv`, requirements, Playwright chromium)
- [ ] Copy `.env.example` → `.env` and add API keys
- [ ] First successful launch of `webui.py`

## Still need to test

- Does Playwright Chromium install cleanly here?
- Can I complete one simple task end-to-end (e.g. open a site, find text)?
- What shows up in the UI logs when a task fails?

## Early observations

Haven't run anything yet — starting with env setup. Expect setup to take longer than the docs suggest.

## Next research direction

Once baseline runs: pick 2–3 fixed tasks, run each a few times, log pass/fail and step count in `results/`. Then decide what to change—not before I trust the baseline.
