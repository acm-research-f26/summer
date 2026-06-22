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
- [x] Add this `implementation_1` folder with run script and notes
- [x] Finish upstream install (`.venv`, requirements, Playwright chromium)
- [x] Copy `.env.example` → `.env` (still need valid API keys)
- [x] First successful launch of `webui.py` via `./run_baseline.sh`

## Still need to test

- [ ] Fix Playwright chromium install (`playwright install` — browser binary missing on first task)
- [ ] Get a working LLM key (Google key was missing; OpenAI hit quota limit)
- [ ] Complete one simple task end-to-end (e.g. open a site, find text)
- [ ] What shows up in the UI logs when a task fails?

## Early observations

**2026-06-21 — first baseline run**

- `./run_baseline.sh` works: Gradio comes up at http://127.0.0.1:7788 using the project venv.
- Submitted a test task: *"open utd website and see when the admissions open"*.
- Gemini failed first — `GOOGLE_API_KEY` not set in `.env`.
- Switched to OpenAI `gpt-4o` in the UI. LLM init got further but hit `429 insufficient_quota`.
- Browser side also failed: Playwright chromium binary not installed yet (`playwright install` needed).
- So UI launch is fine; end-to-end agent run is blocked on keys + Playwright for now.

## Next research direction

Fix Playwright + API keys, rerun the UTD admissions task, then pick 2–3 fixed sites and log pass/fail + step count in `results/`. Not changing the agent until the baseline is trustworthy.
