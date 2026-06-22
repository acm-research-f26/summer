# Implementation 1 — Browser Agent Baseline

This folder is Implementation 1 for my browser-agent research. It is not a new system or verifier; it is an initial baseline built on top of the cloned [browser-use/web-ui](https://github.com/browser-use/web-ui) repository so I can run and observe an existing browser agent before changing anything.

## Objective

Get a working local copy of the web-ui, confirm I can launch the Gradio interface, and keep a small record of setup steps and open questions.

## Prerequisites

- Python 3.11
- Cloned upstream repo at `../web-ui` relative to this folder
- API keys in the upstream `.env` (see upstream README)

## Run instructions

From this directory:

```bash
chmod +x run_baseline.sh
./run_baseline.sh
```

The script checks for `../web-ui`, reminds you to activate `.venv` if needed, and tries to start the app with:

```bash
python webui.py --ip 127.0.0.1 --port 7788
```

Then open http://127.0.0.1:7788 in a browser.

For full install steps (venv, `requirements.txt`, Playwright browsers, `.env`), follow the upstream README in `../web-ui`.

## Next steps (research)

- Run a few simple tasks in the UI and save basic notes under `results/`
- Compare default agent behavior on 2–3 fixed websites
- Decide what to measure first (success rate, steps, failures)
