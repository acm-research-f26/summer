# FIGO percent-intramural measurement pipeline. CPU-only, no GPU anywhere.
PY      := python3.13
VENV    := .venv
BIN     := $(VENV)/bin
PYTHON  := $(BIN)/python
PIP     := $(BIN)/pip

.DEFAULT_GOAL := help
.PHONY: all help setup data demo manifest body extract figo uncertainty borderline robustness report test clean

# NOTE: stage targets are intentionally NOT chained to each other. Making `figo`
# depend on `extract` re-ran the entire 30-minute extraction every time, because
# both are .PHONY and make cannot tell the work was already done. Run stages in
# order, or use `make all`.
all: manifest body extract figo uncertainty borderline robustness report  ## run every stage in order

help:  ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/.stamp  ## create venv and install dependencies
$(VENV)/.stamp: requirements.txt pyproject.toml
	$(PY) -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt
	$(PIP) install --quiet -e .
	@touch $@
	@echo "setup OK -> $(VENV)"

data:  ## download + extract UMD from figshare (4.76 GB) into data/UMD
	./scripts/download_umd.sh

manifest: setup  ## PHASE 1: inventory + DICOM spacing truth table
	$(PYTHON) scripts/build_manifest.py

body: setup  ## PHASE 2: uterine reference-surface reconstruction, gates + QC
	$(PYTHON) scripts/run_body.py

extract: setup  ## PHASE 3: per-fibroid feature table
	$(PYTHON) scripts/extract_features.py

figo: setup  ## PHASE 4: FIGO distribution + submucosal-count correctness gate
	$(PYTHON) scripts/run_figo.py

uncertainty: setup  ## PHASE 5: per-fibroid percent-intramural confidence intervals
	$(PYTHON) scripts/run_uncertainty.py

borderline: setup  ## PHASE 6: flag ambiguous near-50% cases, patient-grouped CV
	$(PYTHON) scripts/run_borderline.py

robustness: setup  ## PHASE 7: slice-decimation + subset sensitivity
	$(PYTHON) scripts/run_robustness.py

demo: setup  ## end-to-end on synthetic masks, no dataset needed
	$(PYTHON) scripts/run_demo.py

report: setup  ## PHASE 8: figures + RESULTS.md
	$(PYTHON) scripts/make_report.py

test: setup  ## run the test suite
	$(BIN)/pytest

clean:  ## remove venv, caches, and generated outputs
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf results/* reports/*
