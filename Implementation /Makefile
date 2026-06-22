PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup download synthetic demo-extract demo-train demo-report demo-preview demo extract train report preview test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

download:
	@echo "Download the UMD dataset (https://figshare.com/articles/dataset/UMD_zip/23541312)"
	@echo "and unzip it under data/UMD/ so that per-patient *_seg.nii.gz masks exist."
	@echo "Expected mask labels: 0 background, 1 wall, 2 cavity, 3 fibroid, 4 nabothian cyst."

synthetic:
	$(PY) scripts/make_synthetic_masks.py --output-dir data/demo/raw --patients 30

demo-extract:
	$(PY) scripts/extract_features.py --mask-dir data/demo/raw --output data/demo/processed/fibroid_features.csv

demo-train:
	$(PY) scripts/train_models.py --features data/demo/processed/fibroid_features.csv --output-dir reports/demo

demo-report:
	$(PY) scripts/make_report.py --features data/demo/processed/fibroid_features.csv --output-dir reports/demo/eda

demo-preview:
	$(PY) scripts/preview_masks.py --mask-dir data/demo/raw --output-dir reports/demo/mask_previews --limit 12

demo: synthetic demo-extract demo-train demo-report demo-preview

extract:
	$(PY) scripts/extract_features.py --mask-dir data/UMD --pattern "*_seg.nii.gz" --output data/processed/fibroid_features.csv

train:
	$(PY) scripts/train_models.py --features data/processed/fibroid_features.csv --output-dir reports

report:
	$(PY) scripts/make_report.py --features data/processed/fibroid_features.csv --output-dir reports/eda

preview:
	$(PY) scripts/preview_masks.py --mask-dir data/UMD --pattern "*_seg.nii.gz" --output-dir reports/mask_previews --limit 12

test:
	$(PY) -m pytest

clean:
	rm -rf data/processed/*.csv reports/*.csv reports/*.png reports/*.joblib reports/*_feature_importance.csv reports/eda reports/mask_previews reports/demo
