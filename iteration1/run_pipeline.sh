#!/usr/bin/env bash
# End-to-end pipeline: fetch data -> preprocess -> baseline -> transformer ->
# evaluate -> error analysis -> contract test -> smoke predictions.
# Idempotent: data clones are skipped if already present.

set -euo pipefail

cd "$(dirname "$0")"

# 1. Fetch data (idempotent)
mkdir -p data/raw
if [ ! -d data/raw/dark-patterns ]; then
  echo "==> cloning aruneshmathur/dark-patterns"
  git clone --depth 1 https://github.com/aruneshmathur/dark-patterns.git data/raw/dark-patterns
fi
if [ ! -d data/raw/ec-darkpattern ]; then
  echo "==> cloning yamanalab/ec-darkpattern"
  git clone --depth 1 https://github.com/yamanalab/ec-darkpattern.git data/raw/ec-darkpattern
fi

# 2. Preprocess + baseline (CPU only)
python -m src.data_prep
python -m src.baseline

# 3. Transformer fine-tuning (GPU strongly preferred; CPU fallback works but slow)
python -m src.train_transformer

# 4. Evaluation + reports
python -m src.evaluate
python -m src.error_analysis

# 5. Frozen-contract test
pytest -q tests/test_inference_contract.py

# 6. Smoke predictions on hand-written strings
python - <<'PY'
import json
from src.inference import predict

samples = {
    "urgency":       "Sale ends in 5 minutes!",
    "scarcity":      "Only 2 left in stock — order soon.",
    "social_proof":  "Joan from Boston just bought this 3 hours ago.",
    "guilt_wording": "No thanks, I prefer to pay full price.",
    "other":         "By continuing you agree to be charged monthly.",
}
for label, text in samples.items():
    print(f"\n[{label}]  {text!r}")
    print(json.dumps(predict(text), indent=2))
PY
