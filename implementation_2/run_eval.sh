#!/usr/bin/env bash
# Run offline ARP evaluation (no browser or API keys required).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Implementation 2 — ARP offline evaluation ==="
echo

# Prefer web-ui venv if available
PYTHON=""
if [ -x "../.venv/bin/python" ]; then
  PYTHON="../.venv/bin/python"
elif [ -x "../../.venv/bin/python" ]; then
  PYTHON="../../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

echo "Using: ${PYTHON}"
echo

# Install lightweight ML deps if missing
if ! "${PYTHON}" -c "import sklearn" 2>/dev/null; then
  echo "Installing scikit-learn (one-time)..."
  "${PYTHON}" -m pip install -q -r requirements.txt
  echo
fi

"${PYTHON}" eval/run_offline_eval.py
echo
"${PYTHON}" tests/test_action_relevance.py
