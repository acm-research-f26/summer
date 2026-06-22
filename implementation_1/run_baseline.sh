#!/usr/bin/env bash
# Simple launcher for the browser-use/web-ui baseline.
# Edit WEB_UI_PATH if your clone lives somewhere else.
# If this folder is inside the web-ui clone, use: WEB_UI_PATH=".."

WEB_UI_PATH=".."
PORT=7788

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -e

echo "=== Implementation 1 — baseline launcher ==="
echo "Looking for web-ui at: ${WEB_UI_PATH}"
echo

if [ ! -d "${WEB_UI_PATH}" ]; then
  echo "Error: directory not found: ${WEB_UI_PATH}"
  echo
  echo "Expected a cloned browser-use/web-ui repo nearby."
  echo "Example:"
  echo "  git clone https://github.com/browser-use/web-ui.git ../web-ui"
  echo
  echo "Or edit WEB_UI_PATH at the top of this script."
  exit 1
fi

cd "${WEB_UI_PATH}"
echo "Using repo: $(pwd)"
echo

# Use .env from this implementation folder if web-ui has none yet
if [ ! -f ".env" ] && [ -f "${SCRIPT_DIR}/.env" ]; then
  cp "${SCRIPT_DIR}/.env" ".env"
  echo "Copied .env from implementation_1/ into web-ui repo."
  echo
fi

# Pick Python: prefer project venv, then python3, then python
PYTHON=""
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
  echo "Using venv: .venv/bin/python"
elif [ -x "venv/bin/python" ]; then
  PYTHON="venv/bin/python"
  echo "Using venv: venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
  echo "Using system python3 (no venv found in web-ui repo)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
  echo "Using system python (no venv found in web-ui repo)"
else
  echo "Error: no Python found."
  echo
  echo "One-time setup in the web-ui repo:"
  echo "  cd ${WEB_UI_PATH}"
  echo "  uv venv --python 3.11    # or: python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  uv pip install -r requirements.txt"
  echo "  playwright install chromium --with-deps"
  echo "  cp .env.example .env     # then add API keys"
  exit 1
fi
echo

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "Note: no .env found. Add keys in implementation_1/.env or run:"
  echo "  cp .env.example .env"
  echo
fi

if [ ! -x ".venv/bin/python" ] && [ ! -x "venv/bin/python" ]; then
  echo "Tip: create a venv in the web-ui repo for a reliable baseline:"
  echo "  uv venv --python 3.11 && source .venv/bin/activate"
  echo "  uv pip install -r requirements.txt && playwright install chromium --with-deps"
  echo
fi

LAUNCH_CMD=("${PYTHON}" webui.py --ip 127.0.0.1 --port "${PORT}")

echo "Start command:"
echo "  ${LAUNCH_CMD[*]}"
echo
echo "Then open: http://127.0.0.1:${PORT}"
echo

if [ ! -f "webui.py" ]; then
  echo "Error: webui.py not found in ${WEB_UI_PATH}"
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Error: port ${PORT} is already in use (maybe an old web-ui still running)."
  echo "Stop it with:"
  echo "  lsof -i :${PORT}"
  echo "  kill <PID>"
  echo
  echo "Or edit PORT at the top of this script."
  exit 1
fi

echo "Starting web-ui..."
exec "${LAUNCH_CMD[@]}"
