#!/usr/bin/env bash
# Launcher for browser-use/web-ui baseline.
# Edit WEB_UI_PATH if your clone lives somewhere else.

WEB_UI_PATH=".."
PORT=7788

set -e

if [ ! -d "${WEB_UI_PATH}" ]; then
  echo "Error: web-ui not found at ${WEB_UI_PATH}"
  echo "Clone it with: git clone https://github.com/browser-use/web-ui.git ../web-ui"
  exit 1
fi

cd "${WEB_UI_PATH}"

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

echo "Starting web-ui on http://127.0.0.1:${PORT}"
exec "${PYTHON}" webui.py --ip 127.0.0.1 --port "${PORT}"
