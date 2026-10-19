#!/bin/bash
# Start backend + frontend for local dev. Run from the project root.
set -e

if [ ! -f "model/model.h5" ]; then
  echo "ERROR: model/model.h5 not found."
  echo "Train the model first: open training/skin_cancer_training.ipynb in Google Colab,"
  echo "then place model.h5 and classes.json in the model/ folder."
  exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
if [ -z "$PYTHON_BIN" ] && [ -n "${CONDA_PREFIX:-}" ]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
fi
if [ -z "$PYTHON_BIN" ] && command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
fi
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(1)
PY
then
  echo "ERROR: This project requires Python 3.12."
  echo "Install Python 3.12, or create a compatible conda env:"
  echo "  conda create -n skin-cancer-ai python=3.12"
  echo "  conda activate skin-cancer-ai"
  exit 1
fi

cd backend
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -q -r requirements.txt

echo "Starting backend on http://localhost:5000 ..."
uvicorn main:app --host 0.0.0.0 --port 5000 --reload &
BACKEND_PID=$!

cd ../frontend
echo "Starting frontend on http://localhost:5500 ..."
python -m http.server 5500 &
FRONTEND_PID=$!

echo ""
echo "Running. Backend: http://localhost:5000   Frontend: http://localhost:5500"
echo "Press Ctrl+C to stop both."
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
