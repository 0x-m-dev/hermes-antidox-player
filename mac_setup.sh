#!/bin/bash
# HERMES anti-dox — one-command Mac setup (Python 3.12, CPU-stable mediapipe)
# Run once:  bash mac_setup.sh   then:   python3 hermes_ui.py
set -e
cd "$(dirname "$0")"

echo "[hermes] Checking Python 3.12..."
PY=""
for c in python3.12 python3.11 python3.10; do
  if command -v $c >/dev/null 2>&1; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  echo "[hermes] Python 3.10-3.12 not found."
  echo "Install it, e.g.:  brew install python@3.12"
  echo "Then re-run this script."
  exit 1
fi
echo "[hermes] using $PY"

if [ ! -d ".venv" ]; then
  echo "[hermes] creating venv..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
echo "[hermes] installing dependencies (this takes a minute)..."
pip install --upgrade pip -q
pip install -q -r requirements.txt
echo ""
echo "[hermes] DONE. Start with:"
echo "        source .venv/bin/activate"
echo "        python3 hermes_ui.py"
echo "        -> browser opens http://localhost:8711 -> Enable Camera"
