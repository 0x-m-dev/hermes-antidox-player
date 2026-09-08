#!/bin/bash
# HERMES anti-dox - one-click launcher (macOS)
# Double-click this file, or: chmod +x launch_streaming.command && ./launch_streaming.command
cd "$(dirname "$0")"

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "[hermes] creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate || { echo "[hermes] FAILED to activate venv. Need Python 3.10+."; read -n1; exit 1; }

pip install -q -r requirements.txt

echo ""
echo "============================================================"
echo "  HERMES anti-dox player + voice shifter"
echo "  Window 1: player (preview)   Window 2: voice shifter"
echo ""
echo "  In OBS: Controls -> Start Virtual Camera, then add a"
echo "  'Video Capture Device' source -> OBS Virtual Camera."
echo "============================================================"
echo ""

# Launch player in its own Terminal window (preview mode)
open -a Terminal "$PWD/run_player_preview.sh"

# Run the voice shifter here
python voice_shifter.py

echo ""
echo "[hermes] stopped."
read -n1 -p "Press any key to close..."
