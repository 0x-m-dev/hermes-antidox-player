#!/bin/bash
# Launches the HERMES player in preview mode (called by launch_streaming.command)
cd "$(dirname "$0")"
source .venv/bin/activate
python webcam_player.py --preview
