@echo off
REM HERMES anti-dox - one-click launcher
REM Usage: double-click, or drag into a terminal.
cd /d "%~dp0"

if not exist .venv (
    echo [hermes] creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [hermes] FAILED to activate venv. Make sure Python 3.10+ is installed.
    pause & exit /b 1
)

pip install -q -r requirements.txt

echo.
echo ============================================================
echo  HERMES anti-dox player + voice shifter
echo  Window 1: player (preview)   Window 2: voice shifter
echo  In OBS: Controls -^> Start Virtual Camera, then add a
echo  "Video Capture Device" source =^> OBS Virtual Camera.
echo ============================================================
echo.

start "hermes-player" cmd /k "python webcam_player.py --preview"
timeout /t 2 >nul
python voice_shifter.py

echo.
echo [hermes] stopped.
pause
