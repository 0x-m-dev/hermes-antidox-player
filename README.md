# HERMES anti-dox webcam player + voice shifter

Local, Hermes-themed (white + blue) anti-dox streaming kit. Reads your webcam,
replaces your face/room with a blue **"you silhouette"**, and feeds the OBS
Virtual Camera — so when you stream you never dox yourself. Plus a live voice
shifter. **Nothing leaves your machine.**

```
camera ──> cv2 ──> mediapipe FaceMesh ──> Hermes silhouette ──> OBS Virtual Cam ──> stream
mic    ──> phase-vocoder pitch shifter ──────────────────────> OBS audio chain ──> stream
```

## Files
- `webcam_player.py` — the player (blue silhouette, white nameplate, blue border)
- `voice_shifter.py` — real-time voice disguise (pitch + timbre) + offline mode
- `launch_streaming.command` — one-click Mac launcher (auto-sets up + runs both)
- `hermes_ui.py` — optional browser control panel (buttons to start/stop, preview)
- `models/face_landmarker.task` — mediapipe face model (auto-required at runtime)
- `requirements.txt` — pinned deps

## The easy way: browser control panel
Skip the terminal entirely. Run the server once, then click buttons in your browser:
```bash
python hermes_ui.py
# open http://localhost:8711
```
It auto-opens your browser. Buttons: **Start/Stop Webcam Player**, **Start/Stop
Voice Shifter**, **Open Preview**. Live status readout, logs in `player.log` /
`voice_shifter.log`. Stops every process from one place. Pure Python stdlib —
no extra install.

## Setup (macOS)
The Python code is identical and cross-platform. macOS-specific bits below.

```bash
cd anti-dox-webcam
chmod +x launch_streaming.command run_player_preview.sh
# Option A (one-click): double-click launch_streaming.command in Finder
# Option B (manual):    ./launch_streaming.command
```
Requires Python 3.10+. If the venv/pip step needs it, use `brew install python`.

**OBS Virtual Camera on macOS:**
- OBS Studio 28+ has the virtual camera built in — but on Mac it's an opt-in
  system extension. In OBS: **Tools → Virtual Camera → Start**, or the
  **Start Virtual Camera** button in Controls.
- If you see "The virtual camera is not installed", allow it in
  **System Settings → Privacy & Security → Security** (allow the OBS extension),
  then restart OBS. OBS will prompt to install the virtual cam system extension.
- Then add a **Video Capture Device** source and pick **OBS Virtual Camera**.

## Setup (Windows)
```powershell
cd anti-dox-webcam
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Requires Python 3.10+ and the **OBS Studio Virtual Camera** (built into OBS ≥ 28;
or install the `obs-virtualcam` plugin).

## Run the player
First, in OBS: **Controls → Start Virtual Camera** (bottom right).

```powershell
python webcam_player.py          # feed the OBS virtual cam (default)
python webcam_player.py --preview  # test in a local window first, no OBS
python webcam_player.py --cam 1    # different webcam index
```

Then in OBS add a **Video Capture Device** source and pick the *OBS Virtual Camera*
— your Hermes silhouette is now the camera. Face the camera; the blue silhouette
follows your head shape in real time. Nobody sees your face or your room.

- **Preview keys:** `q` / `Esc` to quit.
- First frame is slower while mediapipe loads the model (~1s), then smooth.
- Latency tips: keep `--fps 30`, resolution 1280×720; don't stack heavy OBS
  filters on the virtual cam source.

### "You silhouette" rendering
FaceMesh tracks 478 landmarks; the player fills the **face-oval contour** as the
solid blue head shape and adds a white inner outline, on a dark-blue vertical
gradient, with a white **◆ HERMES** nameplate and blue frame border. To restyle:
edit the color constants at the top of `webcam_player.py` (`ACCENT_BLUE`,
`SILHOUETTE`, `BG_TOP`, `BG_BOTTOM`, `NAME`). **BGR order** — swap R and B from
your hex.

## Run the voice shifter
```powershell
python voice_shifter.py                # live, default disguise (deeper)
python voice_shifter.py --ratio 0.85 --tilt 1.5   # tune it
python voice_shifter.py --list         # find your device index
python voice_shifter.py --device 0     # pick an input/output device
```
- `--ratio` = pitch. `< 1` = deeper (0.7–0.95), `> 1` = higher/thinner (1.05–1.4). `1.0` = off.
- `--tilt` = timbre shelf in dB. `+` darkens, `-` brightens. Adds disguise depth.
- Wire the output as a mic/audio source in OBS or your streaming app.
- **Zero-latency alternative:** OBS has a built-in **Pitch Shifter** audio filter
  on the mic source — use that when latency beats disguise depth. This Python
  shifter exists for deeper disguise or non-OBS routing.

### Offline / test mode (no audio device needed)
```powershell
python voice_shifter.py --file in.wav --out out.wav --ratio 0.8
python voice_shifter.py --test   # DSP self-test: verifies pitch shift is exact
```
`--test` is the sanity check — run it on any machine to confirm the DSP works.

## Anti-dox checklist (the software only removes the obvious)
- [ ] Masking your **face** ≠ hiding your **monitor** — keep DMs/chat/email tabs
      off-screen while streaming; that's the #1 real dox leak.
- [ ] No handles/names/tattoos visible in frame or on your setup.
- [ ] Mirrors & reflections still leak — check your backdrop once before going live.
- [ ] Voice: shift pitch/formant to *disguise*; never model a real person's voice.
- [ ] Room: the gradient kills the usual tells (posters, shelves, window light).

## Requirements
`opencv-python`, `mediapipe`, `numpy`, `pyvirtualcam`, `sounddevice`
(see `requirements.txt`; player needs the first four, shifter needs numpy + sounddevice).
