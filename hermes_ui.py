#!/usr/bin/env python3
"""
HERMES anti-dox control panel — in-browser camera
==================================================
Opens your webcam right in the browser, runs the Hermes silhouette through
Python's mediapipe FaceMesh, shows the processed feed live in the page, and can
optionally stream the same feed to the OBS Virtual Camera.

Run:  python hermes_ui.py
Then open http://localhost:8711 in your browser and click "Enable Camera".

Pure stdlib server; mediapipe/opencv used only for the processing backend.
"""
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2
import mediapipe as mp

PORT = 8711
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Hermes brand palette (BGR) — bright royal blue + white ----
ACCENT_BLUE = (235, 99, 37)    # BGR of #2563eb royal blue (frame/border)
SILHOUETTE = (246, 150, 59)    # BGR of #3b96f6 lighter blue (face fill)
OUTLINE_WHITE = (255, 255, 255)  # white outline + features (brand)
BG_TOP = (245, 110, 35)        # BGR of #236ef5 bright royal blue
BG_BOTTOM = (216, 78, 29)      # BGR of #1d4ed8 deeper royal blue
NAME = "HERMES"

# Standard 468-landmark face-oval contour.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ---- shared state ----
_state = {
    "facemesh": None,
    "lock": threading.Lock(),
    "obsenabled": False,
    "vcam": None,
    "voiceproc": None,   # subprocess for the voice shifter
}

mp_face_mesh = mp.solutions.face_mesh


def build_gradient(w, h, t=0.0):
    """Base gradient with a slow vertical drift + soft ambient glow layers.
    t = seconds since start; drives the slow animation."""
    top = np.array(BG_TOP, dtype=np.uint8).astype(np.float32)
    bottom = np.array(BG_BOTTOM, dtype=np.uint8).astype(np.float32)
    hh = np.linspace(0.0, 1.0, h)[:, None, None]

    # slow breathing: lighten/darken the whole field over ~8s
    breath = 0.5 + 0.5 * np.sin(t * 0.8)
    top_adj = top * (1.0 - 0.06 * breath)
    bottom_adj = bottom * (1.0 + 0.06 * breath)

    # drifting gradient midpoint creates gentle vertical motion
    drift = (np.sin(t * 0.25) + 1.0) / 2.0
    pos = np.clip(hh + (drift - 0.5) * 0.3, 0.0, 1.0)
    grad = top_adj * (1 - pos) + bottom_adj * pos
    base = np.repeat(grad.astype(np.uint8), w, axis=1).copy()

    # soft ambient radial glows that wander slowly (2 large soft blobs)
    cy1 = int(h * (0.3 + 0.15 * np.sin(t * 0.18)))
    cx1 = int(w * (0.25 + 0.12 * np.cos(t * 0.12)))
    cy2 = int(h * (0.75 + 0.12 * np.cos(t * 0.15)))
    cx2 = int(w * (0.78 + 0.10 * np.sin(t * 0.10)))
    glow = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r1 = np.sqrt((xx - cx1) ** 2 + (yy - cy1) ** 2) / (0.6 * max(w, h))
    r2 = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2) / (0.6 * max(w, h))
    glow += np.clip(1.0 - r1, 0, 1) ** 2
    glow += np.clip(1.0 - r2, 0, 1) ** 2
    glow = np.clip(glow, 0, 1)[..., None]
    # add warm-white light toward the center (subtle, keeps blue dominant)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:] = (255, 255, 255)
    blend = (overlay.astype(np.float32) * glow * 0.06).astype(np.uint8)
    base = cv2.add(base, blend)
    return base


# FaceMesh landmark index groups for facial features (468-point model)
EYES_LEFT  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYES_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROW_LEFT  = [46, 53, 52, 65, 55]
BROW_RIGHT = [285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94]
FEATURE = (255, 255, 255)  # white feature lines (Hermes brand)


def _pts(lm, idxs, w, h):
    return [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in idxs]


def draw_features(out, lm, w, h):
    """Draw eyes, eyebrows, mouth and nose as stylized outlines so the
    silhouette reads as a recognizable face (still anti-dox: no photo detail)."""
    def polyline(idxs, close=True, thick=3):
        p = _pts(lm, idxs, w, h)
        cv2.polylines(out, [np.array(p, np.int32).reshape(-1, 1, 2)],
                      close, FEATURE, thick, cv2.LINE_AA)
    # eyes (filled dark so they read as open eye shapes)
    for e in (EYES_LEFT, EYES_RIGHT):
        p = np.array(_pts(lm, e, w, h), np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [p], FEATURE)
        cv2.polylines(out, [p], True, OUTLINE_WHITE, 1, cv2.LINE_AA)
    # eyebrows (arcs above the eyes)
    polyline(BROW_LEFT, close=False, thick=3)
    polyline(BROW_RIGHT, close=False, thick=3)
    # mouth: outer lip outline + inner (open mouth) fill
    p_out = np.array(_pts(lm, LIPS_OUTER, w, h), np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [p_out], True, FEATURE, 3, cv2.LINE_AA)
    p_in = np.array(_pts(lm, LIPS_INNER, w, h), np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(out, [p_in], FEATURE)
    # nose: simple bridge + tip dot
    polyline(NOSE, close=False, thick=3)
    nx, ny = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
    cv2.circle(out, (nx, ny), max(2, int(w * 0.006)), FEATURE, -1)


def get_facemesh():
    with _state["lock"]:
        if _state["facemesh"] is None:
            # CPU inference only — avoids mediapipe's flaky Metal/GPU path on mac.
            _state["facemesh"] = mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        return _state["facemesh"]


def process_frame(frame_bgr, ts_ms):
    """frame_bgr (h,w,3) -> processed Hermes silhouette BGR frame."""
    fm = get_facemesh()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = fm.process(rgb)

    h, w = frame_bgr.shape[:2]
    out = build_gradient(w, h, t=ts_ms / 1000.0)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        # head oval + neck/shoulders taper
        poly = []
        for idx in FACE_OVAL:
            p = lm.landmark[idx]
            poly.append([int(p.x * w), int(p.y * h)])
        chin = max(poly, key=lambda p: p[1])
        cx0, cy0 = chin
        neck_w = int(w * 0.16)
        sh_w = int(w * 0.46)
        sh_y = int(h * 1.10)
        neck_y = int(cy0 + (sh_y - cy0) * 0.30)
        poly += [[cx0 - neck_w, neck_y], [cx0 + neck_w, neck_y],
                 [cx0 - sh_w, sh_y], [cx0 + sh_w, sh_y]]
        poly = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [poly], SILHOUETTE)
        cv2.polylines(out, [poly], True, OUTLINE_WHITE, 2, cv2.LINE_AA)
        draw_features(out, lm, w, h)
    else:
        # No face detected — show a clear cue instead of a blank gradient
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = "NO FACE DETECTED - center your face"
        cv2.putText(out, txt, (int(w * 0.08), int(h * 0.5)), font, 0.7,
                    OUTLINE_WHITE, 2, cv2.LINE_AA)
    # border + nameplate
    cv2.rectangle(out, (6, 6), (w - 7, h - 7), ACCENT_BLUE, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"\u25c6  {NAME}"
    (tw, th), _ = cv2.getTextSize(label, font, 0.9, 2)
    bx1, by1 = 24, h - 54
    cv2.rectangle(out, (bx1, by1), (bx1 + tw + 36, by1 + th + 34),
                  OUTLINE_WHITE, -1)
    cv2.rectangle(out, (bx1, by1), (bx1 + tw + 36, by1 + th + 34),
                  ACCENT_BLUE, 2)
    cv2.putText(out, label, (bx1 + 20, by1 + th + 12), font, 0.9,
                ACCENT_BLUE, 2, cv2.LINE_AA)
    return out


def to_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


def send_to_obs(bgr):
    """Push processed frame to the OBS Virtual Camera (pyvirtualcam)."""
    s = _state
    if not s["obsenabled"]:
        return
    try:
        if s["vcam"] is None:
            import pyvirtualcam
            h, w = bgr.shape[:2]
            s["vcam"] = pyvirtualcam.Camera(width=w, height=h, fps=30)
            print(f"[hermes] -> OBS virtual cam {w}x{h}")
        vcam = s["vcam"]
        vcam.send(bgr)
        vcam.sleep_until_next_frame()
    except Exception as e:
        print(f"[hermes] OBS vcam error: {e}")
        s["vcam"] = None
        s["obsenabled"] = False


def voice_start(ratio, tilt, formant=None, preset=None, out_device=None):
    """Launch the voice shifter subprocess (blocking stream)."""
    s = _state
    p = s["voiceproc"]
    if p and p.poll() is None:
        return {"ok": False, "msg": "voice already running"}
    py = sys.executable
    args = [py, os.path.join(HERE, "voice_shifter.py"),
            "--ratio", str(ratio), "--tilt", str(tilt)]
    if formant is not None:
        args += ["--formant", str(formant)]
    if preset:
        args += ["--preset", preset]
    if out_device is not None:
        args += ["--out-device", str(out_device)]
    logf = open(os.path.join(HERE, "voice.log"), "ab")
    proc = subprocess.Popen(args, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
    s["voiceproc"] = proc
    return {"ok": True, "msg": f"voice shifter starting (pid {proc.pid})"}


def voice_stop():
    s = _state
    p = s["voiceproc"]
    if not p:
        return {"ok": True, "msg": "voice not running"}
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    s["voiceproc"] = None
    return {"ok": True, "msg": "voice shifter stopped"}


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HERMES · anti-dox</title>
<style>
  :root{--blue:#4285f4;--deep:#1e3a8a;--navy:#0a1023;--white:#fff;--dark:#0e1830;}
  *{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  body{background:linear-gradient(180deg,#1e3a8a 0%,#0a1023 45%);color:var(--white);min-height:100vh;padding:36px 20px;}
  .wrap{max-width:960px;margin:0 auto;}
  h1{font-size:24px;letter-spacing:2px;display:flex;align-items:center;gap:10px;margin-bottom:6px;}
  .mark{color:var(--blue);}
  .sub{color:#9fb3d8;font-size:13px;margin-bottom:22px;}
  .grid{display:grid;grid-template-columns:1.4fr 1fr;gap:22px;align-items:start;}
  @media(max-width:760px){.grid{grid-template-columns:1fr;}}
  .panel{background:rgba(14,24,48,.72);backdrop-filter:blur(14px);border:1px solid rgba(66,133,244,.35);border-radius:18px;padding:20px;box-shadow:0 20px 60px rgba(0,0,0,.5);}
  #view{width:100%;border-radius:12px;background:#0a1023;display:block;min-height:260px;object-fit:contain;}
  .camoff{text-align:center;color:#9fb3d8;padding:40px 0;}
  .row{display:flex;align-items:center;justify-content:space-between;padding:14px 0;border-bottom:1px solid rgba(66,133,244,.15);}
  .row:last-child{border-bottom:none;}
  .row .name{font-size:15px;font-weight:600;}
  .row .hint{font-size:12px;color:#9fb3d8;margin-top:2px;}
  .status{font-size:12px;padding:3px 10px;border-radius:20px;font-weight:600;}
  .st-off{background:#3a1f2a;color:#ff9db0;}
  .st-on{background:#123b2a;color:#7dffc0;}
  button{cursor:pointer;border:none;border-radius:9px;padding:10px 16px;font-size:13px;font-weight:600;transition:.15s;}
  .on{background:var(--blue);color:#fff;} .on:hover{background:#5a9bff;}
  .off{background:#2a3f66;color:#cddbf3;} .off:hover{background:#35507f;}
  .wide{width:100%;margin-top:6px;}
  .hint2{font-size:12px;color:#9fb3d8;margin-top:10px;line-height:1.6;}
  .fps{position:absolute;color:#7dffc0;font-size:12px;}
</style>
</head>
<body>
<div class="wrap">
  <h1><span class="mark">◆</span> HERMES <span style="color:#fff">anti-dox</span></h1>
  <div class="sub">Camera runs in your browser · processing happens locally · nothing leaves your Mac</div>

  <div class="grid">
    <div class="panel">
      <button class="on wide" id="camBtn" onclick="toggleCam()">Enable Camera</button>
      <div style="position:relative">
        <img id="view" alt="preview" style="display:none">
        <div id="camoff" class="camoff" style="margin-top:14px">Click <b>Enable Camera</b> to see your Hermes silhouette here.</div>
      </div>
      <div class="hint2" id="statusline">Camera off.</div>
    </div>

    <div class="panel">
      <div class="row">
        <div><div class="name">Stream to OBS</div>
          <div class="hint">Send this feed to OBS Virtual Camera</div></div>
        <span class="status st-off" id="st-obs">off</span>
      </div>
      <button class="on wide" onclick="setObs(true)">Enable OBS Stream</button>
      <button class="off wide" style="margin-top:8px" onclick="setObs(false)">Disable OBS Stream</button>
      <div class="hint2">With OBS open: <b>Start Virtual Camera</b>, then add a <b>Video Capture Device</b> source → <b>OBS Virtual Camera</b>. Start OBS stream before enabling here.</div>
    </div>

    <div class="panel">
      <div class="row">
        <div><div class="name">Voice Shifter</div>
          <div class="hint">Real-time pitch + formant disguise</div></div>
        <span class="status st-off" id="st-voice">off</span>
      </div>
      <div class="row" style="border:none">
        <div style="flex:1;margin-right:12px">
          <div class="hint">Pitch (0.5–1.3, lower = deeper)</div>
          <input type="range" id="ratio" min="0.5" max="1.3" step="0.01" value="0.85" style="width:100%">
          <div class="hint" id="ratioLbl">0.85</div>
        </div>
        <div style="flex:1">
          <div class="hint">Formant (0.6–1.4, <1 = darker character)</div>
          <input type="range" id="formant" min="0.6" max="1.4" step="0.01" value="0.90" style="width:100%">
          <div class="hint" id="formantLbl">0.90</div>
        </div>
        <div style="flex:1;margin-left:12px">
          <div class="hint">Tilt (dB)</div>
          <input type="range" id="tilt" min="-6" max="6" step="0.5" value="1.5" style="width:100%">
          <div class="hint" id="tiltLbl">+1.5</div>
        </div>
      </div>
      <div class="hint" style="margin:8px 0 6px">Quick presets</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap">
        <button class="off" style="font-size:12px;padding:6px 10px" onclick="preset('subtle')">Subtle</button>
        <button class="off" style="font-size:12px;padding:6px 10px" onclick="preset('deep')">Deep</button>
        <button class="off" style="font-size:12px;padding:6px 10px" onclick="preset('demon')">Demon</button>
        <button class="off" style="font-size:12px;padding:6px 10px" onclick="preset('alien')">Alien</button>
        <button class="off" style="font-size:12px;padding:6px 10px" onclick="preset('chipmunk')">Chipmunk</button>
        <button class="off" style="font-size:12px;padding:6px 10px" onclick="preset('radio')">Radio</button>
      </div>
      <div class="row" style="border:none">
        <div class="hint" style="flex:1">OBS output device index (0 = default; use a BlackHole virtual cable index for OBS capture)</div>
        <input type="number" id="outDev" min="0" value="0" style="width:70px;background:#0e1830;border:1px solid rgba(66,133,244,.35);color:#fff;border-radius:8px;padding:6px 8px">
      </div>
      <button class="off wide" style="margin-top:6px" onclick="listDevices()">List Audio Devices</button>
      <div class="hint2" id="devList" style="display:none;background:#0e1830;border-radius:8px;padding:10px;font-family:monospace;font-size:11px;white-space:pre-wrap"></div>
      <button class="on wide" style="margin-top:10px" onclick="voice('start')">Start Voice Shifter</button>
      <button class="off wide" style="margin-top:8px" onclick="voice('stop')">Stop Voice Shifter</button>
      <div class="hint2">Pitch changes fundamental; formant changes the vocal-tract character (the "who it sounds like"). To get it INTO OBS, install <b>BlackHole</b> (free virtual audio cable), set the output device above to BlackHole's index, and in OBS add BlackHole as an <b>Audio Input Capture</b> source.</div>
    </div>
  </div>
</div>

<script>
let stream=null, raf=null, camOn=false;

async function toggleCam(){
  if(camOn){ stopCam(); return; }
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}}, audio:false});
  }catch(e){
    document.getElementById('statusline').textContent='Camera error: '+e.message;
    return;
  }
  camOn=true;
  document.getElementById('camBtn').textContent='Disable Camera';
  document.getElementById('camBtn').className='off wide';
  document.getElementById('camoff').style.display='none';
  document.getElementById('view').style.display='block';
  processLoop();
}

function stopCam(){
  camOn=false;
  if(raf) cancelAnimationFrame(raf);
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  // drop the cached video element so a re-enable gets a fresh live stream
  if(vidEl){ vidEl.srcObject=null; vidEl=null; }
  if(lastURL){ URL.revokeObjectURL(lastURL); lastURL=null; }
  document.getElementById('camBtn').textContent='Enable Camera';
  document.getElementById('camBtn').className='on wide';
  document.getElementById('camoff').style.display='block';
  document.getElementById('view').style.display='none';
  document.getElementById('view').src='';
  document.getElementById('statusline').textContent='Camera off.';
}

const canvas=document.createElement('canvas');
let vidEl=null, lastURL=null;
async function processLoop(){
  if(!camOn) return;
  if(!vidEl){ vidEl=document.createElement('video'); vidEl.muted=true; vidEl.srcObject=stream; }
  try{
    if(vidEl.readyState < 2) await vidEl.play();
    if(!vidEl.videoWidth) vidEl.play();
    canvas.width=vidEl.videoWidth||640; canvas.height=vidEl.videoHeight||480;
    canvas.getContext('2d').drawImage(vidEl,0,0);
    canvas.toBlob((blob)=>{
      if(!blob){ setTimeout(processLoop,30); return; }
      const fd=new FormData(); fd.append('frame', blob, 'f.jpg');
      fetch('/api/process',{method:'POST',body:fd}).then(r=>r.blob()).then(jpg=>{
        const el=document.getElementById('view');
        if(lastURL) URL.revokeObjectURL(lastURL);
        lastURL=URL.createObjectURL(jpg);
        el.src=lastURL;
        document.getElementById('statusline').textContent='Live · processing locally · ~'+Math.round(1000/Math.max(performance.now()-t0,1))+' fps';
      }).catch(e=>document.getElementById('statusline').textContent='error: '+e.message)
        .finally(()=>{ t0=performance.now(); setTimeout(processLoop, 30); });
    },'image/jpeg',0.8);
    t0=performance.now();
  }catch(e){
    document.getElementById('statusline').textContent='cam: '+e.message;
    setTimeout(processLoop,100);
  }
}
let t0=performance.now();

async function setObs(on){
  const r=await fetch('/api/obs/'+(on?'on':'off'));
  const d=await r.json();
  document.getElementById('st-obs').textContent=d.obs?'on':'off';
  document.getElementById('st-obs').className='status '+(d.obs?'st-on':'st-off');
}

// voice control
const PRESETS={'subtle':[0.92,0.97,0.5],'deep':[0.80,0.88,2.0],'demon':[0.65,0.80,4.0],'alien':[0.75,1.30,1.0],'chipmunk':[1.30,1.25,-3.0],'radio':[1.00,1.10,3.0]};
function setRange(id,val){const el=document.getElementById(id);el.value=val;el.dispatchEvent(new Event('input'));}
document.getElementById('ratio').oninput=e=>document.getElementById('ratioLbl').textContent=e.target.value;
document.getElementById('formant').oninput=e=>document.getElementById('formantLbl').textContent=e.target.value;
document.getElementById('tilt').oninput=e=>document.getElementById('tiltLbl').textContent=(+e.target.value>=0?'+':'')+e.target.value;
function preset(name){
  const [ratio,formant,tilt]=PRESETS[name];
  setRange('ratio',ratio);setRange('formant',formant);setRange('tilt',tilt);
  statusline.textContent='Preset: '+name+' → Start to apply';
}
async function listDevices(){
  const box=document.getElementById('devList');
  try{
    const r=await fetch('/api/voice/devices');
    const d=await r.json();
    if(d.error){ box.textContent='error: '+d.error; box.style.display='block'; return; }
    let txt='';
    d.devices.forEach(v=>{
      txt+=(v.idx)+': '+(v.in?'[IN] ':'      ')+(v.out?'[OUT] ':'      ')+v.name+'\n';
    });
    box.textContent='Device list (put the OBS/BlackHole index above):\n'+txt;
  }catch(e){ box.textContent='error: '+e.message; }
  box.style.display='block';
}
async function voice(act){
  const ratio=document.getElementById('ratio').value;
  const formant=document.getElementById('formant').value;
  const tilt=document.getElementById('tilt').value;
  const outDev=document.getElementById('outDev').value;
  const r=await fetch(`/api/voice/${act}?ratio=${ratio}&formant=${formant}&tilt=${tilt}&out_device=${outDev}`);
  const d=await r.json();
  const el=document.getElementById('st-voice');
  el.textContent=d.running?'on':'off';
  el.className='status '+(d.running?'st-on':'st-off');
  statusline.textContent=d.msg||statusline.textContent;
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/obs/on":
            _state["obsenabled"] = True
            return self._send_json({"obs": True})
        if u.path == "/api/obs/off":
            _state["obsenabled"] = False
            if _state["vcam"]:
                try:
                    _state["vcam"].close()
                except Exception:
                    pass
                _state["vcam"] = None
            return self._send_json({"obs": False})
        if u.path.startswith("/api/voice/"):
            act = u.path.split("/api/voice/")[1]
            q = parse_qs(u.query)
            running = bool(_state["voiceproc"] and _state["voiceproc"].poll() is None)
            if act == "devices":
                try:
                    import sounddevice as sd
                    devs = []
                    for i, d in enumerate(sd.query_devices()):
                        devs.append({"idx": i, "name": d["name"],
                                     "in": d["max_input_channels"] > 0,
                                     "out": d["max_output_channels"] > 0})
                    return self._send_json({"devices": devs})
                except Exception as e:
                    return self._send_json({"devices": [], "error": str(e)})
            if act == "start":
                ratio = float(q.get("ratio", ["0.85"])[0])
                tilt = float(q.get("tilt", ["1.5"])[0])
                formant = float(q["formant"][0]) if q.get("formant") else None
                preset = q.get("preset", [None])[0] or None
                out_device = int(q["out_device"][0]) if q.get("out_device") else None
                return self._send_json({**voice_start(ratio, tilt, formant, preset, out_device), "running": True})
            if act == "stop":
                return self._send_json({**voice_stop(), "running": False})
            return self._send_json({"running": running})
        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/process":
            # read multipart frame
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            # crude multipart parse: find jpeg payload between boundary
            jpg = body.split(b"\r\n\r\n", 1)[-1].rsplit(b"\r\n", 1)[0]
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return self._send_json({"error": "bad frame"}, 400)
            ts = int(time.monotonic() * 1000)
            out = process_frame(frame, ts)
            send_to_obs(out)
            jpeg = to_jpeg(out)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return
        self._send_json({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


def main():
    print(f"[hermes] control panel on http://localhost:{PORT}")
    print("[hermes] enable camera in the browser; processed feed is local")
    try:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    except Exception:
        pass
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
