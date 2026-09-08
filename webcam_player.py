#!/usr/bin/env python3
"""
HERMES anti-dox webcam player
=============================
Reads your webcam, replaces your face/room with a Hermes-styled blue "you
silhouette", and outputs the result to the OBS Virtual Camera (or a preview
window). Nothing leaves your machine.

Usage:
  python webcam_player.py                     # -> OBS Virtual Camera (default)
  python webcam_player.py --preview           # local window, no OBS needed
  python webcam_player.py --cam 1             # pick a different webcam index
  python webcam_player.py --mock              # synthetic face for testing

Keyboard (preview only): q = quit, s = toggle silhouette fill on/off
"""
import argparse
import time

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ---------------- Hermes theme (white + blue) ----------------
# NOTE: OpenCV works in BGR. Values below are stored as BGR (blue channel
# highest for blue colors). The RGB equivalents are in the comments.
ACCENT_BLUE   = (244, 133, 66)   # BGR of RGB(66,133,244)  -> bright Hermes blue
SILHOUETTE    = (244, 133, 66)   # BGR of RGB(66,133,244)  -> silhouette fill
OUTLINE_WHITE = (255, 255, 255)  # white contour / accent
BG_TOP        = (138, 58, 30)    # BGR of RGB(30,58,138)   -> gradient top (deep blue)
BG_BOTTOM     = (35, 16, 10)     # BGR of RGB(10,16,35)    -> gradient bottom (navy)
NAME          = "HERMES"
NAME_ACCENT   = "◆"              # little hermes mark next to the name

# The canonical FaceMesh "face oval" contour indices (468-landmark model).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Facial feature landmark groups (468-point model)
EYES_LEFT  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYES_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROW_LEFT  = [46, 53, 52, 65, 55]
BROW_RIGHT = [285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94]
FEATURE = (12, 20, 45)  # BGR dark navy feature lines (bolder) on the blue fill


def build_gradient(w, h, top=BG_TOP, bottom=BG_BOTTOM):
    """Vertical gradient background (bgr). Precompute once per resolution."""
    top = np.array(top, dtype=np.uint8)
    bottom = np.array(bottom, dtype=np.uint8)
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    grad = (top * (1 - t) + bottom * t).astype(np.uint8)
    return np.repeat(grad, w, axis=1).copy()


def draw_silhouette(frame, pts_norm, fill=True, scale_x=1.0, scale_y=1.0):
    """Fill the head+neck+shoulders polygon as the 'you silhouette'.
    pts_norm: list of (x, y) normalized [0,1] face-oval points (in FACE_OVAL order)."""
    h, w = frame.shape[:2]
    pts = [[int(x * w), int(y * h)] for x, y in pts_norm]
    # extend chin down into neck + shoulders (bust silhouette)
    chin = pts[0] if len(pts) else [0, 0]
    for x, y in pts:
        if y > chin[1]:
            chin = [x, y]  # lowest point ~ chin (idx 152)
    cx0, cy0 = chin
    neck_w = int(w * 0.16)
    sh_w = int(w * 0.46)   # shoulder width
    sh_y = int(h * 1.10)   # bottom (below frame edge)
    neck_y = int(cy0 + (sh_y - cy0) * 0.30)
    # taper: chin -> neck (narrow) -> shoulders (wide)
    pts += [[cx0 - neck_w, neck_y], [cx0 + neck_w, neck_y],
            [cx0 - sh_w, sh_y], [cx0 + sh_w, sh_y]]
    pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(frame, [pts], SILHOUETTE)
    # white inner outline
    cv2.polylines(frame, [pts], True, OUTLINE_WHITE, 2, cv2.LINE_AA)
    return frame


def draw_features(out, lm, w, h):
    """Draw eyes, eyebrows, mouth and nose as stylized outlines."""
    def polyline(idxs, close=True, thick=3):
        p = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in idxs]
        cv2.polylines(out, [np.array(p, np.int32).reshape(-1, 1, 2)],
                      close, FEATURE, thick, cv2.LINE_AA)
    for e in (EYES_LEFT, EYES_RIGHT):
        p = np.array([(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in e],
                     np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [p], FEATURE)
        cv2.polylines(out, [p], True, OUTLINE_WHITE, 1, cv2.LINE_AA)
    polyline(BROW_LEFT, close=False, thick=3)
    polyline(BROW_RIGHT, close=False, thick=3)
    p_out = np.array([(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in LIPS_OUTER],
                     np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [p_out], True, FEATURE, 3, cv2.LINE_AA)
    p_in = np.array([(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in LIPS_INNER],
                    np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(out, [p_in], FEATURE)
    polyline(NOSE, close=False, thick=3)
    nx, ny = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
    cv2.circle(out, (nx, ny), max(2, int(w * 0.006)), FEATURE, -1)


def draw_glow_border(frame, color=OUTLINE_WHITE, width=3):
    """Subtle accent frame around the whole output (keeps it on-brand)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (6, 6), (w - 7, h - 7), ACCENT_BLUE, 2)
    cv2.rectangle(frame, (3, 3), (w - 4, h - 4), color, 1)


def draw_nameplate(frame, text=NAME, mark=NAME_ACCENT):
    """White nameplate, bottom-left, with the Hermes mark."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"{mark}  {text}"
    (tw, th), _ = cv2.getTextSize(label, font, 0.9, 2)
    bx1, by1 = 24, h - 54
    bx2, by2 = bx1 + tw + 36, by1 + th + 34
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (8, 12, 28), -1)          # plate
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), ACCENT_BLUE, 2)           # blue edge
    cv2.putText(frame, label, (bx1 + 20, by1 + th + 12), font, 0.9,
                OUTLINE_WHITE, 2, cv2.LINE_AA)


class HermesWebcam:
    def __init__(self, cam_index=0, width=1280, height=720, fps=30,
                 model_path="models/face_landmarker.task"):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps

        # CPU FaceMesh — avoids mediapipe's flaky Metal/GPU path on mac.
        self.facemesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.gradient = build_gradient(width, height)

    # -- capture ---------------------------------------------------------
    def open_camera(self, mock=False):
        if mock:
            return MockCamera(self.width, self.height)
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam {self.cam_index}")
        return cap

    # -- one frame -------------------------------------------------------
    def process(self, frame_bgr, ts_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.facemesh.process(rgb)

        out = self.gradient.copy()
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            pts = []
            for idx in FACE_OVAL:
                p = lm.landmark[idx]
                pts.append((p.x, p.y))
            draw_silhouette(out, pts)
            h, w = out.shape[:2]
            draw_features(out, lm, w, h)
        draw_glow_border(out)
        draw_nameplate(out)
        return out

    def close(self):
        if getattr(self, "facemesh", None):
            self.facemesh.close()


class MockCamera:
    """Synthetic 'face' so the pipeline can be tested without a webcam."""
    def __init__(self, w, h):
        self.w, self.h = w, h

    def read(self):
        img = np.full((self.h, self.w, 3), (20, 24, 40), dtype=np.uint8)
        # crude head: big circle + shoulders-ish rect, enough to track
        cx, cy = self.w // 2, int(self.h * 0.42)
        cv2.ellipse(img, (cx, cy), (int(self.w * 0.22), int(self.h * 0.34)),
                    0, 0, 360, (140, 140, 160), -1)
        cv2.rectangle(img, (cx - int(self.w*0.30), int(self.h*0.78)),
                      (cx + int(self.w*0.30), self.h), (90, 96, 120), -1)
        return True, img

    def isOpened(self):
        return True

    def set(self, *a, **k):
        return True

    def release(self):
        pass


def main():
    ap = argparse.ArgumentParser(description="Hermes anti-dox webcam player")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--preview", action="store_true",
                    help="show a local window instead of OBS virtual cam")
    ap.add_argument("--mock", action="store_true",
                    help="use a synthetic face (no webcam needed)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", default="models/face_landmarker.task")
    args = ap.parse_args()

    player = HermesWebcam(args.cam, args.width, args.height, args.fps,
                          args.model)
    cam = player.open_camera(mock=args.mock)

    vcam = None
    if not args.preview:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=args.width, height=args.height,
                                   fps=args.fps)
        print(f"[hermes] -> OBS Virtual Camera: {vcam.device} "
              f"{args.width}x{args.height}@{args.fps}fps")

    try:
        t0 = time.monotonic()
        frame_count = 0
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            ts_ms = int((time.monotonic() - t0) * 1000)
            out = player.process(frame, ts_ms)
            frame_count += 1

            if args.preview:
                cv2.imshow("HERMES anti-dox player", out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                vcam.send(out)
                vcam.sleep_until_next_frame()
    finally:
        cam.release()
        if vcam:
            vcam.close()
        player.close()
        if args.preview:
            cv2.destroyAllWindows()
    print(f"[hermes] done ({frame_count} frames)")


if __name__ == "__main__":
    main()
