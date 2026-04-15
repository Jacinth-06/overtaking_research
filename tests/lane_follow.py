"""
lane_follow.py — GPU-accelerated dual-lane follower + Flask dashboard
Optimised for Jetson Nano 4 GB, black road with white lines on both sides.

Detection strategy:
  • HSV mask isolates white lane lines
  • ROI split into left zone / right zone
  • Largest contour in each zone → left_cx, right_cx
  • Lane center = midpoint (or inferred from one line + expected width)
  • Lane-loss memory prevents edge crashes

Run:   python lane_follow.py
Open:  http://<jetson-ip>:5000
"""

import cv2
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Response, render_template_string, request, jsonify

from jetracer import JetRacer

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"        # Disable MSMF (Windows)
# GStreamer MUST stay enabled — CSI camera outputs raw Bayer (RG10)
# and needs nvarguscamerasrc for ISP debayering
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"      # Disable FFMPEG

app = Flask(__name__)

# ── CUDA availability check ───────────────────────────────────────────────────
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_CUDA:
    print("[init] CUDA device found — GPU path active")
    _gpu_frame = cv2.cuda_GpuMat()
    _gpu_hsv   = cv2.cuda_GpuMat()
    _gpu_mask  = cv2.cuda_GpuMat()
else:
    print("[init] No CUDA device — falling back to CPU")
    _gpu_frame = _gpu_hsv = _gpu_mask = None

# ── Config constants ──────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240          # lower res = less GPU/CPU work
ENCODE_EVERY    = 3                  # encode JPEG only every Nth frame
JPEG_QUALITY    = 30                 # lower = smaller payload, less CPU
MJPEG_INTERVAL  = 1 / 15            # 15 fps to browser
ROI_FRAC        = 0.60               # bottom 40% used as ROI
MAX_LOST_FRAMES = 15                 # ~0.5s at 30fps → full stop

# Pre-allocate morphological kernel (constant — no need to recreate per frame)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "h_lo": 0,  "h_hi": 180,
    "s_lo": 0,  "s_hi": 50,
    "v_lo": 200,  "v_hi": 255,
    "kp": 0.55,   "ki": 0.003,  "kd": 0.25,
    "speed": 0.15,
    "enabled": False,
    "min_contour_area": 300,       # per-zone contour threshold (lower since split)
    "lane_width": 200,             # expected pixel distance between lane lines
    "recovery_steer": 0.6,        # multiplier for lane-loss recovery steering
    # telemetry (read-only from browser)
    "error": 0.0,  "steer": 0.0,  "fps": 0,
    "lane_found": False,
    "left_det": False, "right_det": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()

# Lane-loss memory (accessed only from control-loop thread — no lock needed)
_last_good_error = 0.0
_lost_count      = 0

# Latest JPEG bytes for MJPEG stream
frame_lock   = threading.Lock()
latest_frame = None

# Count of active MJPEG clients — skip annotation when 0
stream_clients = 0
clients_lock   = threading.Lock()

# Async JPEG encoder (1 worker is enough; encoding is sequential)
_encode_pool = ThreadPoolExecutor(max_workers=1)


# ── Camera ────────────────────────────────────────────────────────────────────
def _gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280, capture_height=720,
    display_width=WIDTH, display_height=HEIGHT,
    framerate=60, flip_method=0,
):
    """
    Build a GStreamer pipeline string for nvarguscamerasrc (Jetson CSI cameras).
    The ISP handles debayering RG10 → NV12 in hardware, then we convert to BGR
    for OpenCV.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 max-buffers=1"
    )


def open_camera():
    """
    Open the CSI camera via nvarguscamerasrc GStreamer pipeline.
    Falls back to USB /dev/video0 if GStreamer pipeline fails.
    """
    # --- Try CSI camera via GStreamer (nvarguscamerasrc) ---
    gst = _gstreamer_pipeline()
    print(f"[camera] Trying GStreamer pipeline:\n  {gst}")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] CSI camera via nvarguscamerasrc {WIDTH}×{HEIGHT} OK")
        return cap
    print("[camera] GStreamer pipeline failed, trying USB fallback...")

    # --- Fallback: USB camera ---
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[camera] USB /dev/video0 {WIDTH}×{HEIGHT} OK")
        return cap

    raise RuntimeError("No camera found (CSI and USB both failed)")


# ── GPU HSV mask ──────────────────────────────────────────────────────────────
# Detect which cv2.cuda.cvtColor API is available at startup.
# OCV 4.5.x  → returns a new GpuMat  (functional style)
# OCV 4.6+   → writes into dst arg   (in-place style)
def _make_gpu_hsv_mask():
    """
    Factory: returns the correct gpu_hsv_mask implementation for this
    OpenCV build so the if-branch is paid once, not every frame.
    """
    # Probe with a 1×1 dummy to see which API works
    probe = cv2.cuda_GpuMat()
    probe.upload(np.zeros((1, 1, 3), dtype=np.uint8))
    try:
        result = cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2HSV)
        if result is not None and not result.empty():
            # Functional API (OCV 4.5.x) — cvtColor returns the output GpuMat
            print("[cuda] cvtColor: functional API (returns GpuMat)")
            def _mask_functional(roi_bgr, lo, hi):
                _gpu_frame.upload(roi_bgr)
                hsv_gpu = cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV)
                hsv_cpu = hsv_gpu.download()
                return cv2.inRange(hsv_cpu, lo, hi)
            return _mask_functional
    except Exception:
        pass

    # In-place API (OCV 4.6+) — cvtColor writes into dst
    print("[cuda] cvtColor: in-place API (dst arg)")
    def _mask_inplace(roi_bgr, lo, hi):
        _gpu_frame.upload(roi_bgr)
        cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV, _gpu_hsv)
        hsv_cpu = _gpu_hsv.download()
        return cv2.inRange(hsv_cpu, lo, hi)
    return _mask_inplace


def cpu_hsv_mask(roi_bgr, lo, hi):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lo, hi)


# Build the correct GPU mask fn for this OpenCV version (probed once at startup)
gpu_hsv_mask = _make_gpu_hsv_mask() if USE_CUDA else None


# ── Dual-lane detection + PID ─────────────────────────────────────────────────
def process_frame(frame, s, annotate: bool):
    """
    Detect left and right white lane lines, compute lane center error, run PID.

    Zone layout within the ROI:
        LEFT_ZONE:  x in [0, 0.40 * w)
        DEAD_ZONE:  x in [0.40 * w, 0.60 * w)   (ignored — road centre)
        RIGHT_ZONE: x in [0.60 * w, w)
    """
    global _last_good_error, _lost_count

    h, w = frame.shape[:2]
    roi_top = int(h * ROI_FRAC)
    roi = frame[roi_top:h, :]

    lo = np.array([s["h_lo"], s["s_lo"], s["v_lo"]], dtype=np.uint8)
    hi = np.array([s["h_hi"], s["s_hi"], s["v_hi"]], dtype=np.uint8)

    # --- Mask: GPU or CPU ---
    mask = gpu_hsv_mask(roi, lo, hi) if gpu_hsv_mask is not None else cpu_hsv_mask(roi, lo, hi)

    # --- Morphological cleanup (cached kernel) ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    # --- Contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_boundary  = int(w * 0.40)
    right_boundary = int(w * 0.60)
    min_area       = s["min_contour_area"]
    lane_width_px  = s["lane_width"]

    # Classify contours into left/right zones by centroid x
    left_cx, right_cx = None, None
    left_cy, right_cy = None, None
    best_left_area, best_right_area = 0, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if cx < left_boundary and area > best_left_area:
            left_cx, left_cy = cx, cy
            best_left_area = area
        elif cx > right_boundary and area > best_right_area:
            right_cx, right_cy = cx, cy
            best_right_area = area

    # --- Compute lane center ---
    l_det = left_cx is not None
    r_det = right_cx is not None
    lane_found = l_det or r_det

    if l_det and r_det:
        lane_center = (left_cx + right_cx) // 2
    elif l_det:
        lane_center = left_cx + lane_width_px // 2
    elif r_det:
        lane_center = right_cx - lane_width_px // 2
    else:
        lane_center = w // 2   # fallback — error will use memory below

    # --- Error ---
    if lane_found:
        error = (lane_center - w // 2) / (w // 2)
        _last_good_error = error
        _lost_count = 0
    else:
        # Lane lost — steer in the last known direction with recovery multiplier
        _lost_count += 1
        error = _last_good_error * s["recovery_steer"]

    # --- PID ---
    now = time.time()
    dt  = max(now - pid_state["last_time"], 0.001)
    pid_state["integral"]  += error * dt
    pid_state["integral"]   = max(-1.0, min(1.0, pid_state["integral"]))
    derivative              = (error - pid_state["last_error"]) / dt
    pid_state["last_error"] = error
    pid_state["last_time"]  = now

    steer = (s["kp"] * error
           + s["ki"] * pid_state["integral"]
           + s["kd"] * derivative)
    steer = max(-1.0, min(1.0, steer))

    # --- Annotate only when a browser is watching ---
    if annotate:
        annotated = frame.copy()
        # ROI boundary line
        cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)

        # Zone boundaries (faint vertical lines)
        cv2.line(annotated, (left_boundary, roi_top), (left_boundary, h), (100, 100, 100), 1)
        cv2.line(annotated, (right_boundary, roi_top), (right_boundary, h), (100, 100, 100), 1)

        # Mask overlay
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_3ch[:, :, 0] = 0
        annotated[roi_top:h, :] = cv2.addWeighted(
            annotated[roi_top:h, :], 0.7, mask_3ch, 0.3, 0)

        cy_mid = roi_top + (h - roi_top) // 2

        # Left lane marker
        if l_det:
            cv2.circle(annotated, (left_cx, roi_top + left_cy), 8, (0, 255, 0), -1)
            cv2.circle(annotated, (left_cx, roi_top + left_cy), 8, (255, 255, 255), 2)
        elif r_det:
            # Ghost marker for inferred left position
            inferred_lx = right_cx - lane_width_px
            cv2.circle(annotated, (max(0, inferred_lx), cy_mid), 6, (0, 0, 200), 2)

        # Right lane marker
        if r_det:
            cv2.circle(annotated, (right_cx, roi_top + right_cy), 8, (0, 255, 0), -1)
            cv2.circle(annotated, (right_cx, roi_top + right_cy), 8, (255, 255, 255), 2)
        elif l_det:
            # Ghost marker for inferred right position
            inferred_rx = left_cx + lane_width_px
            cv2.circle(annotated, (min(w - 1, inferred_rx), cy_mid), 6, (0, 0, 200), 2)

        # Lane center marker (cyan)
        if lane_found:
            cv2.circle(annotated, (lane_center, cy_mid), 6, (255, 200, 0), -1)

        # Frame center line
        cv2.line(annotated, (w // 2, roi_top), (w // 2, h), (0, 200, 255), 1)

        # Steer arrow
        arrow_x = int(w // 2 + steer * (w // 3))
        cv2.arrowedLine(annotated, (w // 2, 22), (arrow_x, 22),
                        (0, 140, 255), 2, tipLength=0.35)

        # Status text
        status = "DRIVING" if s["enabled"] else "STOPPED"
        color  = (0, 220, 60) if s["enabled"] else (60, 60, 220)
        cv2.putText(annotated, status, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(annotated, f"e{error:+.2f} s{steer:+.2f}", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(annotated, f"fps {s['fps']}", (5, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        # Lane detection indicators
        l_tag = "L\u2713" if l_det else "L\u2717"
        r_tag = "R\u2713" if r_det else "R\u2717"
        l_col = (0, 220, 60) if l_det else (0, 60, 220)
        r_col = (0, 220, 60) if r_det else (0, 60, 220)
        cv2.putText(annotated, l_tag, (w - 60, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_col, 1)
        cv2.putText(annotated, r_tag, (w - 30, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_col, 1)

        if _lost_count > 0:
            cv2.putText(annotated, f"LOST {_lost_count}", (w // 2 - 30, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    else:
        annotated = frame   # no copy needed

    return annotated, error, steer, lane_found, l_det, r_det


# ── Async JPEG encode ─────────────────────────────────────────────────────────
def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return
    with frame_lock:
        global latest_frame
        latest_frame = jpeg.tobytes()


# ── Control loop ──────────────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    frame_idx = 0
    print("[loop] Control loop started")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Ensure frame is 3D (BGR)
        if len(frame.shape) != 3:
            continue

        if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with state_lock:
            s_copy = dict(state)

        with clients_lock:
            has_clients = stream_clients > 0

        # Annotate only if someone is watching; encode every Nth frame
        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        annotated, error, steer, lane_found, l_det, r_det = process_frame(frame, s_copy, do_annotate)

        # Async encode (fire-and-forget)
        if do_annotate:
            _encode_pool.submit(_do_encode, annotated)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        # Drive
        if s_copy["enabled"]:
            if lane_found:
                car.steer(steer)
                car.forward(s_copy["speed"])
            elif _lost_count < MAX_LOST_FRAMES:
                # Lane lost but within recovery window — steer using memory
                car.steer(steer)
                # Gradually reduce speed as we stay lost
                decay = max(0.3, 1.0 - _lost_count / MAX_LOST_FRAMES)
                car.forward(s_copy["speed"] * decay)
            else:
                # Lost for too long — full stop
                car.stop()
        else:
            car.stop()

        with state_lock:
            state["error"]      = round(error, 3)
            state["steer"]      = round(steer, 3)
            state["lane_found"] = lane_found
            state["left_det"]   = l_det
            state["right_det"]  = r_det

        frame_idx += 1

    cap.release()


# ── Flask / dashboard ───────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetRacer Lane Follower</title>
<style>
  :root {
    --bg: #0e1117; --surface: #161b27; --border: #2a3040;
    --accent: #00d4aa; --warn: #ffb020; --danger: #ff4d4d;
    --text: #e8ecf1; --muted: #6b7a99;
    --font: 'JetBrains Mono', 'Fira Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 1rem; }
  h1 { font-size: 1rem; letter-spacing: .15em; color: var(--accent);
       text-transform: uppercase; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 340px; gap: 1rem; width: 100%; max-width: 1100px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
  .card h2 { font-size: .7rem; letter-spacing: .12em; color: var(--muted); text-transform: uppercase;
             margin-bottom: .75rem; }
  img#feed { width: 100%; border-radius: 6px; display: block; background: #000; min-height: 180px; }
  .status-bar { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: .75rem; }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
  .lane-ind { display: inline-block; font-size: .85rem; font-weight: 700; padding: 2px 6px;
              border-radius: 4px; margin-right: 4px; }
  .lane-ok  { background: #0a3a2a; color: #00d4aa; }
  .lane-no  { background: #3a1a1a; color: #ff4d4d; }
  .slider-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .55rem; }
  .slider-row label { font-size: .7rem; color: var(--muted); width: 65px; flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val { font-size: .75rem; width: 45px; text-align: right; color: var(--text); }
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; }
  button { padding: .45rem 1.1rem; border: none; border-radius: 6px; cursor: pointer;
           font-family: var(--font); font-size: .8rem; font-weight: 600; letter-spacing: .04em; }
  #btn-go   { background: var(--accent); color: #061612; }
  #btn-stop { background: var(--danger); color: #fff; }
  #btn-go:hover   { filter: brightness(1.1); }
  #btn-stop:hover { filter: brightness(1.1); }
  .error-track { position: relative; height: 18px; background: var(--border);
                 border-radius: 9px; margin-top: .5rem; overflow: hidden; }
  #error-bar { position: absolute; height: 100%; width: 4px; background: var(--accent);
               left: 50%; transform: translateX(-50%); transition: left .1s; border-radius: 9px; }
  .divider { border: none; border-top: 1px solid var(--border); margin: .75rem 0; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>&#9675; JetRacer &#183; Dual-Lane Dashboard</h1>
<div class="grid">
  <div class="card">
    <h2>Camera feed (annotated)</h2>
    <img id="feed" src="/video_feed" alt="camera">
    <div class="status-bar" style="margin-top:.75rem">
      <div class="stat"><span class="stat-val" id="v-fps">0</span><span class="stat-lbl">fps</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span><span class="stat-lbl">error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span><span class="stat-lbl">steer</span></div>
      <div class="stat">
        <span class="stat-val" id="v-lane">
          <span id="v-left" class="lane-ind lane-no">L</span>
          <span id="v-right" class="lane-ind lane-no">R</span>
        </span>
        <span class="stat-lbl">lanes</span>
      </div>
    </div>
    <div class="error-track" title="Lane error (centre = 0)">
      <div id="error-bar"></div>
    </div>
  </div>
  <div class="card">
    <h2>Drive</h2>
    <div class="slider-row">
      <label>Speed</label>
      <input type="range" id="speed" min="0" max="60" value="15" step="1">
      <span class="val" id="v-speed">0.15</span>
    </div>
    <div class="btn-row">
      <button id="btn-go"   onclick="setEnabled(true)">&#9654; GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
    </div>
    <hr class="divider">
    <h2>PID gains</h2>
    <div class="slider-row">
      <label>Kp</label>
      <input type="range" id="kp" min="0" max="1" value="0.55" step="0.01">
      <span class="val" id="v-kp">0.55</span>
    </div>
    <div class="slider-row">
      <label>Ki</label>
      <input type="range" id="ki" min="0" max="0.05" value="0.003" step="0.001">
      <span class="val" id="v-ki">0.003</span>
    </div>
    <div class="slider-row">
      <label>Kd</label>
      <input type="range" id="kd" min="0" max="0.5" value="0.25" step="0.01">
      <span class="val" id="v-kd">0.25</span>
    </div>
    <hr class="divider">
    <h2>Lane detection</h2>
    <div class="slider-row">
      <label>Lane W</label>
      <input type="range" id="lane_width" min="50" max="300" value="200" step="5">
      <span class="val" id="v-lane_width">200</span>
    </div>
    <div class="slider-row">
      <label>Recovery</label>
      <input type="range" id="recovery_steer" min="0" max="1" value="0.6" step="0.05">
      <span class="val" id="v-recovery_steer">0.60</span>
    </div>
    <div class="slider-row">
      <label>Min area</label>
      <input type="range" id="min_contour_area" min="50" max="5000" value="300" step="50">
      <span class="val" id="v-min_contour_area">300</span>
    </div>
    <hr class="divider">
    <h2>HSV mask</h2>
    <div class="slider-row">
      <label>H lo</label>
      <input type="range" id="h_lo" min="0" max="179" value="0" step="1">
      <span class="val" id="v-h_lo">0</span>
    </div>
    <div class="slider-row">
      <label>H hi</label>
      <input type="range" id="h_hi" min="0" max="179" value="180" step="1">
      <span class="val" id="v-h_hi">180</span>
    </div>
    <div class="slider-row">
      <label>S lo</label>
      <input type="range" id="s_lo" min="0" max="255" value="0" step="1">
      <span class="val" id="v-s_lo">0</span>
    </div>
    <div class="slider-row">
      <label>S hi</label>
      <input type="range" id="s_hi" min="0" max="255" value="50" step="1">
      <span class="val" id="v-s_hi">50</span>
    </div>
    <div class="slider-row">
      <label>V lo</label>
      <input type="range" id="v_lo" min="0" max="255" value="200" step="1">
      <span class="val" id="v-v_lo">200</span>
    </div>
    <div class="slider-row">
      <label>V hi</label>
      <input type="range" id="v_hi" min="0" max="255" value="255" step="1">
      <span class="val" id="v-v_hi">255</span>
    </div>
  </div>
</div>
<script>
const sliders = ["speed","kp","ki","kd","h_lo","h_hi","s_lo","s_hi","v_lo","v_hi",
                 "min_contour_area","lane_width","recovery_steer"];
sliders.forEach(id => {
  const el = document.getElementById(id);
  const disp = document.getElementById("v-"+id);
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (id === "speed") {
      disp.textContent = (v/100).toFixed(2);
      sendParam(id, v/100);
    } else if (id === "recovery_steer") {
      disp.textContent = v.toFixed(2);
      sendParam(id, v);
    } else {
      disp.textContent = Number.isInteger(v) ? v : v.toFixed(3);
      sendParam(id, v);
    }
  });
});
function sendParam(key, value) {
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({[key]: value})});
}
function setEnabled(v) {
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({enabled: v})});
}
async function poll() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    document.getElementById("v-fps").textContent  = d.fps;
    document.getElementById("v-err").textContent  = d.error.toFixed(2);
    document.getElementById("v-str").textContent  = d.steer.toFixed(2);

    const leftEl  = document.getElementById("v-left");
    const rightEl = document.getElementById("v-right");
    leftEl.className  = "lane-ind " + (d.left_det  ? "lane-ok" : "lane-no");
    rightEl.className = "lane-ind " + (d.right_det ? "lane-ok" : "lane-no");
    leftEl.textContent  = d.left_det  ? "L\u2713" : "L\u2717";
    rightEl.textContent = d.right_det ? "R\u2713" : "R\u2717";

    const pct = (d.error + 1) / 2 * 100;
    const bar = document.getElementById("error-bar");
    bar.style.left = pct + "%";
    bar.style.background = Math.abs(d.error) > 0.5 ? "#ff4d4d" : "#00d4aa";
  } catch(e) {}
  setTimeout(poll, 250);
}
poll();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def generate_mjpeg():
    global stream_clients
    with clients_lock:
        stream_clients += 1
    try:
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.02)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(MJPEG_INTERVAL)
    finally:
        with clients_lock:
            stream_clients -= 1


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        return jsonify({k: state[k] for k in
                        ("fps", "error", "steer", "lane_found", "enabled",
                         "left_det", "right_det")})


@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k in state:
                state[k] = v
                if k in ("kp", "ki", "kd"):
                    pid_state["integral"]   = 0.0
                    pid_state["last_error"] = 0.0
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = JetRacer()
    car.arm(delay=3)

    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    # use_reloader=False is critical — reloader forks and doubles CPU load
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)