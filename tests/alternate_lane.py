#!/usr/bin/env python3
"""
alternate_lane.py — GPU-accelerated bird-eye lane follower + Flask dashboard
Optimised for Jetson Nano 4 GB.

Detection pipeline:
  frame → grayscale → Gaussian blur → (Canny + Binary threshold)
  → OR combine → morphology clean → bird-eye warp
  → histogram → sliding-window lane detect → lane centre → PID → motor

Run:   python alternate_lane.py
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
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

app = Flask(__name__)

# ── CUDA availability check ───────────────────────────────────────────────────
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_CUDA:
    print("[init] CUDA device found — GPU path active")
    _gpu_frame = cv2.cuda_GpuMat()
    _gpu_gray  = cv2.cuda_GpuMat()
    _gpu_blur  = cv2.cuda_GpuMat()
    _gpu_tmp   = cv2.cuda_GpuMat()
else:
    print("[init] No CUDA device — falling back to CPU")
    _gpu_frame = _gpu_gray = _gpu_blur = _gpu_tmp = None

# ── Config constants ──────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240
ENCODE_EVERY    = 3
JPEG_QUALITY    = 30
MJPEG_INTERVAL  = 1 / 15

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    # Canny
    "canny_lo": 50,   "canny_hi": 150,
    # Binary threshold
    "binary_thresh": 200,
    # Gaussian blur
    "blur_ksize": 5,
    # Morphology
    "morph_ksize": 5,  "morph_iters": 2,
    # Bird-eye ROI quad (fractions of frame w/h)
    "bev_tl_x": 0.30, "bev_tl_y": 0.60,
    "bev_tr_x": 0.70, "bev_tr_y": 0.60,
    "bev_br_x": 0.95, "bev_br_y": 0.95,
    "bev_bl_x": 0.05, "bev_bl_y": 0.95,
    # Histogram / sliding window
    "hist_row_frac": 0.5,
    "sw_nwindows": 9,
    "sw_margin": 60,
    "sw_minpix": 25,
    # PID
    "kp": 0.55,  "ki": 0.003,  "kd": 0.25,
    # Drive
    "speed": 0.15,
    "enabled": False,
    # Telemetry (read-only from browser)
    "error": 0.0, "steer": 0.0, "fps": 0,
    "lane_found": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()

_last_steer = 0.0

# Latest JPEG bytes for MJPEG stream
frame_lock   = threading.Lock()
latest_frame = None

# Count of active MJPEG clients — skip annotation when 0
stream_clients = 0
clients_lock   = threading.Lock()

# Async JPEG encoder
_encode_pool = ThreadPoolExecutor(max_workers=1)


# ── Camera ────────────────────────────────────────────────────────────────────
def _gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280, capture_height=720,
    display_width=WIDTH, display_height=HEIGHT,
    framerate=60, flip_method=0,
):
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
    gst = _gstreamer_pipeline()
    print(f"[camera] Trying GStreamer pipeline:\n  {gst}")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] CSI camera via nvarguscamerasrc {WIDTH}×{HEIGHT} OK")
        return cap
    print("[camera] GStreamer pipeline failed, trying USB fallback...")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[camera] USB /dev/video0 {WIDTH}×{HEIGHT} OK")
        return cap

    raise RuntimeError("No camera found (CSI and USB both failed)")


# ── GPU helpers (probed at startup, like line_follow.py) ──────────────────────
def _make_gpu_grayscale():
    """Probe cv2.cuda.cvtColor for BGR→GRAY; return a callable or None."""
    probe = cv2.cuda_GpuMat()
    probe.upload(np.zeros((1, 1, 3), dtype=np.uint8))
    try:
        result = cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        if result is not None and not result.empty():
            print("[cuda] cvtColor BGR→GRAY: functional API")
            def _gray_func(bgr_cpu):
                _gpu_frame.upload(bgr_cpu)
                g = cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2GRAY)
                return g.download()
            return _gray_func
    except Exception:
        pass

    try:
        cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2GRAY, _gpu_gray)
        print("[cuda] cvtColor BGR→GRAY: in-place API")
        def _gray_inplace(bgr_cpu):
            _gpu_frame.upload(bgr_cpu)
            cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2GRAY, _gpu_gray)
            return _gpu_gray.download()
        return _gray_inplace
    except Exception:
        pass

    print("[cuda] cvtColor BGR→GRAY: FAILED — will use CPU")
    return None

gpu_grayscale = _make_gpu_grayscale() if USE_CUDA else None

def cpu_grayscale(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def to_gray(bgr):
    if gpu_grayscale is not None:
        return gpu_grayscale(bgr)
    return cpu_grayscale(bgr)


# ── Bird-eye perspective ──────────────────────────────────────────────────────
_bev_M    = None
_bev_Minv = None
_bev_last = None

def _build_bev(W, H, s):
    """Rebuild perspective matrices when BEV params change."""
    global _bev_M, _bev_Minv, _bev_last
    key = (s["bev_tl_x"], s["bev_tl_y"], s["bev_tr_x"], s["bev_tr_y"],
           s["bev_br_x"], s["bev_br_y"], s["bev_bl_x"], s["bev_bl_y"], W, H)
    if key == _bev_last:
        return
    src = np.float32([
        [s["bev_tl_x"] * W, s["bev_tl_y"] * H],
        [s["bev_tr_x"] * W, s["bev_tr_y"] * H],
        [s["bev_br_x"] * W, s["bev_br_y"] * H],
        [s["bev_bl_x"] * W, s["bev_bl_y"] * H],
    ])
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    _bev_M    = cv2.getPerspectiveTransform(src, dst)
    _bev_Minv = cv2.getPerspectiveTransform(dst, src)
    _bev_last = key
    print(f"[bev] Perspective matrix rebuilt for {W}×{H}")


# ── Sliding-window lane detection ─────────────────────────────────────────────
_left_base_prev  = None
_right_base_prev = None

def detect_lanes(bev_binary, s):
    """
    Histogram + sliding-window on a binary bird-eye image.
    Returns (center_x, found, dbg_img).
    """
    global _left_base_prev, _right_base_prev

    H, W = bev_binary.shape
    row_start = int(s["hist_row_frac"] * H)
    margin    = int(s["sw_margin"])
    nwindows  = int(s["sw_nwindows"])
    minpix    = int(s["sw_minpix"])

    # ── Histogram on lower portion ────────────────────────────────────────
    hist = np.sum(bev_binary[row_start:, :].astype(np.int32), axis=0)
    midpoint = W // 2

    left_base  = int(np.argmax(hist[:midpoint]))
    right_base = int(np.argmax(hist[midpoint:]) + midpoint)

    # IIR smooth with previous frame
    alpha = 0.6
    if _left_base_prev is not None:
        left_base  = int(alpha * left_base  + (1 - alpha) * _left_base_prev)
        right_base = int(alpha * right_base + (1 - alpha) * _right_base_prev)
    _left_base_prev  = left_base
    _right_base_prev = right_base

    # ── Sliding windows ───────────────────────────────────────────────────
    win_h = max(H // nwindows, 1)
    nonzero  = bev_binary.nonzero()
    nzy, nzx = np.array(nonzero[0]), np.array(nonzero[1])

    left_cur, right_cur = left_base, right_base
    left_idx_all, right_idx_all = [], []

    dbg = cv2.cvtColor(bev_binary, cv2.COLOR_GRAY2BGR)

    for w_i in range(nwindows):
        y_lo = H - (w_i + 1) * win_h
        y_hi = H - w_i * win_h
        xl_lo, xl_hi = left_cur  - margin, left_cur  + margin
        xr_lo, xr_hi = right_cur - margin, right_cur + margin

        cv2.rectangle(dbg, (xl_lo, y_lo), (xl_hi, y_hi), (0, 255, 0), 1)
        cv2.rectangle(dbg, (xr_lo, y_lo), (xr_hi, y_hi), (0, 0, 255), 1)

        good_l = ((nzy >= y_lo) & (nzy < y_hi) &
                  (nzx >= xl_lo) & (nzx < xl_hi)).nonzero()[0]
        good_r = ((nzy >= y_lo) & (nzy < y_hi) &
                  (nzx >= xr_lo) & (nzx < xr_hi)).nonzero()[0]

        left_idx_all.append(good_l)
        right_idx_all.append(good_r)

        if len(good_l) > minpix:
            left_cur = int(np.mean(nzx[good_l]))
        if len(good_r) > minpix:
            right_cur = int(np.mean(nzx[good_r]))

    left_idx  = np.concatenate(left_idx_all)  if left_idx_all  else np.array([])
    right_idx = np.concatenate(right_idx_all) if right_idx_all else np.array([])

    if len(left_idx) < 5 or len(right_idx) < 5:
        return None, False, dbg

    # ── Polynomial fit (2nd order) ────────────────────────────────────────
    ly, lx = nzy[left_idx],  nzx[left_idx]
    ry, rx = nzy[right_idx], nzx[right_idx]

    try:
        left_fit  = np.polyfit(ly, lx, 2)
        right_fit = np.polyfit(ry, rx, 2)
    except np.linalg.LinAlgError:
        return None, False, dbg

    # ── Lane centre at bottom row ─────────────────────────────────────────
    bot = H - 1
    left_x  = np.polyval(left_fit,  bot)
    right_x = np.polyval(right_fit, bot)
    center_x = (left_x + right_x) / 2.0

    # Draw fitted curves on debug image
    ys = np.linspace(0, H - 1, H).astype(int)
    lxs = np.clip(np.polyval(left_fit,  ys).astype(int), 0, W - 1)
    rxs = np.clip(np.polyval(right_fit, ys).astype(int), 0, W - 1)
    for y_val, lx_val, rx_val in zip(ys, lxs, rxs):
        cv2.circle(dbg, (lx_val, y_val), 1, (255, 200, 0), -1)
        cv2.circle(dbg, (rx_val, y_val), 1, (0, 200, 255), -1)

    # Draw centre marker
    cv2.circle(dbg, (int(center_x), bot), 6, (0, 255, 255), -1)

    return center_x, True, dbg


# ── Frame processing pipeline ────────────────────────────────────────────────
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def process_frame(frame, s, annotate: bool):
    """
    Pipeline: grayscale → blur → (Canny + Binary) → OR → morphology
              → bird-eye → histogram → lane centre → PID
    """
    global _last_steer
    h, w = frame.shape[:2]

    # 1. Grayscale (GPU-accelerated if available)
    gray = to_gray(frame)

    # 2. Gaussian blur
    bk = s["blur_ksize"] | 1   # ensure odd
    blurred = cv2.GaussianBlur(gray, (bk, bk), 0)

    # 3a. Canny edge detection
    edges = cv2.Canny(blurred, s["canny_lo"], s["canny_hi"])

    # 3b. Binary threshold (white lines)
    _, binary = cv2.threshold(blurred, s["binary_thresh"], 255, cv2.THRESH_BINARY)

    # 4. OR combine
    combined = cv2.bitwise_or(edges, binary)

    # 5. Morphology clean
    mk = s["morph_ksize"] | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=s["morph_iters"])
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel, iterations=1)

    # 6. Bird-eye warp
    _build_bev(w, h, s)
    bev = cv2.warpPerspective(cleaned, _bev_M, (w, h))

    # 7. Histogram + sliding-window lane detection
    center_x, lane_found, bev_dbg = detect_lanes(bev, s)

    # 8. PID
    if lane_found and center_x is not None:
        error = (center_x - w / 2.0) / (w / 2.0)   # normalise to [-1, 1]

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
        _last_steer = steer
    else:
        error = 0.0
        steer = _last_steer   # hold last known steer

    # ── Annotate ──────────────────────────────────────────────────────────
    if annotate:
        # Build a side-by-side: original (left) + BEV debug (right)
        annotated = frame.copy()

        # Draw BEV source quad on main image
        pts = np.int32([
            [s["bev_tl_x"] * w, s["bev_tl_y"] * h],
            [s["bev_tr_x"] * w, s["bev_tr_y"] * h],
            [s["bev_br_x"] * w, s["bev_br_y"] * h],
            [s["bev_bl_x"] * w, s["bev_bl_y"] * h],
        ])
        cv2.polylines(annotated, [pts], True, (0, 255, 255), 1)

        # Frame center line
        cv2.line(annotated, (w // 2, 0), (w // 2, h), (0, 200, 255), 1)

        # Steer arrow
        arrow_x = int(w // 2 + steer * (w // 3))
        cv2.arrowedLine(annotated, (w // 2, 22), (arrow_x, 22),
                        (0, 140, 255), 2, tipLength=0.35)

        # Status text
        status = "DRIVING" if s["enabled"] else "STOPPED"
        color  = (0, 220, 60) if s["enabled"] else (60, 60, 220)
        cv2.putText(annotated, status, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(annotated, f"e{error:+.2f} s{steer:+.2f}", (5, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(annotated, f"fps {s['fps']}", (5, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        lane_col = (0, 220, 60) if lane_found else (0, 60, 220)
        cv2.putText(annotated, "OK" if lane_found else "NO", (w - 30, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_col, 1)

        # Resize BEV debug to fit bottom-right corner
        bev_small = cv2.resize(bev_dbg, (w // 3, h // 3))
        y1 = h - bev_small.shape[0]
        x1 = w - bev_small.shape[1]
        annotated[y1:h, x1:w] = bev_small
    else:
        annotated = frame

    return annotated, error, steer, lane_found


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

        if len(frame.shape) != 3:
            continue

        if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with state_lock:
            s_copy = dict(state)

        with clients_lock:
            has_clients = stream_clients > 0

        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        annotated, error, steer, lane_found = process_frame(frame, s_copy, do_annotate)

        if do_annotate:
            _encode_pool.submit(_do_encode, annotated)

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        if s_copy["enabled"]:
            car.steer(steer)
            car.forward(s_copy["speed"])
        else:
            car.stop()

        with state_lock:
            state["error"]      = round(error, 3)
            state["steer"]      = round(steer, 3)
            state["lane_found"] = lane_found

        frame_idx += 1

    cap.release()


# ── Flask / dashboard ───────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetRacer Bird-Eye Lane Follower</title>
<style>
  :root {
    --bg: #0e1117; --surface: #161b27; --border: #2a3040;
    --accent: #00d4aa; --warn: #ffb020; --danger: #ff4d4d;
    --text: #e8ecf1; --muted: #6b7a99;
    --font: 'JetBrains Mono', 'Fira Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         display: flex; flex-direction: column; align-items: center;
         min-height: 100vh; padding: 1rem; }
  h1 { font-size: 1rem; letter-spacing: .15em; color: var(--accent);
       text-transform: uppercase; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 360px;
          gap: 1rem; width: 100%; max-width: 1200px; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 1rem; }
  .card h2 { font-size: .7rem; letter-spacing: .12em; color: var(--muted);
             text-transform: uppercase; margin-bottom: .75rem; }
  img#feed { width: 100%; border-radius: 6px; display: block;
             background: #000; min-height: 180px; }
  .status-bar { display: flex; gap: 1.5rem; flex-wrap: wrap;
                margin-bottom: .75rem; }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase;
              letter-spacing: .08em; }
  .slider-row { display: flex; align-items: center; gap: .5rem;
                margin-bottom: .55rem; }
  .slider-row label { font-size: .7rem; color: var(--muted); width: 85px;
                      flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val { font-size: .75rem; width: 45px; text-align: right;
                     color: var(--text); }
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; }
  button { padding: .45rem 1.1rem; border: none; border-radius: 6px;
           cursor: pointer; font-family: var(--font); font-size: .8rem;
           font-weight: 600; letter-spacing: .04em; }
  #btn-go   { background: var(--accent); color: #061612; }
  #btn-stop { background: var(--danger); color: #fff; }
  #btn-go:hover   { filter: brightness(1.1); }
  #btn-stop:hover { filter: brightness(1.1); }
  .error-track { position: relative; height: 18px; background: var(--border);
                 border-radius: 9px; margin-top: .5rem; overflow: hidden; }
  #error-bar { position: absolute; height: 100%; width: 4px;
               background: var(--accent); left: 50%;
               transform: translateX(-50%); transition: left .1s;
               border-radius: 9px; }
  .divider { border: none; border-top: 1px solid var(--border);
             margin: .75rem 0; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>&#9675; JetRacer &#183; Bird-Eye Lane Dashboard</h1>
<div class="grid">
  <div class="card">
    <h2>Camera feed (annotated)</h2>
    <img id="feed" src="/video_feed" alt="camera">
    <div class="status-bar" style="margin-top:.75rem">
      <div class="stat"><span class="stat-val" id="v-fps">0</span>
                        <span class="stat-lbl">fps</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span>
                        <span class="stat-lbl">error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span>
                        <span class="stat-lbl">steer</span></div>
      <div class="stat"><span class="stat-val" id="v-lane">&mdash;</span>
                        <span class="stat-lbl">lane</span></div>
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
      <input type="range" id="kp" min="0" max="2" value="0.55" step="0.01">
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
    <h2>Vision pipeline</h2>
    <div class="slider-row">
      <label>Canny lo</label>
      <input type="range" id="canny_lo" min="0" max="255" value="50" step="1">
      <span class="val" id="v-canny_lo">50</span>
    </div>
    <div class="slider-row">
      <label>Canny hi</label>
      <input type="range" id="canny_hi" min="0" max="255" value="150" step="1">
      <span class="val" id="v-canny_hi">150</span>
    </div>
    <div class="slider-row">
      <label>Binary thr</label>
      <input type="range" id="binary_thresh" min="0" max="255" value="200" step="1">
      <span class="val" id="v-binary_thresh">200</span>
    </div>
    <div class="slider-row">
      <label>Blur ksize</label>
      <input type="range" id="blur_ksize" min="1" max="21" value="5" step="2">
      <span class="val" id="v-blur_ksize">5</span>
    </div>
    <div class="slider-row">
      <label>Morph ksize</label>
      <input type="range" id="morph_ksize" min="1" max="21" value="5" step="2">
      <span class="val" id="v-morph_ksize">5</span>
    </div>
    <div class="slider-row">
      <label>Morph iters</label>
      <input type="range" id="morph_iters" min="1" max="5" value="2" step="1">
      <span class="val" id="v-morph_iters">2</span>
    </div>
    <hr class="divider">
    <h2>Bird-eye ROI</h2>
    <div class="slider-row">
      <label>TL x</label>
      <input type="range" id="bev_tl_x" min="0" max="0.5" value="0.30" step="0.01">
      <span class="val" id="v-bev_tl_x">0.30</span>
    </div>
    <div class="slider-row">
      <label>TL y</label>
      <input type="range" id="bev_tl_y" min="0.3" max="0.9" value="0.60" step="0.01">
      <span class="val" id="v-bev_tl_y">0.60</span>
    </div>
    <div class="slider-row">
      <label>TR x</label>
      <input type="range" id="bev_tr_x" min="0.5" max="1" value="0.70" step="0.01">
      <span class="val" id="v-bev_tr_x">0.70</span>
    </div>
    <div class="slider-row">
      <label>TR y</label>
      <input type="range" id="bev_tr_y" min="0.3" max="0.9" value="0.60" step="0.01">
      <span class="val" id="v-bev_tr_y">0.60</span>
    </div>
    <div class="slider-row">
      <label>BR x</label>
      <input type="range" id="bev_br_x" min="0.5" max="1" value="0.95" step="0.01">
      <span class="val" id="v-bev_br_x">0.95</span>
    </div>
    <div class="slider-row">
      <label>BR y</label>
      <input type="range" id="bev_br_y" min="0.5" max="1" value="0.95" step="0.01">
      <span class="val" id="v-bev_br_y">0.95</span>
    </div>
    <div class="slider-row">
      <label>BL x</label>
      <input type="range" id="bev_bl_x" min="0" max="0.5" value="0.05" step="0.01">
      <span class="val" id="v-bev_bl_x">0.05</span>
    </div>
    <div class="slider-row">
      <label>BL y</label>
      <input type="range" id="bev_bl_y" min="0.5" max="1" value="0.95" step="0.01">
      <span class="val" id="v-bev_bl_y">0.95</span>
    </div>
    <hr class="divider">
    <h2>Sliding window</h2>
    <div class="slider-row">
      <label>N windows</label>
      <input type="range" id="sw_nwindows" min="3" max="20" value="9" step="1">
      <span class="val" id="v-sw_nwindows">9</span>
    </div>
    <div class="slider-row">
      <label>Margin</label>
      <input type="range" id="sw_margin" min="10" max="150" value="60" step="5">
      <span class="val" id="v-sw_margin">60</span>
    </div>
    <div class="slider-row">
      <label>Min pix</label>
      <input type="range" id="sw_minpix" min="5" max="100" value="25" step="5">
      <span class="val" id="v-sw_minpix">25</span>
    </div>
    <div class="slider-row">
      <label>Hist frac</label>
      <input type="range" id="hist_row_frac" min="0.2" max="0.9" value="0.5" step="0.05">
      <span class="val" id="v-hist_row_frac">0.5</span>
    </div>
  </div>
</div>
<script>
const sliders = [
  "speed","kp","ki","kd",
  "canny_lo","canny_hi","binary_thresh","blur_ksize",
  "morph_ksize","morph_iters",
  "bev_tl_x","bev_tl_y","bev_tr_x","bev_tr_y",
  "bev_br_x","bev_br_y","bev_bl_x","bev_bl_y",
  "sw_nwindows","sw_margin","sw_minpix","hist_row_frac"
];
sliders.forEach(id => {
  const el = document.getElementById(id);
  const disp = document.getElementById("v-"+id);
  if (!el) return;
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (id === "speed") {
      disp.textContent = (v/100).toFixed(2);
      sendParam(id, v/100);
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

    const laneEl = document.getElementById("v-lane");
    laneEl.textContent = d.lane_found ? "✓" : "✗";
    laneEl.style.color = d.lane_found ? "#00d4aa" : "#ff4d4d";

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
                        ("fps", "error", "steer", "lane_found", "enabled")})


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
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)