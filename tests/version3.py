#!/usr/bin/env python3
"""
version3.py — State Estimation + Lane Follower + Lidar Trigger + Dashboard
Combines:  actual lane following (no offsets)
           state estimation (dead reckoning: x, y, yaw, speed)
           Flask dashboard with telemetry live updates
           Lidar safety trigger (OVERTAKE state if < 800mm)
"""

import cv2
import numpy as np
import threading
import time
import serial
import math
import sys
import os
import types
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Response, render_template_string, request, jsonify

# Add parent directory to path so we can import local jetracer package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jetracer import JetRacer

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
SPEED_SCALE     = 0.00748

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
    # ROI
    "roi_top_frac": 0.5,
    "roi_side_limit": 0.0,
    # Drive
    "speed": 0.15,
    "enabled": False,
    # Lidar safety
    "stop_distance": 800.0,   # mm — trigger if object closer than this
    # State Estimation (Dead Reckoning)
    "x": 0.0,
    "y": 0.0,
    "yaw": 0.0,      # Radians
    "dr_speed": 0.0, # m/s
    # Telemetry (read-only from browser)
    "error": 0.0, "steer": 0.0, "fps": 0,
    "lane_found": False,
    "lane_width": 0.0,
    "lidar_closest": 0.0,     # closest front distance in mm
    "lidar_blocked": False,   # True when obstacle within stop_distance
    "autonomy_state": "FOLLOW",
}

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

# ── GPU helpers ───────────────────────────────────────────────────────────────
def _make_gpu_grayscale():
    probe = cv2.cuda_GpuMat()
    probe.upload(np.zeros((1, 1, 3), dtype=np.uint8))
    try:
        result = cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        if result is not None and not result.empty():
            def _gray_func(bgr_cpu):
                _gpu_frame.upload(bgr_cpu)
                g = cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2GRAY)
                return g.download()
            return _gray_func
    except Exception:
        pass
    try:
        cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2GRAY, _gpu_gray)
        def _gray_inplace(bgr_cpu):
            _gpu_frame.upload(bgr_cpu)
            cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2GRAY, _gpu_gray)
            return _gpu_gray.download()
        return _gray_inplace
    except Exception:
        pass
    return None

gpu_grayscale = _make_gpu_grayscale() if USE_CUDA else None

def cpu_grayscale(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def to_gray(bgr):
    if gpu_grayscale is not None:
        return gpu_grayscale(bgr)
    return cpu_grayscale(bgr)

# ── Frame processing pipeline ────────────────────────────────────────────────
def process_frame(frame, s, annotate: bool):
    h, w = frame.shape[:2]
    roi_top = int(h * s.get("roi_top_frac", 0.5))
    x_start = int(w * s.get("roi_side_limit", 0.0))
    x_end   = w - x_start

    roi_bgr = frame[roi_top:h, x_start:x_end]
    gray = to_gray(roi_bgr)

    bk = s["blur_ksize"] | 1
    blurred = cv2.GaussianBlur(gray, (bk, bk), 0)

    edges = cv2.Canny(blurred, s["canny_lo"], s["canny_hi"])
    _, binary = cv2.threshold(blurred, s["binary_thresh"], 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(edges, binary)

    mk = s["morph_ksize"] | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    cleaned  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=s["morph_iters"])
    roi_mask = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel, iterations=1)

    ys, xs = np.where(roi_mask > 0)
    lane_found = False
    lane_width = 0.0

    if len(xs) > 50:
        lane_found = True
        roi_width = x_end - x_start
        mid_point = roi_width / 2.0
        
        left_pixels = xs[xs < mid_point]
        right_pixels = xs[xs >= mid_point]
        
        left_found = len(left_pixels) > 10
        right_found = len(right_pixels) > 10
        
        if left_found and right_found:
            left_x = np.mean(left_pixels) + x_start
            right_x = np.mean(right_pixels) + x_start
            target_x = (left_x + right_x) / 2.0
            lane_width = right_x - left_x
        elif len(left_pixels) > 10:
            left_x = np.mean(left_pixels) + x_start
            target_x = left_x + 140
        elif len(right_pixels) > 10:
            right_x = np.mean(right_pixels) + x_start
            target_x = right_x - 140
        else:
            target_x = w / 2.0

        dx_pixels = target_x - w / 2.0
    else:
        dx_pixels = 0.0
        target_x = w / 2.0

    if annotate:
        annotated = frame.copy()
        cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)
        cv2.line(annotated, (x_start, roi_top), (x_start, h), (255, 0, 255), 1)
        cv2.line(annotated, (x_end, roi_top), (x_end, h), (255, 0, 255), 1)

        mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        mask_3ch[:, :, 0] = 0
        annotated[roi_top:h, x_start:x_end] = cv2.addWeighted(
            annotated[roi_top:h, x_start:x_end], 0.7, mask_3ch, 0.3, 0)

        if lane_found:
            cv2.circle(annotated, (int(target_x), roi_top + 10), 8, (0, 255, 0), -1)
    else:
        annotated = frame

    return annotated, dx_pixels, lane_found, lane_width

def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret: return
    with frame_lock:
        global latest_frame
        latest_frame = jpeg.tobytes()

# ── Lidar background thread ───────────────────────────────────────────────────
_lidar_cache      = {"closest": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()

def lidar_loop(car: JetRacer):
    print("[lidar] Background safety thread started")
    while True:
        try:
            with state_lock:
                STOP_DISTANCE = state["stop_distance"]

            scan = car.lidar_scan(samples=150)
            front_distances = [dist for ang, dist in scan.items() if (ang >= 320 or ang <= 40) and dist > 10]
            
            closest_front = min(front_distances) if front_distances else 0.0
            is_blocked = 0.0 < closest_front < STOP_DISTANCE
            
            with _lidar_cache_lock:
                _lidar_cache["closest"] = round(closest_front, 1)
                _lidar_cache["blocked"] = is_blocked
                
        except Exception as e:
            print(f"[lidar] scan error: {e}")
            with _lidar_cache_lock:
                _lidar_cache["closest"] = 0.0
                _lidar_cache["blocked"] = True
        time.sleep(0.10)

# ── State Estimation / Sensor Thread ──────────────────────────────────────────
_sensor_cache = {"x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 0.0}
_sensor_cache_lock = threading.Lock()

def parse_telemetry_packet(ser, head1=0xAA, head2=0x55):
    b = ser.read(1)
    if not b or b[0] != head1: return None
    b = ser.read(1)
    if not b or b[0] != head2: return None
    b = ser.read(1)
    if not b: return None
    frame_size = b[0]
    if frame_size < 5 or frame_size > 50: return None
    remaining = frame_size - 3
    rest = ser.read(remaining)
    if len(rest) != remaining: return None

    frame = bytes([head1, head2, frame_size]) + rest
    calc_sum = sum(frame[:-1]) & 0xFF
    if calc_sum != frame[-1]: return None

    gz_raw = int.from_bytes(frame[8:10], 'big', signed=True)
    gz_deg = (gz_raw / 32768.0) * 2000.0

    lvel = int.from_bytes(frame[34:36], 'big', signed=True)
    rvel = int.from_bytes(frame[36:38], 'big', signed=True)

    return gz_deg, lvel, rvel

def calibrate_gyro(ser, duration=2.0):
    print(f"\n[init] Keep rover completely still. Calibrating gyroscope bias for {duration}s...")
    start_time = time.time()
    samples = []
    ser.reset_input_buffer()

    while time.time() - start_time < duration:
        parsed = parse_telemetry_packet(ser)
        if parsed is not None:
            gz_deg, _, _ = parsed
            samples.append(gz_deg)
        time.sleep(0.01)

    bias = sum(samples) / len(samples) if samples else 0.0
    print(f"[init] Calibration successful over {len(samples)} samples. Bias: {bias:.4f} deg/s")
    return bias

def sensor_loop(ser, gz_bias):
    print("[sensors] Background dead reckoning thread started")
    last_time = time.time()
    
    curr_x = 0.0
    curr_y = 0.0
    curr_yaw = 0.0
    curr_speed = 0.0
    
    while True:
        try:
            parsed = parse_telemetry_packet(ser)
            if parsed is None:
                continue

            now = time.time()
            dt = now - last_time
            last_time = now

            if dt <= 0:
                continue

            gz_deg, lvel, rvel = parsed
            
            # Gyro Heading (Yaw)
            gz_calibrated = gz_deg - gz_bias
            omega = math.radians(gz_calibrated)
            curr_yaw += omega * dt
            curr_yaw = (curr_yaw + math.pi) % (2.0 * math.pi) - math.pi

            # Encoder Speed
            avg_vel = (lvel + rvel) / 2.0
            curr_speed = avg_vel * SPEED_SCALE

            # Dead Reckoning Position
            curr_x += curr_speed * math.cos(curr_yaw) * dt
            curr_y += curr_speed * math.sin(curr_yaw) * dt

            with _sensor_cache_lock:
                _sensor_cache["x"] = round(curr_x, 3)
                _sensor_cache["y"] = round(curr_y, 3)
                _sensor_cache["yaw"] = curr_yaw
                _sensor_cache["speed"] = round(curr_speed, 3)

        except Exception as e:
            time.sleep(0.01)

# ── Control Thread ────────────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    global _last_steer
    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    frame_idx = 0
    print("[loop] Control loop started (Pure Pursuit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with state_lock:
            s_copy = dict(state)

        with clients_lock:
            has_clients = stream_clients > 0

        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        annotated, dx_pixels, lane_found, lane_width = process_frame(frame, s_copy, do_annotate)

        if do_annotate:
            _encode_pool.submit(_do_encode, annotated)

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        with _lidar_cache_lock:
            lidar_closest = _lidar_cache["closest"]
            lidar_blocked = _lidar_cache["blocked"]

        with _sensor_cache_lock:
            curr_x = _sensor_cache["x"]
            curr_y = _sensor_cache["y"]
            curr_yaw = _sensor_cache["yaw"]
            curr_speed = _sensor_cache["speed"]

        autonomy_state = s_copy.get("autonomy_state", "FOLLOW")

        # PURE PURSUIT PARAMETERS
        Ld = 0.25 + 0.8 * curr_speed
        Ld = max(0.4, Ld) # safety min
        L = 0.16
        max_steer_rad = 0.52

        # STATE MACHINE Logic
        if s_copy["enabled"]:
            if lidar_blocked:
                if autonomy_state == "FOLLOW":
                    # Trigger Overtake state
                    autonomy_state = "OVERTAKING"
                    with state_lock:
                        state["overtake_x"] = curr_x - 0.28 * math.sin(curr_yaw)
                        state["overtake_y"] = curr_y + 0.28 * math.cos(curr_yaw)
                        state["overtake_yaw"] = curr_yaw
                
                # In OVERTAKING state, track the parallel lane
                with state_lock:
                    ox = state.get("overtake_x", curr_x)
                    oy = state.get("overtake_y", curr_y)
                    oyaw = state.get("overtake_yaw", curr_yaw)
                
                dx_track = curr_x - ox
                dy_track = curr_y - oy
                closest_s = dx_track * math.cos(oyaw) + dy_track * math.sin(oyaw)
                
                tx = ox + (closest_s + Ld) * math.cos(oyaw)
                ty = oy + (closest_s + Ld) * math.sin(oyaw)
                
                global_dx = tx - curr_x
                global_dy = ty - curr_y
                local_x = global_dx * math.cos(-curr_yaw) - global_dy * math.sin(-curr_yaw)
                local_y = global_dx * math.sin(-curr_yaw) + global_dy * math.cos(-curr_yaw)
                
                gamma = 2.0 * local_y / (Ld ** 2)
                delta = math.atan(L * gamma)
                steer = -delta / max_steer_rad
                
            else:
                if autonomy_state == "OVERTAKING":
                    # Obstacle cleared, resume following
                    autonomy_state = "FOLLOW"
                
                if autonomy_state == "FOLLOW":
                    if lane_found:
                        # Convert pixel error to lateral offset
                        local_y = (dx_pixels / 160.0) * 0.5
                        gamma = 2.0 * local_y / (Ld ** 2)
                        delta = math.atan(L * gamma)
                        steer = -delta / max_steer_rad
                    else:
                        steer = _last_steer
                        
            steer = max(-1.0, min(1.0, steer))
            _last_steer = steer
            car.steer(steer)
            car.forward(s_copy["speed"])
        else:
            car.stop()
            steer = 0.0
            autonomy_state = "FOLLOW"

        with state_lock:
            state["error"] = round(dx_pixels, 3)
            state["steer"] = round(steer, 3)
            state["lane_found"] = lane_found
            state["lane_width"] = round(lane_width, 1)
            state["lidar_closest"] = lidar_closest
            state["lidar_blocked"] = lidar_blocked
            state["autonomy_state"] = autonomy_state
            state["x"] = curr_x
            state["y"] = curr_y
            state["yaw"] = curr_yaw
            state["dr_speed"] = curr_speed

        frame_idx += 1

    cap.release()

# ── Flask / dashboard ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetRacer Dashboard — State Estimation & Lane Following</title>
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
<h1>&#9675; JetRacer &#183; Dashboard v3</h1>
<div class="grid">
  <div class="card">
    <h2>Camera feed (annotated)</h2>
    <img id="feed" src="/video_feed" alt="camera">
    <div class="status-bar" style="margin-top:.75rem">
      <div class="stat"><span class="stat-val" id="v-state">FOLLOW</span>
                        <span class="stat-lbl">state</span></div>
      <div class="stat"><span class="stat-val" id="v-x">0.00</span>
                        <span class="stat-lbl">X (m)</span></div>
      <div class="stat"><span class="stat-val" id="v-y">0.00</span>
                        <span class="stat-lbl">Y (m)</span></div>
      <div class="stat"><span class="stat-val" id="v-yaw">0.0</span>
                        <span class="stat-lbl">Yaw (&deg;)</span></div>
      <div class="stat"><span class="stat-val" id="v-dr-spd">0.00</span>
                        <span class="stat-lbl">Speed (m/s)</span></div>
      <div class="stat"><span class="stat-val" id="v-fps">0</span>
                        <span class="stat-lbl">fps</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span>
                        <span class="stat-lbl">error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span>
                        <span class="stat-lbl">steer</span></div>
      <div class="stat"><span class="stat-val" id="v-lane">&mdash;</span>
                        <span class="stat-lbl">lane</span></div>
      <div class="stat"><span class="stat-val" id="v-lanew">0.0</span>
                        <span class="stat-lbl">lane w</span></div>
      <div class="stat"><span class="stat-val" id="v-lidar">0</span>
                        <span class="stat-lbl">lidar front</span></div>
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
    <h2>Lane Tracking & Safety</h2>
    <div class="slider-row">
      <label>ROI Top Frac</label>
      <input type="range" id="roi_top_frac" min="0.1" max="0.9" value="0.5" step="0.05">
      <span class="val" id="v-roi_top_frac">0.5</span>
    </div>
    <div class="slider-row">
      <label>Stop Dist mm</label>
      <input type="range" id="stop_distance" min="100" max="2000" value="800" step="10">
      <span class="val" id="v-stop_distance">800</span>
    </div>
  </div>
</div>
<script>
const sliders = [
  "speed",
  "canny_lo","canny_hi","binary_thresh","blur_ksize",
  "morph_ksize","morph_iters",
  "roi_top_frac","stop_distance"
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
    } else if (id === "stop_distance") {
      disp.textContent = v;
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
    document.getElementById("v-state").textContent = d.autonomy_state;
    document.getElementById("v-fps").textContent  = d.fps;
    document.getElementById("v-err").textContent  = d.error.toFixed(2);
    document.getElementById("v-str").textContent  = d.steer.toFixed(2);
    
    // Convert yaw from radians to degrees for display
    const yaw_deg = (d.yaw * 180 / Math.PI).toFixed(1);
    
    document.getElementById("v-x").textContent = d.x.toFixed(3);
    document.getElementById("v-y").textContent = d.y.toFixed(3);
    document.getElementById("v-yaw").textContent = yaw_deg;
    document.getElementById("v-dr-spd").textContent = d.dr_speed.toFixed(3);

    const laneEl = document.getElementById("v-lane");
    laneEl.textContent = d.lane_found ? "✓" : "✗";
    laneEl.style.color = d.lane_found ? "#00d4aa" : "#ff4d4d";

    document.getElementById("v-lanew").textContent = (d.lane_width || 0).toFixed(1);

    const lidarEl = document.getElementById("v-lidar");
    lidarEl.textContent = d.lidar_closest.toFixed(0);

    const pct = (d.error / 160.0 + 1) / 2 * 100;
    const bar = document.getElementById("error-bar");
    bar.style.left = pct + "%";
    bar.style.background = Math.abs(d.error) > 80 ? "#ff4d4d" : "#00d4aa";
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
                        ("fps", "error", "steer", "lane_found", "lane_width", "enabled",
                         "lidar_closest", "lidar_blocked", "autonomy_state",
                         "x", "y", "yaw", "dr_speed")})

@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k in state:
                state[k] = v
    return jsonify({"ok": True})

# --- Silent overrides for JetRacer driver to keep output clean ---
def silent_steer(self, val):
    val = max(-1.0, min(1.0, val))
    if val >= 0:
        us = self.STEER_CENTER + val * (self.STEER_RIGHT - self.STEER_CENTER)
    else:
        us = self.STEER_CENTER + val * (self.STEER_CENTER - self.STEER_LEFT)
    self._set_us(self.STEER_CH, us)

def silent_throttle(self, val):
    val = max(-1.0, min(1.0, val))
    if val >= 0:
        us = self.THROTTLE_NEUTRAL + val * (self.THROTTLE_FWD_MAX - self.THROTTLE_NEUTRAL)
    else:
        us = self.THROTTLE_NEUTRAL + val * (self.THROTTLE_NEUTRAL - self.THROTTLE_REV_MAX)
    self._set_us(self.THROTTLE_CH, us)

def silent_stop(self):
    self._set_us(self.THROTTLE_CH, self.THROTTLE_NEUTRAL)
    self._set_us(self.STEER_CH, self.STEER_CENTER)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[init] Initializing JetRacer motor and Lidar controllers...")
    try:
        car = JetRacer(init_lidar=True)
        car.arm(delay=2)
        
        # Monkey-patch driver to prevent console logging on actuator updates
        car.steer = types.MethodType(silent_steer, car)
        car.throttle = types.MethodType(silent_throttle, car)
        car.stop = types.MethodType(silent_stop, car)
    except Exception as e:
        print(f"\n[error] Failed to initialize JetRacer: {e}")
        sys.exit(1)

    serial_port = '/dev/ttyACM0'
    baud_rate = 115200
    print(f"[init] Connecting to serial port: {serial_port} at {baud_rate} baud...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1.0)
    except Exception as e:
        print(f"\n[error] Failed to open serial port {serial_port}: {e}")
        car.stop()
        sys.exit(1)

    # Start the Lidar background thread FIRST so it doesn't buffer overflow during gyro calibration
    lt = threading.Thread(target=lidar_loop, args=(car,), daemon=True)
    lt.start()

    # Perform gyro calibration (blocks for 2.0s)
    gz_bias = calibrate_gyro(ser, duration=2.0)

    st = threading.Thread(target=sensor_loop, args=(ser, gz_bias), daemon=True)
    st.start()

    ct = threading.Thread(target=control_loop, args=(car,), daemon=True)
    ct.start()

    print("\n[flask] Dashboard ready at → http://0.0.0.0:5000")
    print("[init] Automated tests removed. Control via web dashboard.")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
