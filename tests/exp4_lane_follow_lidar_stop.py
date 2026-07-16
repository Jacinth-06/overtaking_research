#!/usr/bin/env python3
"""
exp5_lane_follow_lidar_stop.py — Lane following + LiDAR safety stop
====================================================================
Car uses camera + vision pipeline to follow lanes via PID.
If the front LiDAR cone detects an obstacle closer than stop_distance,
the car stops (overriding PID output). Resumes lane following when obstacle clears.

Run:   python exp5_lane_follow_lidar_stop.py
Open:  http://<jetson-ip>:5000
"""

import cv2
import numpy as np
import threading
import time
import serial
import queue
import csv
import io
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Response, render_template_string, request, jsonify

from jetracer import JetRacer

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

app = Flask(__name__)

# ── CUDA availability check ──────────────────────────────────────────────────
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

# ── Config constants ─────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240
ENCODE_EVERY    = 3
JPEG_QUALITY    = 30
MJPEG_INTERVAL  = 1 / 15

# ── Shared state ─────────────────────────────────────────────────────────────
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
    # PID
    "kp": 0.55,  "ki": 0.003,  "kd": 0.25,
    # Drive
    "speed": 0.15,
    "enabled": False,
    # Lidar safety
    "stop_distance": 400.0,
    # Telemetry (read-only from browser)
    "error": 0.0, "steer": 0.0, "fps": 0,
    "lane_found": False,
    "lidar_closest": 0.0,
    "lidar_blocked": False,
    # Encoder data
    "enc_speed": 0.0, "enc_dist": 0.0,
    # IMU data
    "imu_ax": 0, "imu_ay": 0, "imu_az": 0,
    "imu_gx": 0, "imu_gy": 0, "imu_gz": 0,
    # Telemetry
    "is_testing": False,
    "test_id": "",
    "reset_encoder_dist": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()

data_log = []
data_lock = threading.Lock()

def telemetry_loop():
    print("[telemetry] Local data logging loop started (20Hz)")
    while True:
        with state_lock:
            s_copy = dict(state)
        with _encoder_cache_lock:
            enc_copy = dict(_encoder_cache)
        with _lidar_cache_lock:
            lidar_copy = dict(_lidar_cache)

        if s_copy.get("is_testing"):
            dp = {
                "timestamp": round(time.time(), 3),
                "error": s_copy.get("error", 0.0),
                "steer": s_copy.get("steer", 0.0),
                "enc_speed": enc_copy.get("speed", 0.0),
                "lidar_dist": lidar_copy.get("closest", 0.0),
            }
            with data_lock:
                data_log.append(dp)
        
        time.sleep(0.05)

_last_steer = 0.0

frame_lock   = threading.Lock()
latest_frame = None
stream_clients = 0
clients_lock   = threading.Lock()
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


# ── GPU helpers ──────────────────────────────────────────────────────────────
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

def to_gray(bgr):
    if gpu_grayscale is not None:
        return gpu_grayscale(bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ── Frame processing pipeline ────────────────────────────────────────────────
def process_frame(frame, s, annotate: bool):
    global _last_steer
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
    left_found = False
    right_found = False
    lane_width = 0.0
    left_x = 0.0
    right_x = 0.0

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
            target_x = left_x - 140
        elif len(right_pixels) > 10:
            right_x = np.mean(right_pixels) + x_start
            target_x = right_x + 140
        else:
            target_x = w / 2.0

        error = (target_x - w / 2.0) / (w / 2.0) * 3.5

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
        steer = max(-1, min(1, steer))
        _last_steer = steer
    else:
        error = 0.0
        steer = _last_steer

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

    return annotated, error, steer, left_found, right_found, lane_width, left_x, right_x


def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return
    with frame_lock:
        global latest_frame
        latest_frame = jpeg.tobytes()





# ── Lidar background thread ──────────────────────────────────────────────────
_lidar_cache = {"closest": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()

def lidar_loop(car: JetRacer):
    print("[lidar] Background safety thread started")
    while True:
        try:
            with state_lock:
                STOP_DISTANCE = state["stop_distance"]
            scan = car.lidar_scan(samples=150)
            front_distances = [dist for ang, dist in scan.items()
                               if (ang >= 320 or ang <= 40) and dist > 10]
            closest_front = min(front_distances) if front_distances else 0.0
            is_blocked = closest_front > 0 and closest_front < STOP_DISTANCE
            with _lidar_cache_lock:
                _lidar_cache["closest"] = round(closest_front, 1)
                _lidar_cache["blocked"] = is_blocked
        except Exception as e:
            print(f"[lidar] scan error: {e}")
            with _lidar_cache_lock:
                _lidar_cache["closest"] = 0.0
                _lidar_cache["blocked"] = True
        time.sleep(0.05)


# ── Sensor loop ──────────────────────────────────────────────────────────────
_imu_cache = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}
_imu_cache_lock = threading.Lock()
_encoder_cache = {"speed": 0.0, "distance": 0.0}
_encoder_cache_lock = threading.Lock()

def sensor_loop():
    print("[sensors] Background thread started")
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    except Exception as e:
        print(f"[sensors] Failed to open serial: {e}")
        return

    HEAD1 = 0xAA
    HEAD2 = 0x55
    total_distance = 0.0
    last_time = time.time()
    SPEED_SCALE = 0.00748

    while True:
        with state_lock:
            if state.get("reset_encoder_dist"):
                total_distance = 0.0
                state["reset_encoder_dist"] = False
        try:
            b = ser.read(1)
            if not b or b[0] != HEAD1: continue
            b = ser.read(1)
            if not b or b[0] != HEAD2: continue
            b = ser.read(1)
            if not b: continue
            frame_size = b[0]
            if frame_size < 5 or frame_size > 50: continue
            remaining = frame_size - 3
            rest = ser.read(remaining)
            if len(rest) != remaining: continue
            frame = bytes([HEAD1, HEAD2, frame_size]) + rest
            calc_sum = sum(frame[:-1]) & 0xFF
            recv_sum = frame[-1]
            if calc_sum != recv_sum: continue

            gx = int.from_bytes(frame[4:6],   'big', signed=True) / 32768 * 2000
            gy = int.from_bytes(frame[6:8],   'big', signed=True) / 32768 * 2000
            gz = int.from_bytes(frame[8:10],  'big', signed=True) / 32768 * 2000
            ax = int.from_bytes(frame[10:12], 'big', signed=True) / 32768 * 2 * 9.8
            ay = int.from_bytes(frame[12:14], 'big', signed=True) / 32768 * 2 * 9.8
            az = int.from_bytes(frame[14:16], 'big', signed=True) / 32768 * 2 * 9.8

            with _imu_cache_lock:
                _imu_cache["ax"] = round(ax, 2); _imu_cache["ay"] = round(ay, 2)
                _imu_cache["az"] = round(az, 2); _imu_cache["gx"] = round(gx, 1)
                _imu_cache["gy"] = round(gy, 1); _imu_cache["gz"] = round(gz, 1)

            lvel = int.from_bytes(frame[34:36], 'big', signed=True)
            rvel = int.from_bytes(frame[36:38], 'big', signed=True)
            now = time.time()
            dt = now - last_time
            if dt > 0:
                avg_vel = (lvel + rvel) / 2.0
                speed_ms = avg_vel * SPEED_SCALE
                total_distance += speed_ms * dt
                with _encoder_cache_lock:
                    _encoder_cache["speed"] = round(speed_ms, 3)
                    _encoder_cache["distance"] = round(total_distance, 3)
            last_time = now
        except Exception as e:
            print(f"[sensors] error: {e}")
            time.sleep(0.1)


# ── Control loop ──────────────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    frame_idx = 0

    print("[loop] Lane follow + lidar stop control loop started")

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
        annotated, error, steer, left_found, right_found, lane_width, left_x, right_x = process_frame(frame, s_copy, do_annotate)
        lane_found = left_found or right_found

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

        with _imu_cache_lock:
            imu_data = dict(_imu_cache)
        with _encoder_cache_lock:
            enc_speed = _encoder_cache["speed"]
            enc_dist = _encoder_cache["distance"]

        # Lane following + lidar safety override
        if s_copy["enabled"]:
            if lidar_blocked:
                # Obstacle — stop completely
                car.stop()
            else:
                # Normal lane following
                car.steer(steer)
                car.forward(s_copy["speed"])
        else:
            car.stop()

        with state_lock:
            state["error"]         = round(error, 3)
            state["steer"]         = round(steer, 3)
            state["lane_found"]    = lane_found
            state["lidar_closest"] = lidar_closest
            state["lidar_blocked"] = lidar_blocked
            state["enc_speed"]     = enc_speed
            state["enc_dist"]      = enc_dist
            state["imu_ax"] = imu_data["ax"]; state["imu_ay"] = imu_data["ay"]
            state["imu_az"] = imu_data["az"]; state["imu_gx"] = imu_data["gx"]
            state["imu_gy"] = imu_data["gy"]; state["imu_gz"] = imu_data["gz"]



        frame_idx += 1

    cap.release()


# ── Flask / dashboard ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 4 — Lane Follow + LiDAR Stop</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg: #0b0f19; --surface: #111827; --surface-border: #1f2937;
    --accent: #3b82f6; --accent-hover: #2563eb; 
    --success: #10b981; --danger: #ef4444; --warn: #f59e0b;
    --text-main: #f3f4f6; --text-muted: #9ca3af;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text-main); font-family: var(--font); padding: 2rem; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; border-bottom: 1px solid var(--surface-border); padding-bottom: 1rem; }
  .header h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: 0.05em; }
  .badge { background: var(--surface-border); padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-family: monospace; color: var(--text-muted); }
  
  .dashboard { display: grid; grid-template-columns: 1fr 380px; gap: 2rem; }
  .panel { background: var(--surface); border: 1px solid var(--surface-border); border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
  .panel-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; font-weight: 600; }
  
  .slider-group { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.2rem; }
  .slider-group label { font-size: 0.85rem; color: var(--text-muted); display: flex; justify-content: space-between; }
  .slider-group input[type=range] { width: 100%; accent-color: var(--accent); }
  
  .btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1.5rem; }
  button { padding: 0.75rem 1rem; border: none; border-radius: 8px; font-family: var(--font); font-weight: 600; font-size: 0.9rem; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { background: var(--accent-hover); }
  .btn-danger { background: var(--surface-border); color: var(--danger); }
  .btn-danger:hover { background: var(--danger); color: white; }
  
  .test-controls { margin-top: 1rem; border-top: 1px solid var(--surface-border); padding-top: 1.5rem; }
  .btn-test { background: var(--success); color: white; width: 100%; margin-bottom: 0.5rem; }
  .btn-test.active { background: var(--warn); color: #000; }
  .btn-download { background: var(--surface-border); color: var(--text-main); width: 100%; text-decoration: none; display: inline-block; text-align: center; font-weight: 600; font-size: 0.9rem; padding: 0.75rem 1rem; border-radius: 8px; }
  .btn-download:hover { background: #374151; }
  
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;}
  .chart-container { position: relative; height: 260px; width: 100%; background: #182235; border-radius: 8px; padding: 1rem; border: 1px solid var(--surface-border); }

  .status-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .stat-box { background: #182235; padding: 1rem; border-radius: 8px; border: 1px solid var(--surface-border); }
  .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem; }
  .stat-value { font-size: 1.25rem; font-weight: 600; font-family: monospace; color: var(--accent); }
  .stat-value.danger { color: var(--danger); }

  img#feed { width: 100%; border-radius: 6px; display: block; background: #000; min-height: 180px; margin-bottom: 1.5rem;}
  
  .error-track { position: relative; height: 12px; background: var(--surface-border); border-radius: 6px; margin-top: .5rem; overflow: hidden; }
  #error-bar { position: absolute; height: 100%; width: 6px; background: var(--accent); left: 50%; transform: translateX(-50%); transition: left .1s; border-radius: 3px; }

  @media (max-width: 1200px) { .charts-grid { grid-template-columns: 1fr; } .status-grid { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 900px) { .dashboard { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>Exp 4: Vision Lane Follow + LiDAR Stop</h1>
  <div class="badge">Telemetry Mode: <span id="mode-badge">IDLE</span></div>
</div>

<div class="dashboard">
  <!-- Telemetry and Feed Panel -->
  <div>
    <div class="panel">
      <div class="panel-title">Vision Pipeline & Feedback</div>
      <img id="feed" src="/video_feed" alt="camera">
      <div class="error-track" title="Lane error (centre = 0)"><div id="error-bar"></div></div>
    </div>
    
    <div class="status-grid" style="margin-top: 1.5rem;">
      <div class="stat-box">
        <div class="stat-label">Lane Found</div>
        <div class="stat-value" id="disp-lane">NO</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">LiDAR Blocked</div>
        <div class="stat-value" id="disp-blocked">NO</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">LiDAR Dist</div>
        <div class="stat-value" id="disp-lidar">0 mm</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Encoder Spd</div>
        <div class="stat-value" id="disp-spd">0.00 m/s</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-container"><canvas id="chartError"></canvas></div>
      <div class="chart-container"><canvas id="chartLidar"></canvas></div>
    </div>
  </div>

  <!-- Control Panel -->
  <div class="panel">
    <div class="panel-title">Drive & Parameters</div>
    <div class="btn-group">
      <button class="btn-primary" onclick="setEnabled(true)">Enable Drive</button>
      <button class="btn-danger" onclick="setEnabled(false)">Stop / Disable</button>
    </div>

    <div class="slider-group">
      <label>Speed <span id="v-speed">0.15</span></label>
      <input type="range" id="speed" min="0" max="60" value="15" step="1">
    </div>
    <div class="slider-group">
      <label>Stop Dist (mm) <span id="v-stop_distance">400</span></label>
      <input type="range" id="stop_distance" min="100" max="2000" value="400" step="10">
    </div>

    <hr style="border:0; border-top:1px solid var(--surface-border); margin: 1.5rem 0;">

    <div class="slider-group">
      <label>Kp <span id="v-kp">0.55</span></label>
      <input type="range" id="kp" min="0" max="2" value="0.55" step="0.01">
    </div>
    <div class="slider-group">
      <label>Ki <span id="v-ki">0.003</span></label>
      <input type="range" id="ki" min="0" max="0.05" value="0.003" step="0.001">
    </div>
    <div class="slider-group">
      <label>Kd <span id="v-kd">0.25</span></label>
      <input type="range" id="kd" min="0" max="0.5" value="0.25" step="0.01">
    </div>

    <div class="test-controls">
      <div class="panel-title">Data Collection</div>
      <button id="btn-test" class="btn-test" onclick="toggleTest()">START RECORDING</button>
      <a href="/download_csv" class="btn-download button" target="_blank" style="box-sizing: border-box;">Download CSV</a>
    </div>

    <hr style="border:0; border-top:1px solid var(--surface-border); margin: 1.5rem 0;">
    
    <div class="panel-title">Vision Thresholds</div>
    <div class="slider-group">
      <label>Canny lo <span id="v-canny_lo">50</span></label>
      <input type="range" id="canny_lo" min="0" max="255" value="50" step="1">
    </div>
    <div class="slider-group">
      <label>Canny hi <span id="v-canny_hi">150</span></label>
      <input type="range" id="canny_hi" min="0" max="255" value="150" step="1">
    </div>
    <div class="slider-group">
      <label>Binary thr <span id="v-binary_thresh">200</span></label>
      <input type="range" id="binary_thresh" min="0" max="255" value="200" step="1">
    </div>
  </div>
</div>

<script>
const sliders = ["speed","stop_distance","kp","ki","kd","canny_lo","canny_hi","binary_thresh"];
sliders.forEach(id => {
  const el = document.getElementById(id);
  const disp = document.getElementById("v-"+id);
  if (!el) return;
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (id === "speed") { disp.textContent = (v/100).toFixed(2); sendParam(id, v/100); }
    else if (id === "stop_distance") { disp.textContent = v; sendParam(id, v); }
    else { disp.textContent = Number.isInteger(v) ? v : v.toFixed(3); sendParam(id, v); }
  });
});

function sendParam(key, value) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({[key]: value})}); }
function setEnabled(v) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled: v})}); }

// Chart.js Setup
Chart.defaults.color = '#9ca3af';
Chart.defaults.font.family = "'Inter', sans-serif";
const commonOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
  scales: { x: { display: false }, y: { grid: { color: '#1f2937' } } },
  plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
  elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.1 } }
};

const chartError = new Chart(document.getElementById('chartError'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Vision Error (Normalized)', borderColor: '#ef4444', data: [] },
    { label: 'Steer Angle', borderColor: '#3b82f6', data: [] }
  ]},
  options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: -1.2, max: 1.2 } } }
});

const chartLidar = new Chart(document.getElementById('chartLidar'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'LiDAR Front Dist (mm)', borderColor: '#f59e0b', data: [] }
  ]},
  options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 1500 } } }
});

let isTesting = false;
let dataPoints = 0;

function toggleTest() {
  isTesting = !isTesting;
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({is_testing: isTesting})});
  if (isTesting) {
    chartError.data.labels = []; chartError.data.datasets[0].data = []; chartError.data.datasets[1].data = [];
    chartLidar.data.labels = []; chartLidar.data.datasets[0].data = [];
    dataPoints = 0;
  }
}

async function poll() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    
    document.getElementById("disp-spd").textContent = d.enc_speed.toFixed(3) + " m/s";
    document.getElementById("disp-lidar").textContent = d.lidar_closest.toFixed(0) + " mm";
    
    const laneEl = document.getElementById("disp-lane");
    if(d.lane_found) {
      laneEl.textContent = "YES"; laneEl.className = "stat-value"; laneEl.style.color = "#10b981";
    } else {
      laneEl.textContent = "NO"; laneEl.className = "stat-value danger"; laneEl.style.color = "#ef4444";
    }

    const bEl = document.getElementById("disp-blocked");
    if(d.lidar_blocked) {
      bEl.textContent = "YES"; bEl.className = "stat-value danger"; bEl.style.color = "#ef4444";
    } else {
      bEl.textContent = "NO"; bEl.className = "stat-value"; bEl.style.color = "#10b981";
    }

    const pct = (d.error + 1) / 2 * 100;
    const bar = document.getElementById("error-bar");
    bar.style.left = pct + "%";
    bar.style.background = Math.abs(d.error) > 0.5 ? "#ef4444" : "#10b981";
    
    isTesting = d.is_testing;
    const btn = document.getElementById("btn-test");
    const badge = document.getElementById("mode-badge");
    if(isTesting) { 
      btn.innerHTML = "STOP RECORDING"; btn.classList.add("active"); 
      badge.textContent = "RECORDING"; badge.style.color = "#f59e0b";
      
      dataPoints++;
      chartError.data.labels.push(dataPoints);
      chartError.data.datasets[0].data.push(d.error);
      chartError.data.datasets[1].data.push(d.steer);
      chartError.update();

      chartLidar.data.labels.push(dataPoints);
      chartLidar.data.datasets[0].data.push(d.lidar_closest);
      chartLidar.update();
    } else { 
      btn.innerHTML = "START RECORDING"; btn.classList.remove("active"); 
      badge.textContent = "IDLE"; badge.style.color = "#9ca3af";
    }
  } catch(e) {}
  setTimeout(poll, 100);
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
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with state_lock:
        return jsonify({k: state[k] for k in
                        ("fps", "error", "steer", "lane_found", "enabled",
                         "lidar_closest", "lidar_blocked",
                         "enc_speed", "enc_dist", "is_testing", "test_id")})

@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k == "is_testing":
                if v and not state.get("is_testing", False):
                    state["test_id"] = f"test_{int(time.time())}"
                    state["is_testing"] = True
                    with data_lock:
                        data_log.clear()
                elif not v and state.get("is_testing", False):
                    state["is_testing"] = False
            elif k in state:
                if k == "enabled" and state["enabled"] and not v:
                    state["reset_encoder_dist"] = True
                state[k] = v
                if k in ("kp", "ki", "kd"):
                    pid_state["integral"]   = 0.0
                    pid_state["last_error"] = 0.0
    return jsonify({"ok": True})

@app.route("/download_csv")
def download_csv():
    with data_lock:
        if not data_log:
            return "No data recorded yet. Click 'START RECORDING' first.", 400
        keys = data_log[0].keys()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_log)
    return Response(output.getvalue(), mimetype="text/csv", 
                    headers={"Content-Disposition": "attachment;filename=exp4_telemetry.csv"})



if __name__ == "__main__":
    car = JetRacer(init_lidar=True)
    car.arm(delay=3)

    lt = threading.Thread(target=lidar_loop, args=(car,), daemon=True)
    lt.start()
    tt = threading.Thread(target=telemetry_loop, daemon=True)
    tt.start()
    st = threading.Thread(target=sensor_loop, daemon=True)
    st.start()
    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
