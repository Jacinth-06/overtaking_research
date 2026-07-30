#!/usr/bin/env python3
"""
exp6_integrated.py — Lane Follow + LiDAR Stop + Colour-Signal Speed Control
=============================================================================

Full behaviour (priority order — highest first):

  1. LIDAR_STOP   — Front obstacle within stop_distance → car stops completely.
  2. RED_STOP     — Red colour signal detected → car stops completely.
                    Resumes BASE_SPEED lane-following once red disappears.
  3. BLUE_SLOW    — Blue colour signal detected → car drops to BASE_SPEED
                    (cancels any active BOOST). Another green can re-trigger.
  4. GREEN_BOOST  — Green colour signal detected → car drives at FAST_SPEED.
  5. LANE_FOLLOW  — Default state — PID lane-centering at BASE_SPEED.

Autonomy states reported to dashboard:
  FOLLOW | BOOST | SLOWDOWN | RED_STOP | LIDAR_STOP | DISABLED

Detection pipeline (camera is shared across lane + colour detection):
  frame → lane-follow PID (lower ROI) + colour blob detection (full frame)

Threads:
  • lidar_loop      — LiDAR scan @ ~20 Hz, updates _lidar_cache
  • colour_loop     — Colour blob detection @ 10 Hz, updates _colour_cache
  • sensor_loop     — IMU + encoder via serial @ ~100 Hz
  • control_loop    — Camera read + lane PID + drive commands @ camera FPS
  • firebase_loop   — Batches telemetry to Firebase every 2 s

Flask dashboard served on port 5000:
  GET  /            — Dashboard UI
  GET  /video_feed  — Annotated MJPEG stream
  GET  /status      — JSON status snapshot
  POST /set         — Update any state key at runtime

Run:   python3 tests/Voltaero/exp6_integrated.py
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
import requests
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Response, render_template_string, request, jsonify

from jetracer import JetRacer
from detector import detect_objects, detect_green_objects, detect_blue_objects

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
else:
    print("[init] No CUDA device — falling back to CPU")
    _gpu_frame = _gpu_gray = None

# ── Config constants ──────────────────────────────────────────────────────────
WIDTH, HEIGHT  = 320, 240
ENCODE_EVERY   = 3
JPEG_QUALITY   = 30
MJPEG_INTERVAL = 1 / 15
COLOUR_LOOP_HZ = 10

FIREBASE_URL = "https://jetracer-f1b1c-default-rtdb.asia-southeast1.firebasedatabase.app"

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    # Vision / PID
    "canny_lo": 50,      "canny_hi": 150,
    "binary_thresh": 200,
    "blur_ksize": 5,
    "morph_ksize": 5,    "morph_iters": 2,
    "roi_top_frac": 0.5,
    "roi_side_limit": 0.0,
    "kp": 0.55,  "ki": 0.003,  "kd": 0.25,
    # Speed
    "base_speed": 0.15,   # normal / post-blue speed
    "fast_speed": 0.25,   # green-boost speed
    # Colour trigger thresholds (blob width in px)
    "green_box_width": 250,
    "blue_box_width":  200,
    "red_box_width":   150,
    # LiDAR
    "stop_distance": 400.0,
    # Control
    "enabled": False,
    # Telemetry (read-only)
    "error": 0.0, "steer": 0.0, "fps": 0,
    "lane_found": False,
    "lidar_closest": 0.0, "lidar_closest_left": 0.0, "lidar_blocked": False,
    "colour_green": False, "colour_blue": False, "colour_red": False,
    "autonomy_state": "DISABLED",
    "enc_speed": 0.0, "enc_dist": 0.0,
    "imu_ax": 0, "imu_ay": 0, "imu_az": 0,
    "imu_gx": 0, "imu_gy": 0, "imu_gz": 0,
    # Testing
    "is_testing": False, "test_id": "",
    "reset_encoder_dist": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()

# ── Per-subsystem caches ──────────────────────────────────────────────────────
_lidar_cache      = {"closest": 0.0, "closest_left": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()

_colour_cache      = {"green": False, "blue": False, "red": False}
_colour_cache_lock = threading.Lock()

_imu_cache      = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}
_imu_cache_lock = threading.Lock()

_encoder_cache      = {"speed": 0.0, "distance": 0.0}
_encoder_cache_lock = threading.Lock()

# ── MJPEG stream helpers ──────────────────────────────────────────────────────
frame_lock    = threading.Lock()
latest_frame  = None
stream_clients = 0
clients_lock   = threading.Lock()
_encode_pool   = ThreadPoolExecutor(max_workers=1)

# ── Telemetry queue (→ Firebase) ──────────────────────────────────────────────
telemetry_queue = queue.Queue()

# ── Data log (local CSV) ──────────────────────────────────────────────────────
data_log  = []
data_lock = threading.Lock()

# ── Drive state (colour machine) ─────────────────────────────────────────────
_boosting = False          # True while green triggered and blue not yet seen
_boosting_lock = threading.Lock()

# ── Last steer hold ───────────────────────────────────────────────────────────
_last_steer = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    cap = cv2.VideoCapture(_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    raise RuntimeError("No camera found")


# ─────────────────────────────────────────────────────────────────────────────
# GPU / CPU grayscale
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Lane-following frame processing (PID)
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame, s, annotate: bool):
    global _last_steer
    h, w = frame.shape[:2]
    roi_top = int(h * s.get("roi_top_frac", 0.5))
    x_start = int(w * s.get("roi_side_limit", 0.0))
    x_end   = w - x_start
    roi_bgr = frame[roi_top:h, x_start:x_end]
    gray    = to_gray(roi_bgr)
    bk      = s["blur_ksize"] | 1
    blurred = cv2.GaussianBlur(gray, (bk, bk), 0)
    edges   = cv2.Canny(blurred, s["canny_lo"], s["canny_hi"])
    _, binary = cv2.threshold(blurred, s["binary_thresh"], 255, cv2.THRESH_BINARY)
    combined  = cv2.bitwise_and(edges, binary)
    mk        = s["morph_ksize"] | 1
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    cleaned   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=s["morph_iters"])
    roi_mask  = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel, iterations=1)
    ys, xs    = np.where(roi_mask > 0)

    lane_found = left_found = right_found = False
    lane_width = left_x = right_x = 0.0
    error = 0.0
    target_x = w / 2.0

    if len(xs) > 50:
        lane_found  = True
        roi_width   = x_end - x_start
        mid_point   = roi_width / 2.0
        left_pixels  = xs[xs  < mid_point]
        right_pixels = xs[xs >= mid_point]
        left_found   = len(left_pixels)  > 10
        right_found  = len(right_pixels) > 10

        if left_found and right_found:
            left_x   = np.mean(left_pixels)  + x_start
            right_x  = np.mean(right_pixels) + x_start
            target_x = (left_x + right_x) / 2.0
            lane_width = right_x - left_x
        elif left_found:
            left_x   = np.mean(left_pixels) + x_start
            target_x = left_x - 140
        elif right_found:
            right_x  = np.mean(right_pixels) + x_start
            target_x = right_x + 140

        error = (target_x - w / 2.0) / (w / 2.0) * 3.5
        now   = time.time()
        dt    = max(now - pid_state["last_time"], 0.001)
        pid_state["integral"]  += error * dt
        pid_state["integral"]   = max(-1.0, min(1.0, pid_state["integral"]))
        derivative              = (error - pid_state["last_error"]) / dt
        pid_state["last_error"] = error
        pid_state["last_time"]  = now
        steer = s["kp"] * error + s["ki"] * pid_state["integral"] + s["kd"] * derivative
        steer = max(-1.0, min(1.0, steer))
        _last_steer = steer
    else:
        steer = _last_steer

    if annotate:
        annotated = frame.copy()
        cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)
        cv2.line(annotated, (x_start, roi_top), (x_start, h), (255, 0, 255), 1)
        cv2.line(annotated, (x_end,   roi_top), (x_end, h),   (255, 0, 255), 1)
        mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        mask_3ch[:, :, 0] = 0
        annotated[roi_top:h, x_start:x_end] = cv2.addWeighted(
            annotated[roi_top:h, x_start:x_end], 0.7, mask_3ch, 0.3, 0)
        if lane_found:
            cv2.circle(annotated, (int(target_x), roi_top + 10), 8, (0, 255, 0), -1)
    else:
        annotated = frame

    return annotated, error, steer, left_found, right_found, lane_width, left_x, right_x


# ─────────────────────────────────────────────────────────────────────────────
# JPEG encode (async)
# ─────────────────────────────────────────────────────────────────────────────
def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return
    with frame_lock:
        global latest_frame
        latest_frame = jpeg.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Background threads
# ─────────────────────────────────────────────────────────────────────────────
def lidar_loop(car: JetRacer):
    """Continuously scans LiDAR and updates _lidar_cache."""
    print("[lidar] thread started")
    while True:
        try:
            with state_lock:
                STOP_DISTANCE = state["stop_distance"]
            scan = car.lidar_scan(samples=150)
            front = [d for a, d in scan.items() if (a >= 320 or a <= 40) and d > 10]
            left  = [d for a, d in scan.items() if (250 <= a <= 310) and d > 10]
            closest_front = min(front) if front else 0.0
            closest_left  = min(left)  if left  else 0.0
            blocked = closest_front > 0 and closest_front < STOP_DISTANCE
            with _lidar_cache_lock:
                _lidar_cache["closest"]      = round(closest_front, 1)
                _lidar_cache["closest_left"] = round(closest_left, 1)
                _lidar_cache["blocked"]      = blocked
        except Exception:
            with _lidar_cache_lock:
                _lidar_cache["closest"] = _lidar_cache["closest_left"] = 0.0
                _lidar_cache["blocked"] = True
        time.sleep(0.05)


def colour_loop():
    """
    Dedicated thread that opens its own camera connection and detects
    green / blue / red blobs at COLOUR_LOOP_HZ.
    Uses a separate camera cap so the control loop is not affected.
    Falls back gracefully if the camera cannot be opened a second time.
    """
    print("[colour] thread started")
    cap = None
    try:
        cap = open_camera()
    except Exception as e:
        print(f"[colour] secondary camera open failed ({e}), colour detection disabled")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        with state_lock:
            s = {
                "green_box_width": state["green_box_width"],
                "blue_box_width":  state["blue_box_width"],
                "red_box_width":   state["red_box_width"],
            }

        green_results = detect_green_objects(frame)
        blue_results  = detect_blue_objects(frame)
        red_results   = detect_objects(frame)

        green_seen = any((x2 - x1) > s["green_box_width"] for _, _, (x1, y1, x2, y2) in green_results)
        blue_seen  = any((x2 - x1) > s["blue_box_width"]  for _, _, (x1, y1, x2, y2) in blue_results)
        red_seen   = any((x2 - x1) > s["red_box_width"]   for _, _, (x1, y1, x2, y2) in red_results)

        with _colour_cache_lock:
            _colour_cache["green"] = green_seen
            _colour_cache["blue"]  = blue_seen
            _colour_cache["red"]   = red_seen

        time.sleep(1.0 / COLOUR_LOOP_HZ)

    cap.release()


def sensor_loop():
    """Reads IMU + encoder data over serial."""
    print("[sensors] thread started")
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    except Exception as e:
        print(f"[sensors] failed to open serial: {e}")
        return

    HEAD1, HEAD2 = 0xAA, 0x55
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
            rest = ser.read(frame_size - 3)
            if len(rest) != frame_size - 3: continue
            frame = bytes([HEAD1, HEAD2, frame_size]) + rest
            if (sum(frame[:-1]) & 0xFF) != frame[-1]: continue

            gx = int.from_bytes(frame[4:6],   'big', signed=True) / 32768 * 2000
            gy = int.from_bytes(frame[6:8],   'big', signed=True) / 32768 * 2000
            gz = int.from_bytes(frame[8:10],  'big', signed=True) / 32768 * 2000
            ax = int.from_bytes(frame[10:12], 'big', signed=True) / 32768 * 2 * 9.8
            ay = int.from_bytes(frame[12:14], 'big', signed=True) / 32768 * 2 * 9.8
            az = int.from_bytes(frame[14:16], 'big', signed=True) / 32768 * 2 * 9.8
            with _imu_cache_lock:
                _imu_cache.update({
                    "ax": round(ax, 2), "ay": round(ay, 2), "az": round(az, 2),
                    "gx": round(gx, 1), "gy": round(gy, 1), "gz": round(gz, 1),
                })

            lvel = int.from_bytes(frame[34:36], 'big', signed=True)
            rvel = int.from_bytes(frame[36:38], 'big', signed=True)
            now  = time.time()
            dt   = now - last_time
            if dt > 0:
                speed_ms = ((lvel + rvel) / 2.0) * SPEED_SCALE
                total_distance += speed_ms * dt
                with _encoder_cache_lock:
                    _encoder_cache["speed"]    = round(speed_ms, 3)
                    _encoder_cache["distance"] = round(total_distance, 3)
            last_time = now
        except Exception:
            time.sleep(0.1)


def firebase_loop():
    """Batches telemetry and uploads to Firebase every 2 s."""
    batch = {}
    last_upload = time.time()
    while True:
        try:
            test_id, ts, dp = telemetry_queue.get(timeout=0.5)
            if test_id not in batch:
                batch[test_id] = {}
            batch[test_id][str(int(ts * 1000))] = dp
        except queue.Empty:
            pass
        if time.time() - last_upload >= 2.0:
            if batch:
                try:
                    for tid, b_data in batch.items():
                        requests.patch(
                            f"{FIREBASE_URL}/Tune Q/{tid}.json",
                            json=b_data, timeout=5,
                        )
                    batch.clear()
                except Exception:
                    pass
            last_upload = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Main control loop
# ─────────────────────────────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    """
    Reads camera, runs lane PID, resolves colour + lidar priority,
    and issues drive commands to the car.
    """
    global _boosting
    cap = open_camera()
    fps_counter, fps_time, frame_idx = 0, time.time(), 0

    print("[loop] Integrated control loop started")

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
        annotated, error, steer, left_found, right_found, lane_width, left_x, right_x = \
            process_frame(frame, s_copy, do_annotate)
        lane_found = left_found or right_found
        if do_annotate:
            _encode_pool.submit(_do_encode, annotated)

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        # Read all sensor caches
        with _lidar_cache_lock:
            lidar_closest      = _lidar_cache["closest"]
            lidar_closest_left = _lidar_cache["closest_left"]
            lidar_blocked      = _lidar_cache["blocked"]
        with _colour_cache_lock:
            c_green = _colour_cache["green"]
            c_blue  = _colour_cache["blue"]
            c_red   = _colour_cache["red"]
        with _encoder_cache_lock:
            enc_speed = _encoder_cache["speed"]
            enc_dist  = _encoder_cache["distance"]
        with _imu_cache_lock:
            imu = dict(_imu_cache)

        # ── Drive command resolution (priority: lidar > red > blue > green > lane) ──
        autonomy_state = "DISABLED"

        if s_copy["enabled"]:
            if lidar_blocked:
                # Highest priority: physical obstacle
                car.stop()
                autonomy_state = "LIDAR_STOP"

            elif c_red:
                # Red signal: full stop regardless of colour boost state
                car.stop()
                autonomy_state = "RED_STOP"

            else:
                # Colour speed state machine
                if c_blue and _boosting:
                    # Blue cancels the boost — back to base speed
                    _boosting = False

                if c_green and not _boosting:
                    # Green triggers boost (only once until blue resets it)
                    _boosting = True

                speed = s_copy["fast_speed"] if _boosting else s_copy["base_speed"]
                car.steer(steer)
                car.forward(speed)

                if _boosting:
                    autonomy_state = "BOOST"
                elif c_blue:
                    # We just came off a boost via blue
                    autonomy_state = "SLOWDOWN"
                else:
                    autonomy_state = "FOLLOW"
        else:
            car.stop()
            _boosting = False

        # Update shared state
        with state_lock:
            state["error"]             = round(error, 3)
            state["steer"]             = round(steer, 3)
            state["lane_found"]        = lane_found
            state["lidar_closest"]     = lidar_closest
            state["lidar_closest_left"] = lidar_closest_left
            state["lidar_blocked"]     = lidar_blocked
            state["colour_green"]      = c_green
            state["colour_blue"]       = c_blue
            state["colour_red"]        = c_red
            state["autonomy_state"]    = autonomy_state
            state["enc_speed"]         = enc_speed
            state["enc_dist"]          = enc_dist
            state["imu_ax"] = imu["ax"]; state["imu_ay"] = imu["ay"]; state["imu_az"] = imu["az"]
            state["imu_gx"] = imu["gx"]; state["imu_gy"] = imu["gy"]; state["imu_gz"] = imu["gz"]

        # Telemetry
        if s_copy.get("is_testing") and s_copy.get("test_id"):
            dp = {
                "error": error, "steer": steer,
                "autonomy_state": autonomy_state,
                "speed": s_copy["fast_speed"] if _boosting else s_copy["base_speed"],
                "lidar_closest": lidar_closest,
                "enc_dist": enc_dist,
            }
            telemetry_queue.put((s_copy["test_id"], time.time(), dp))
            with data_lock:
                data_log.append({"timestamp": round(time.time(), 3), **dp})

        frame_idx += 1

    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Flask dashboard
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 6 — Integrated Lane + LiDAR + Colour</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg: #0b0f19; --surface: #111827; --border: #1f2937;
    --accent: #3b82f6; --accent-h: #2563eb;
    --green: #10b981; --red: #ef4444; --warn: #f59e0b; --blue: #38bdf8;
    --text: #f3f4f6; --muted: #9ca3af;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); padding: 1.5rem; }

  .header { display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 1.5rem; }
  .header h1 { font-size: 1.25rem; font-weight: 700; letter-spacing: 0.04em; }
  .badge { background: var(--border); padding: .2rem .7rem; border-radius: 999px;
           font-size: .78rem; font-family: monospace; color: var(--muted); }

  .layout { display: grid; grid-template-columns: 1fr 360px; gap: 1.5rem; }
  .panel { background: var(--surface); border: 1px solid var(--border);
           border-radius: 12px; padding: 1.25rem; }
  .panel-title { font-size: .8rem; text-transform: uppercase; letter-spacing: .1em;
                 color: var(--muted); margin-bottom: .9rem; font-weight: 600; }

  /* State badge */
  #state-pill {
    display: inline-block; padding: .35rem 1.1rem; border-radius: 999px;
    font-size: 1rem; font-weight: 700; letter-spacing: .06em;
    background: var(--border); color: var(--text); margin-bottom: 1rem;
    transition: background .25s, color .25s;
  }
  #state-pill.FOLLOW   { background: var(--green); color: #061612; }
  #state-pill.BOOST    { background: #a3e635; color: #1a2e05; }
  #state-pill.SLOWDOWN { background: var(--blue); color: #082637; }
  #state-pill.RED_STOP { background: var(--red); color: #fff; }
  #state-pill.LIDAR_STOP { background: var(--warn); color: #1c1000; }
  #state-pill.DISABLED { background: var(--border); color: var(--muted); }

  /* Colour indicators */
  .colour-row { display: flex; gap: .6rem; margin-bottom: 1rem; }
  .colour-dot { width: 28px; height: 28px; border-radius: 50%; opacity: .2;
                transition: opacity .2s, box-shadow .2s; border: 2px solid transparent; }
  .colour-dot.active { opacity: 1; box-shadow: 0 0 10px 3px currentColor; }
  .dot-green { background: #22c55e; color: #22c55e; }
  .dot-blue  { background: #38bdf8; color: #38bdf8; }
  .dot-red   { background: #ef4444; color: #ef4444; }

  /* Stats grid */
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin-bottom: 1rem; }
  .stat-box { background: #182235; padding: .75rem; border-radius: 8px; border: 1px solid var(--border); }
  .stat-label { font-size: .7rem; color: var(--muted); text-transform: uppercase; margin-bottom: .2rem; }
  .stat-value { font-size: 1.1rem; font-weight: 700; font-family: monospace; color: var(--accent); }

  /* Camera feed */
  #feed { width: 100%; border-radius: 8px; display: block; background: #000; min-height: 180px; margin-bottom: .75rem; }
  .error-track { position: relative; height: 10px; background: var(--border); border-radius: 5px; overflow: hidden; }
  #error-bar { position: absolute; height: 100%; width: 6px; background: var(--accent);
               left: 50%; transform: translateX(-50%); transition: left .1s; border-radius: 3px; }

  /* Controls */
  .slider-group { display: flex; flex-direction: column; gap: .35rem; margin-bottom: 1rem; }
  .slider-group label { font-size: .82rem; color: var(--muted); display: flex; justify-content: space-between; }
  .slider-group input[type=range] { width: 100%; accent-color: var(--accent); }
  .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-bottom: 1rem; }
  button { padding: .65rem 1rem; border: none; border-radius: 8px; font-family: var(--font);
           font-weight: 600; font-size: .85rem; cursor: pointer; transition: all .2s; }
  .btn-go   { background: var(--green); color: #061612; }
  .btn-stop { background: var(--red);   color: #fff; }
  .btn-test { background: var(--accent); color: #fff; width: 100%; margin-bottom: .5rem; }
  .btn-test.active { background: var(--warn); color: #1c1000; }
  .btn-dl { background: var(--border); color: var(--text); width: 100%;
            text-decoration: none; display: block; text-align: center;
            font-weight: 600; font-size: .85rem; padding: .65rem; border-radius: 8px; }
  hr.sep { border: 0; border-top: 1px solid var(--border); margin: 1rem 0; }

  /* Chart */
  .chart-wrap { position: relative; height: 220px; background: #182235; border-radius: 8px;
                padding: .75rem; border: 1px solid var(--border); margin-top: 1rem; }

  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>&#9675; Exp 6 &mdash; Integrated Lane + LiDAR + Colour</h1>
  <div class="badge">Mode: <span id="mode-badge">IDLE</span></div>
</div>

<div class="layout">
  <!-- Left: Feed + telemetry -->
  <div>
    <div class="panel">
      <div class="panel-title">Camera (annotated)</div>
      <img id="feed" src="/video_feed" alt="camera">
      <div class="error-track" style="margin-top:.5rem" title="Lane error">
        <div id="error-bar"></div>
      </div>
    </div>

    <!-- State pill + colour dots -->
    <div class="panel" style="margin-top:1rem">
      <div class="panel-title">Autonomy State</div>
      <span id="state-pill" class="DISABLED">DISABLED</span>
      <div class="colour-row">
        <div class="colour-dot dot-green" id="dot-green" title="Green signal"></div>
        <div class="colour-dot dot-blue"  id="dot-blue"  title="Blue signal"></div>
        <div class="colour-dot dot-red"   id="dot-red"   title="Red signal"></div>
      </div>

      <div class="stats">
        <div class="stat-box"><div class="stat-label">Lane</div><div class="stat-value" id="v-lane">NO</div></div>
        <div class="stat-box"><div class="stat-label">FPS</div><div class="stat-value" id="v-fps">0</div></div>
        <div class="stat-box"><div class="stat-label">Enc Speed</div><div class="stat-value" id="v-spd">0.000</div></div>
        <div class="stat-box"><div class="stat-label">LiDAR Front</div><div class="stat-value" id="v-lidar">0 mm</div></div>
        <div class="stat-box"><div class="stat-label">LiDAR Left</div><div class="stat-value" id="v-lidar-l">0 mm</div></div>
        <div class="stat-box"><div class="stat-label">Enc Dist</div><div class="stat-value" id="v-dist">0.00 m</div></div>
      </div>

      <div class="chart-wrap">
        <canvas id="chartMain"></canvas>
      </div>
    </div>
  </div>

  <!-- Right: Controls -->
  <div class="panel">
    <div class="panel-title">Drive Control</div>
    <div class="btn-row">
      <button class="btn-go"   onclick="setEnabled(true)">&#9654; GO</button>
      <button class="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
    </div>

    <div class="slider-group">
      <label>Base Speed <span id="v-base_speed">0.15</span></label>
      <input type="range" id="base_speed" min="0" max="60" value="15" step="1">
    </div>
    <div class="slider-group">
      <label>Fast Speed (Green) <span id="v-fast_speed">0.25</span></label>
      <input type="range" id="fast_speed" min="0" max="60" value="25" step="1">
    </div>

    <hr class="sep">
    <div class="panel-title">LiDAR Safety</div>
    <div class="slider-group">
      <label>Stop Distance (mm) <span id="v-stop_distance">400</span></label>
      <input type="range" id="stop_distance" min="100" max="2000" value="400" step="10">
    </div>

    <hr class="sep">
    <div class="panel-title">Colour Triggers (px width)</div>
    <div class="slider-group">
      <label>Green box width <span id="v-green_box_width">250</span></label>
      <input type="range" id="green_box_width" min="50" max="500" value="250" step="5">
    </div>
    <div class="slider-group">
      <label>Blue box width <span id="v-blue_box_width">200</span></label>
      <input type="range" id="blue_box_width" min="50" max="500" value="200" step="5">
    </div>
    <div class="slider-group">
      <label>Red box width <span id="v-red_box_width">150</span></label>
      <input type="range" id="red_box_width" min="50" max="500" value="150" step="5">
    </div>

    <hr class="sep">
    <div class="panel-title">PID Tuning</div>
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

    <hr class="sep">
    <div class="panel-title">Data Collection</div>
    <button id="btn-test" class="btn-test" onclick="toggleTest()">START RECORDING</button>
    <a href="/download_csv" class="btn-dl" target="_blank">&#11123; Download CSV</a>
  </div>
</div>

<script>
// ── Slider setup ────────────────────────────────────────────────────────────
const sliderDefs = [
  { id: "base_speed",      scale: 1/100, fmt: v => v.toFixed(2) },
  { id: "fast_speed",      scale: 1/100, fmt: v => v.toFixed(2) },
  { id: "stop_distance",   scale: 1,     fmt: v => v.toFixed(0) },
  { id: "green_box_width", scale: 1,     fmt: v => v.toFixed(0) },
  { id: "blue_box_width",  scale: 1,     fmt: v => v.toFixed(0) },
  { id: "red_box_width",   scale: 1,     fmt: v => v.toFixed(0) },
  { id: "kp",  scale: 1, fmt: v => v.toFixed(3) },
  { id: "ki",  scale: 1, fmt: v => v.toFixed(3) },
  { id: "kd",  scale: 1, fmt: v => v.toFixed(3) },
];
sliderDefs.forEach(({id, scale, fmt}) => {
  const el   = document.getElementById(id);
  const disp = document.getElementById("v-" + id);
  if (!el) return;
  el.addEventListener("input", () => {
    const v = parseFloat(el.value) * scale;
    disp.textContent = fmt(v);
    sendParam(id, v);
  });
});

function sendParam(k, v) {
  fetch("/set", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({[k]:v}) });
}
function setEnabled(v) {
  fetch("/set", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled:v}) });
}

// ── Chart ───────────────────────────────────────────────────────────────────
Chart.defaults.color = '#9ca3af';
Chart.defaults.font.family = "'Inter', sans-serif";
const chartMain = new Chart(document.getElementById('chartMain'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Lane Error', borderColor: '#ef4444', data: [], fill: false },
    { label: 'Steer',      borderColor: '#3b82f6', data: [], fill: false },
  ]},
  options: {
    responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
    scales: { x: { display: false }, y: { min: -1.3, max: 1.3, grid: { color: '#1f2937' } } },
    plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
    elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.1 } }
  }
});

// ── Telemetry polling ────────────────────────────────────────────────────────
let isTesting = false, dataPoints = 0;
function toggleTest() {
  isTesting = !isTesting;
  fetch("/set", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({is_testing: isTesting}) });
  if (isTesting) {
    chartMain.data.labels = [];
    chartMain.data.datasets.forEach(d => d.data = []);
    dataPoints = 0;
  }
}

const STATE_CLASSES = ["FOLLOW","BOOST","SLOWDOWN","RED_STOP","LIDAR_STOP","DISABLED"];
async function poll() {
  try {
    const d = await (await fetch("/status")).json();

    // State pill
    const pill = document.getElementById("state-pill");
    pill.textContent = d.autonomy_state;
    STATE_CLASSES.forEach(c => pill.classList.remove(c));
    pill.classList.add(d.autonomy_state);

    // Colour dots
    document.getElementById("dot-green").classList.toggle("active", d.colour_green);
    document.getElementById("dot-blue" ).classList.toggle("active", d.colour_blue);
    document.getElementById("dot-red"  ).classList.toggle("active", d.colour_red);

    // Stats
    document.getElementById("v-lane" ).textContent = d.lane_found ? "YES" : "NO";
    document.getElementById("v-fps"  ).textContent = d.fps;
    document.getElementById("v-spd"  ).textContent = d.enc_speed.toFixed(3) + " m/s";
    document.getElementById("v-lidar").textContent = d.lidar_closest.toFixed(0) + " mm";
    document.getElementById("v-lidar-l").textContent = d.lidar_closest_left.toFixed(0) + " mm";
    document.getElementById("v-dist" ).textContent = d.enc_dist.toFixed(2) + " m";

    // Error bar
    const pct = (d.error + 1) / 2 * 100;
    const bar = document.getElementById("error-bar");
    bar.style.left = pct + "%";
    bar.style.background = Math.abs(d.error) > 0.5 ? "#ef4444" : "#10b981";

    // Mode badge
    isTesting = d.is_testing;
    const btn = document.getElementById("btn-test");
    const badge = document.getElementById("mode-badge");
    if (isTesting) {
      btn.textContent = "STOP RECORDING"; btn.classList.add("active");
      badge.textContent = "RECORDING"; badge.style.color = "#f59e0b";
      dataPoints++;
      chartMain.data.labels.push(dataPoints);
      chartMain.data.datasets[0].data.push(d.error);
      chartMain.data.datasets[1].data.push(d.steer);
      chartMain.update();
    } else {
      btn.textContent = "START RECORDING"; btn.classList.remove("active");
      badge.textContent = "IDLE"; badge.style.color = "#9ca3af";
    }
  } catch(e) {}
  setTimeout(poll, 150);
}
poll();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(MJPEG_INTERVAL)
    finally:
        with clients_lock:
            stream_clients -= 1

@app.route("/status")
def status():
    with state_lock:
        return jsonify({k: state[k] for k in (
            "fps", "error", "steer", "lane_found", "enabled",
            "lidar_closest", "lidar_closest_left", "lidar_blocked",
            "colour_green", "colour_blue", "colour_red",
            "autonomy_state",
            "enc_speed", "enc_dist",
            "imu_ax", "imu_ay", "imu_az",
            "imu_gx", "imu_gy", "imu_gz",
            "is_testing", "test_id",
        )})

@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k == "is_testing":
                if v and not state.get("is_testing", False):
                    state["test_id"]   = f"test_{int(time.time())}"
                    state["is_testing"] = True
                    with data_lock:
                        data_log.clear()
                elif not v:
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
            return "No data recorded yet.", 400
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data_log[0].keys())
        writer.writeheader()
        writer.writerows(data_log)
    return Response(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=exp6_telemetry.csv"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = JetRacer(init_lidar=True)
    car.arm(delay=3)

    threading.Thread(target=lidar_loop,   args=(car,), daemon=True).start()
    threading.Thread(target=colour_loop,              daemon=True).start()
    threading.Thread(target=sensor_loop,              daemon=True).start()
    threading.Thread(target=firebase_loop,            daemon=True).start()
    threading.Thread(target=control_loop, args=(car,), daemon=True).start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
