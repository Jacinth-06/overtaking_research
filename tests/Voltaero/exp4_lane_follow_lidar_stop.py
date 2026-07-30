#!/usr/bin/env python3
"""
exp4_lane_follow_lidar_stop.py — Lane follower + Lidar stop + Flask dashboard
(No Overtaking)
"""

import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import threading
import time
import serial
import struct
import requests
import queue
import math
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
    "canny_lo": 50,   "canny_hi": 150,
    "binary_thresh": 200,
    "blur_ksize": 5,
    "morph_ksize": 5,  "morph_iters": 2,
    "roi_top_frac": 0.5,
    "roi_side_limit": 0.0,
    "kp": 0.55,  "ki": 0.003,  "kd": 0.25,
    "speed": 0.15,
    "enabled": False,
    "stop_distance": 400.0,
    "imu_ax": 0, "imu_ay": 0, "imu_az": 0,
    "imu_gx": 0, "imu_gy": 0, "imu_gz": 0,
    "enc_speed": 0.0, "enc_dist": 0.0,
    "error": 0.0, "steer": 0.0, "fps": 0,
    "lane_found": False,
    "lidar_closest": 0.0,
    "lidar_closest_left": 0.0,
    "lidar_blocked": False,
    "autonomy_state": "FOLLOW",
    "is_testing": False,
    "test_id": "",
    "reset_encoder_dist": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time(), "state_start_time": time.time()}
state_lock = threading.Lock()

FIREBASE_URL = "https://jetracer-f1b1c-default-rtdb.asia-southeast1.firebasedatabase.app"
telemetry_queue = queue.Queue()

_last_steer = 0.0

frame_lock   = threading.Lock()
latest_frame = None
stream_clients = 0
clients_lock   = threading.Lock()
_encode_pool = ThreadPoolExecutor(max_workers=1)

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
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    raise RuntimeError("No camera found")

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

    lane_found, left_found, right_found = False, False, False
    lane_width, left_x, right_x = 0.0, 0.0, 0.0

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

        steer = (s["kp"] * error + s["ki"] * pid_state["integral"] + s["kd"] * derivative)
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
        if lane_found: cv2.circle(annotated, (int(target_x), roi_top + 10), 8, (0, 255, 0), -1)
    else:
        annotated = frame
    return annotated, error, steer, left_found, right_found, lane_width, left_x, right_x

def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret: return
    with frame_lock:
        global latest_frame
        latest_frame = jpeg.tobytes()

def firebase_loop():
    batch = {}
    last_upload_time = time.time()
    while True:
        try:
            test_id, timestamp, data_point = telemetry_queue.get(timeout=0.5)
            if test_id not in batch: batch[test_id] = {}
            key = str(int(timestamp * 1000))
            batch[test_id][key] = data_point
        except queue.Empty: pass
        now = time.time()
        if now - last_upload_time >= 2.0:
            if batch:
                try:
                    for tid, b_data in batch.items():
                        requests.patch(f"{FIREBASE_URL}/Tune Q/{tid}.json", json=b_data, timeout=5)
                    batch.clear()
                except Exception as e: pass
            last_upload_time = now

_lidar_cache      = {"closest": 0.0, "closest_left": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()
def lidar_loop(car: JetRacer):
    while True:
        try:
            with state_lock: STOP_DISTANCE = state["stop_distance"]
            scan = car.lidar_scan(samples=150)
            front_distances = [dist for ang, dist in scan.items() if (ang >= 320 or ang <= 40) and dist > 10]
            left_distances = [dist for ang, dist in scan.items() if (250 <= ang <= 310) and dist > 10]
            closest_front = min(front_distances) if front_distances else 0.0
            closest_left = min(left_distances) if left_distances else 0.0
            is_blocked = closest_front > 0 and closest_front < STOP_DISTANCE
            with _lidar_cache_lock:
                _lidar_cache["closest"] = round(closest_front, 1)
                _lidar_cache["closest_left"] = round(closest_left, 1)
                _lidar_cache["blocked"] = is_blocked
        except Exception as e:
            with _lidar_cache_lock:
                _lidar_cache["closest"] = 0.0; _lidar_cache["closest_left"] = 0.0; _lidar_cache["blocked"] = True
        time.sleep(0.05)

_imu_cache = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}
_imu_cache_lock = threading.Lock()
_encoder_cache = {"speed": 0.0, "distance": 0.0}
_encoder_cache_lock = threading.Lock()

def sensor_loop():
    try: ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    except Exception as e: return
    HEAD1, HEAD2 = 0xAA, 0x55
    total_distance, last_lvel, last_rvel, last_time = 0.0, 0, 0, time.time()
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
                _imu_cache["ax"] = round(ax, 2); _imu_cache["ay"] = round(ay, 2); _imu_cache["az"] = round(az, 2)
                _imu_cache["gx"] = round(gx, 1); _imu_cache["gy"] = round(gy, 1); _imu_cache["gz"] = round(gz, 1)
            lvel = int.from_bytes(frame[34:36], 'big', signed=True)
            rvel = int.from_bytes(frame[36:38], 'big', signed=True)
            now = time.time(); dt = now - last_time
            if dt > 0:
                speed_ms  = ((lvel + rvel) / 2.0) * SPEED_SCALE
                total_distance += speed_ms * dt
                with _encoder_cache_lock:
                    _encoder_cache["speed"]    = round(speed_ms, 3)
                    _encoder_cache["distance"] = round(total_distance, 3)
            last_lvel, last_rvel, last_time = lvel, rvel, now
        except Exception: time.sleep(0.1)

def control_loop(car: JetRacer):
    cap = open_camera()
    fps_counter, fps_time, frame_idx = 0, time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.01); continue
        if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH: frame = cv2.resize(frame, (WIDTH, HEIGHT))
        with state_lock: s_copy = dict(state)
        with clients_lock: has_clients = stream_clients > 0
        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        annotated, error, steer, left_found, right_found, lane_width, left_x, right_x = process_frame(frame, s_copy, do_annotate)
        lane_found = left_found or right_found
        if do_annotate: _encode_pool.submit(_do_encode, annotated)
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock: state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        with _lidar_cache_lock: lidar_closest = _lidar_cache["closest"]; lidar_blocked = _lidar_cache["blocked"]
        with _encoder_cache_lock: enc_speed = _encoder_cache["speed"]; enc_dist = _encoder_cache["distance"]

        autonomy_state = "FOLLOW"
        
        if s_copy["enabled"]:
            if lidar_blocked:
                car.stop()
                autonomy_state = "STOPPED"
            else:
                car.steer(steer)
                car.forward(s_copy["speed"])
        else:
            car.stop()

        with state_lock:
            state["error"] = round(error, 3); state["steer"] = round(steer, 3); state["autonomy_state"] = autonomy_state
            state["enc_speed"] = enc_speed; state["enc_dist"] = enc_dist

        if s_copy.get("is_testing", False) and s_copy.get("test_id"):
            telemetry_queue.put((s_copy["test_id"], time.time(), {"error": error, "steer": steer, "autonomy_state": autonomy_state, "enc_dist": enc_dist}))
        frame_idx += 1
    cap.release()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 4 — Lane Follow + LiDAR Stop</title>
<style>
  :root { --bg: #0e1117; --surface: #161b27; --border: #2a3040; --accent: #00d4aa; --warn: #ffb020; --danger: #ff4d4d; --text: #e8ecf1; --muted: #6b7a99; --font: 'JetBrains Mono', 'Fira Mono', monospace; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 1rem; }
  h1 { font-size: 1rem; letter-spacing: .15em; color: var(--accent); text-transform: uppercase; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 360px; gap: 1rem; width: 100%; max-width: 1200px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
  .card h2 { font-size: .7rem; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; margin-bottom: .75rem; }
  img#feed { width: 100%; border-radius: 6px; display: block; background: #000; min-height: 180px; }
  .status-bar { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: .75rem; }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
  .slider-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .55rem; }
  .slider-row label { font-size: .7rem; color: var(--muted); width: 85px; flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val { font-size: .75rem; width: 45px; text-align: right; color: var(--text); }
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; }
  button { padding: .45rem 1.1rem; border: none; border-radius: 6px; cursor: pointer; font-family: var(--font); font-size: .8rem; font-weight: 600; letter-spacing: .04em; }
  #btn-go { background: var(--accent); color: #061612; }
  #btn-stop { background: var(--danger); color: #fff; }
</style>
</head>
<body>
<h1>&#9675; Exp 4 &mdash; Lane Follow + LiDAR Stop</h1>
<div class="grid">
  <div class="card">
    <h2>Camera feed (annotated)</h2>
    <img id="feed" src="/video_feed" alt="camera">
    <div class="status-bar" style="margin-top:.75rem">
      <div class="stat"><span class="stat-val" id="v-state">FOLLOW</span><span class="stat-lbl">state</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span><span class="stat-lbl">error</span></div>
      <div class="stat"><span class="stat-val" id="v-enc-dist">0.00</span><span class="stat-lbl">enc dist</span></div>
    </div>
  </div>
  <div class="card">
    <h2>Drive</h2>
    <div class="slider-row"><label>Speed</label><input type="range" id="speed" min="0" max="60" value="15" step="1"><span class="val" id="v-speed">0.15</span></div>
    <div class="btn-row">
      <button id="btn-go" onclick="setEnabled(true)">&#9654; GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
      <button id="btn-test" onclick="toggleTest()" style="background: #6b7a99; color: #fff;">&#9654; START TEST</button>
    </div>
    <hr class="divider">
    <h2>Lidar Safety</h2>
    <div class="slider-row"><label>Stop Dist</label><input type="range" id="stop_distance" min="100" max="2000" value="400" step="10"><span class="val" id="v-stop_distance">400</span></div>
  </div>
</div>
<script>
let isTesting = false;
function toggleTest() { isTesting = !isTesting; fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({is_testing: isTesting})}); }
function setEnabled(v) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled: v})}); }
document.getElementById("speed").addEventListener("input", (e) => { document.getElementById("v-speed").textContent = (e.target.value/100).toFixed(2); fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({speed: e.target.value/100})}); });
async function poll() {
  try {
    const r = await fetch("/status"); const d = await r.json();
    document.getElementById("v-state").textContent = d.autonomy_state;
    document.getElementById("v-err").textContent = d.error.toFixed(2);
    document.getElementById("v-enc-dist").textContent = d.enc_dist.toFixed(2);
  } catch(e) {}
  setTimeout(poll, 250);
}
poll();
</script>
</body>
</html>"""

@app.route("/")
def index(): return render_template_string(DASHBOARD_HTML)
@app.route("/video_feed")
def video_feed(): return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")
def generate_mjpeg():
    global stream_clients; with clients_lock: stream_clients += 1
    try:
        while True:
            with frame_lock: frame = latest_frame
            if frame is None: time.sleep(0.02); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(MJPEG_INTERVAL)
    finally:
        with clients_lock: stream_clients -= 1

@app.route("/status")
def status():
    with state_lock: return jsonify({k: state[k] for k in ("error", "autonomy_state", "enc_dist")})

@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k == "is_testing":
                if v: state["test_id"] = f"test_{int(time.time())}"; state["is_testing"] = True
                else: state["is_testing"] = False
            elif k in state:
                if k == "enabled" and not v: state["reset_encoder_dist"] = True
                state[k] = v
    return jsonify({"ok": True})

if __name__ == "__main__":
    car = JetRacer(init_lidar=True)
    car.arm(delay=3)
    threading.Thread(target=lidar_loop, args=(car,), daemon=True).start()
    threading.Thread(target=firebase_loop, daemon=True).start()
    threading.Thread(target=sensor_loop, daemon=True).start()
    threading.Thread(target=control_loop, args=(car,), daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
