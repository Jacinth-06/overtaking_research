#!/usr/bin/env python3
"""
version1.py — GPU-accelerated actual lane follower + Lidar safety stop + Flask dashboard
Combines:  actual lane following (no offsets)
           safety_stop.py     (lidar obstacle detection)
Optimised for Jetson Nano 4 GB.

Detection pipeline:
  frame → grayscale → Gaussian blur → (Canny + Binary threshold)
  → OR combine → morphology clean → bird-eye warp
  → histogram → sliding-window lane detect → lane centre → PID → motor

Safety layer:
  lidar front cone (320°–360° + 0°–40°) → if min distance < STOP_DISTANCE → override stop
  Lidar runs in its own background thread at ~10 Hz so it never blocks the camera loop.

Run:   python version1.py
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
    # ROI
    "roi_top_frac": 0.5,
    "roi_side_limit": 0.0,
    # PID
    "kp": 0.55,  "ki": 0.003,  "kd": 0.25,
    # Drive
    "speed": 0.15,
    "enabled": False,
    # Lidar safety
    "stop_distance": 400.0,   # mm — stop if object closer than this
    # Telemetry (read-only from browser)
    "error": 0.0, "steer": 0.0, "fps": 0,
    "lane_found": False,
    "lidar_closest": 0.0,     # closest front distance in mm
    "lidar_closest_left": 0.0, # closest left distance in mm
    "lidar_blocked": False,   # True when obstacle within stop_distance
    "autonomy_state": "FOLLOW",
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time(), "state_start_time": time.time()}
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


# ── GPU helpers (probed at startup) ──────────────────────────────────────────
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


# ── Frame processing pipeline ────────────────────────────────────────────────
def process_frame(frame, s, annotate: bool):
    """
    Pipeline: grayscale → blur → (Canny + Binary) → OR → morphology
              → ROI contour → lane centre → PID
    """
    global _last_steer
    h, w = frame.shape[:2]
    roi_top = int(h * s.get("roi_top_frac", 0.5))
    x_start = int(w * s.get("roi_side_limit", 0.0))
    x_end   = w - x_start

    # Crop early to save processing time
    roi_bgr = frame[roi_top:h, x_start:x_end]

    # 1. Grayscale (GPU-accelerated if available)
    gray = to_gray(roi_bgr)

    # 2. Gaussian blur
    bk = s["blur_ksize"] | 1   # ensure odd
    blurred = cv2.GaussianBlur(gray, (bk, bk), 0)

    # 3a. Canny edge detection
    edges = cv2.Canny(blurred, s["canny_lo"], s["canny_hi"])

    # 3b. Binary threshold (white lines)
    _, binary = cv2.threshold(blurred, s["binary_thresh"], 255, cv2.THRESH_BINARY)

    # 4. AND combine — drop the extra dilate; morph_close below handles gap-filling
    combined = cv2.bitwise_and(edges, binary)

    # 5. Morphology clean (two ops instead of three — removed stray dilate)
    mk = s["morph_ksize"] | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    cleaned  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=s["morph_iters"])
    roi_mask = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel, iterations=1)

    # 6. ROI and Contour Logic
    ys, xs = np.where(roi_mask > 0)

    lane_found = False
    left_found = False
    right_found = False
    lane_width = 0.0

    # 7. PID
    if len(xs) > 50:
        lane_found = True
        
        # Calculate the geometric middle of your cropped ROI
        roi_width = x_end - x_start
        mid_point = roi_width / 2.0
        
        # Split detected pixels into left-of-center and right-of-center
        left_pixels = xs[xs < mid_point]
        right_pixels = xs[xs >= mid_point]
        
        left_found = len(left_pixels) > 10
        right_found = len(right_pixels) > 10
        
        lane_width = 0.0

        if left_found and right_found:
            # BOTH LANES DETECTED
            # Find the center of the left line and center of the right line individually
            left_x = np.mean(left_pixels) + x_start
            right_x = np.mean(right_pixels) + x_start
            
            # The true centroid is exactly between the two lines
            target_x = (left_x + right_x) / 2.0
            lane_width = right_x - left_x
            
        elif len(left_pixels) > 10:
            # ONLY LEFT LANE DETECTED (e.g., sharp left turn)
            left_x = np.mean(left_pixels) + x_start
            target_x = left_x + 140  # Hardcoded fallback offset
            
        elif len(right_pixels) > 10:
            # ONLY RIGHT LANE DETECTED (e.g., sharp right turn)
            right_x = np.mean(right_pixels) + x_start
            target_x = right_x - 140 # Hardcoded fallback offset
            
        else:
            # Failsafe
            target_x = w / 2.0

        error = (target_x - w / 2.0) / (w / 2.0) * 3.5  # normalise to [-1, 1]

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
        annotated = frame.copy()

        # Draw ROI boundary
        cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)
        cv2.line(annotated, (x_start, roi_top), (x_start, h), (255, 0, 255), 1)
        cv2.line(annotated, (x_end, roi_top), (x_end, h), (255, 0, 255), 1)

        # Mask overlay
        mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        mask_3ch[:, :, 0] = 0
        annotated[roi_top:h, x_start:x_end] = cv2.addWeighted(
            annotated[roi_top:h, x_start:x_end], 0.7, mask_3ch, 0.3, 0)

        if lane_found:
            cv2.circle(annotated, (int(target_x), roi_top + 10), 8, (0, 255, 0), -1)
    else:
        annotated = frame

    return annotated, error, steer, left_found, right_found, lane_width


# ── Async JPEG encode ─────────────────────────────────────────────────────────
def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return
    with frame_lock:
        global latest_frame
        latest_frame = jpeg.tobytes()


# ── Lidar background thread ───────────────────────────────────────────────────
# Runs independently at ~10 Hz; never blocks the camera / vision loop.
# The control loop does a cheap non-blocking dict read instead of calling
# car.lidar_scan() every frame (which was the main FPS killer).

_lidar_cache      = {"closest": 0.0, "closest_left": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()

def lidar_loop(car: JetRacer):
    """
    Background thread: poll lidar and cache the result.
    Scans front (320-360, 0-40) and left (250-310) continuously.
    Does not block the control loop.
    """
    print("[lidar] Background safety thread started")
    while True:
        try:
            with state_lock:
                STOP_DISTANCE = state["stop_distance"]

            # Use enough samples to get good coverage for both front and left
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
            print(f"[lidar] scan error: {e}")
            with _lidar_cache_lock:
                _lidar_cache["closest"] = 0.0
                _lidar_cache["closest_left"] = 0.0
                _lidar_cache["blocked"] = True

        time.sleep(0.10)   # ~10 Hz — fast enough for obstacle reaction


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

        if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # 1. Grab a quick snapshot of the state and RELEASE the lock immediately
        with state_lock:
            s_copy = dict(state)

        with clients_lock:
            has_clients = stream_clients > 0

        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        annotated, error, steer, left_found, right_found, lane_width = process_frame(frame, s_copy, do_annotate)
        lane_found = left_found or right_found

        if do_annotate:
            _encode_pool.submit(_do_encode, annotated)

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        # 2. Get Lidar readings from RAM cache (Ultra-fast, non-blocking)
        with _lidar_cache_lock:
            lidar_closest = _lidar_cache["closest"]
            lidar_closest_left = _lidar_cache.get("closest_left", 0.0)
            lidar_blocked = _lidar_cache["blocked"]

        # Track state using our local copy baseline
        autonomy_state = s_copy.get("autonomy_state", "FOLLOW")
        now = time.time()

        # 3. STATE MACHINE (Strictly IF/ELIF, NO WHILE LOOPS)
        if s_copy["enabled"]:
            if autonomy_state == "FOLLOW":
                # Track nominal lane width
                if left_found and right_found and lane_width > 50:
                    old_lw = pid_state.get("nominal_lane_width", lane_width)
                    pid_state["nominal_lane_width"] = 0.95 * old_lw + 0.05 * lane_width

                if lidar_blocked:
                    autonomy_state = "OVERTAKING"
                    pid_state["crossing_phase"] = 1
                    pid_state.pop("phase_debounce_time", None) # Clear any old timers
                    print(f"\n[STATE CHANGE] -> OVERTAKING. Obstacle at {lidar_closest}mm", flush=True)
                    car.steer(0.8)                              # Aggressive lane change initiation
                    car.forward(s_copy["speed"])
                else:
                    car.steer(steer)
                    car.forward(s_copy["speed"])
                    
            elif autonomy_state == "OVERTAKING":
                car.steer(0.8)                                  # Sharp right turn during overtaking
                car.forward(s_copy["speed"])
                
                phase = pid_state.get("crossing_phase", 1)
                now = time.time()
                
                # PHASE 1: Wait for original left lane to disappear STABLY for 2 full seconds
                if phase == 1:
                    if not left_found:
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 2.0:
                            pid_state["crossing_phase"] = 2
                            pid_state.pop("phase_debounce_time", None) # reset timer for next phase
                            print("\n[STATE CHANGE] OVERTAKING Phase 1 -> 2 (CONFIRMED lost old left lane)", flush=True)
                    else:
                        pid_state.pop("phase_debounce_time", None) # reset if line flickers back
                        
                # PHASE 2: Wait for original right lane to cross center and stay on the left half for 2 seconds
                elif phase == 2:
                    if left_found:
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 2.0:
                            pid_state["crossing_phase"] = 3
                            pid_state.pop("phase_debounce_time", None)
                            print("\n[STATE CHANGE] OVERTAKING Phase 2 -> 3 (CONFIRMED old right lane became new left)", flush=True)
                    else:
                        pid_state.pop("phase_debounce_time", None)
                        
                # PHASE 3: Wait for a brand new right lane to appear and stay stable for 2 seconds
                elif phase == 3:
                    target_lw = pid_state.get("nominal_lane_width", 200.0)
                    if right_found and (target_lw * 0.6 < lane_width < target_lw * 1.4):
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 2.0:
                            autonomy_state = "CHECKING"
                            pid_state["crossing_phase"] = 1
                            pid_state.pop("phase_debounce_time", None)
                            print(f"\n[STATE CHANGE] -> CHECKING. Switched to right lane safely. Width: {lane_width:.1f}", flush=True)
                            pid_state["integral"] = 0.0
                            pid_state["last_error"] = 0.0
                    else:
                        pid_state.pop("phase_debounce_time", None)
                    
            elif autonomy_state == "CHECKING":
                car.steer(steer)
                car.forward(s_copy["speed"])
                # Wait until safe left distance (obstacle crossed)
                phase = pid_state.get("crossing_phase", 1)
                if phase == 1:
                    # phase 1: detect obstacle on the left
                    if lidar_closest_left > 0.0 and lidar_closest_left < 600.0:
                        pid_state["crossing_phase"] = 2
                        print(f"\n[STATE CHANGE] CHECKING -> Phase 2 (obstacle {lidar_closest_left}mm to left)", flush=True)
                elif phase == 2:
                    # phase 2: obstacle clears
                    if lidar_closest_left == 0.0 or lidar_closest_left > 300.0:
                        autonomy_state = "RECOVERY"
                        pid_state["crossing_phase"] = 1
                        pid_state.pop("phase_debounce_time", None) # Clear timer for recovery
                        print("\n[STATE CHANGE] -> RECOVERY. Left side clear.", flush=True)
                    
            elif autonomy_state == "RECOVERY":
                car.steer(-0.8)                                 # Sharp left turn during recovery
                car.forward(s_copy["speed"])
                
                phase = pid_state.get("crossing_phase", 1)
                now = time.time()
                
                # PHASE 1: Wait for right lane to disappear STABLY for 2 seconds
                if phase == 1:
                    if not right_found:
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 2.0:
                            pid_state["crossing_phase"] = 2
                            pid_state.pop("phase_debounce_time", None)
                            print("\n[STATE CHANGE] RECOVERY Phase 1 -> 2 (CONFIRMED lost old right lane)", flush=True)
                    else:
                        pid_state.pop("phase_debounce_time", None)
                        
                # PHASE 2: Wait for left lane to cross center and become the new right lane for 2 seconds
                elif phase == 2:
                    if right_found:
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 2.0:
                            pid_state["crossing_phase"] = 3
                            pid_state.pop("phase_debounce_time", None)
                            print("\n[STATE CHANGE] RECOVERY Phase 2 -> 3 (CONFIRMED old left lane became new right)", flush=True)
                    else:
                        pid_state.pop("phase_debounce_time", None)
                        
                # PHASE 3: Wait for a brand new left lane to appear stably for 2 seconds
                elif phase == 3:
                    target_lw = pid_state.get("nominal_lane_width", 200.0)
                    if left_found and (target_lw * 0.6 < lane_width < target_lw * 1.4):
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 2.0:
                            autonomy_state = "FOLLOW"
                            pid_state["crossing_phase"] = 1
                            pid_state.pop("phase_debounce_time", None)
                            print(f"\n[STATE CHANGE] -> FOLLOW. Back in original lane safely. Width: {lane_width:.1f}", flush=True)
                            pid_state["integral"] = 0.0
                            pid_state["last_error"] = 0.0
                    else:
                        pid_state.pop("phase_debounce_time", None)
                    
            # Set telemetry steering monitor value based on current operational mode
            if autonomy_state == "OVERTAKING": steer = 0.8
            elif autonomy_state == "RECOVERY": steer = -0.8
        else:
            car.stop()
            autonomy_state = "FOLLOW"

        # 4. Save current state values back to global dictionary safely
        with state_lock:
            state["error"]         = round(error, 3)
            state["steer"]         = round(steer, 3)
            state["lane_found"]    = lane_found
            state["lidar_closest"] = lidar_closest
            state["lidar_closest_left"] = lidar_closest_left
            state["lidar_blocked"] = lidar_blocked
            state["autonomy_state"] = autonomy_state

        frame_idx += 1

    cap.release()

# ── Flask / dashboard ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetRacer MVP — Lane Follower + Lidar Safety</title>
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
<h1>&#9675; JetRacer &#183; MVP &mdash; Lane + Lidar</h1>
<div class="grid">
  <div class="card">
    <h2>Camera feed (annotated)</h2>
    <img id="feed" src="/video_feed" alt="camera">
    <div class="status-bar" style="margin-top:.75rem">
      <div class="stat"><span class="stat-val" id="v-state">FOLLOW</span>
                        <span class="stat-lbl">state</span></div>
      <div class="stat"><span class="stat-val" id="v-fps">0</span>
                        <span class="stat-lbl">fps</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span>
                        <span class="stat-lbl">error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span>
                        <span class="stat-lbl">steer</span></div>
      <div class="stat"><span class="stat-val" id="v-lane">&mdash;</span>
                        <span class="stat-lbl">lane</span></div>
      <div class="stat"><span class="stat-val" id="v-lidar">0</span>
                        <span class="stat-lbl">lidar front</span></div>
      <div class="stat"><span class="stat-val" id="v-lidar-left">0</span>
                        <span class="stat-lbl">lidar left</span></div>
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
    <h2>Lane Tracking Parameters</h2>
    <div class="slider-row">
      <label>ROI Top Frac</label>
      <input type="range" id="roi_top_frac" min="0.1" max="0.9" value="0.5" step="0.05">
      <span class="val" id="v-roi_top_frac">0.5</span>
    </div>
    <div class="slider-row">
      <label>ROI Side Limit</label>
      <input type="range" id="roi_side_limit" min="0.0" max="0.45" value="0.0" step="0.01">
      <span class="val" id="v-roi_side_limit">0.0</span>
    </div>
    <hr class="divider">
    <h2>Lidar Safety</h2>
    <div class="slider-row">
      <label>Stop Dist mm</label>
      <input type="range" id="stop_distance" min="100" max="2000" value="400" step="10">
      <span class="val" id="v-stop_distance">400</span>
    </div>
  </div>
</div>
<script>
const sliders = [
  "speed","kp","ki","kd",
  "canny_lo","canny_hi","binary_thresh","blur_ksize",
  "morph_ksize","morph_iters",
  "roi_top_frac","roi_side_limit",
  "stop_distance"
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

    const laneEl = document.getElementById("v-lane");
    laneEl.textContent = d.lane_found ? "✓" : "✗";
    laneEl.style.color = d.lane_found ? "#00d4aa" : "#ff4d4d";

    const lidarEl = document.getElementById("v-lidar");
    lidarEl.textContent = d.lidar_closest.toFixed(0);
    
    const lidarLeftEl = document.getElementById("v-lidar-left");
    lidarLeftEl.textContent = d.lidar_closest_left.toFixed(0);

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
                         "lidar_closest", "lidar_closest_left", "lidar_blocked", "autonomy_state")})


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
    car = JetRacer(init_lidar=True)
    car.arm(delay=3)

    # Lidar runs in its own thread — never blocks the camera loop
    lt = threading.Thread(target=lidar_loop, args=(car,), daemon=True)
    lt.start()

    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
