#!/usr/bin/env python3
"""
experiments.py — Multi-experiment Lane Follower & Overtaking Logic
Includes: lanefollow, laneswitch, emergencybrake, linefollow, overtake with/without recovery.
Optimised for Jetson Nano 4 GB.
"""

import cv2
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
    _gpu_hsv   = cv2.cuda_GpuMat()
else:
    print("[init] No CUDA device — falling back to CPU")
    _gpu_frame = _gpu_gray = _gpu_blur = _gpu_tmp = _gpu_hsv = None

# ── Config constants ──────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240
ENCODE_EVERY    = 3
JPEG_QUALITY    = 30
MJPEG_INTERVAL  = 1 / 15

# Pre-allocate morphological kernel
MORPH_KERNEL_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
    "stop_distance": 400.0,
    # IMU data
    "imu_ax": 0, "imu_ay": 0, "imu_az": 0,
    "imu_gx": 0, "imu_gy": 0, "imu_gz": 0,
    # Encoder data
    "enc_speed": 0.0, "enc_dist": 0.0,
    # Line follow params
    "h_lo": 0,  "h_hi": 180,
    "s_lo": 0,  "s_hi": 50,
    "v_lo": 200, "v_hi": 255,
    "min_contour_area": 300,
    # Experiment mode
    "experiment": "lanefollow",
    # Telemetry
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

# Latest JPEG bytes for MJPEG stream
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

# ── GPU helpers ──────────────────────────────────────────
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

def _make_gpu_hsv_mask():
    probe = cv2.cuda_GpuMat()
    probe.upload(np.zeros((1, 1, 3), dtype=np.uint8))
    try:
        result = cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2HSV)
        if result is not None and not result.empty():
            def _mask_functional(roi_bgr, lo, hi):
                _gpu_frame.upload(roi_bgr)
                hsv_gpu = cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV)
                hsv_cpu = hsv_gpu.download()
                return cv2.inRange(hsv_cpu, lo, hi)
            return _mask_functional
    except Exception:
        pass

    try:
        cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2HSV, _gpu_hsv)
        def _mask_inplace(roi_bgr, lo, hi):
            _gpu_frame.upload(roi_bgr)
            cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV, _gpu_hsv)
            hsv_cpu = _gpu_hsv.download()
            return cv2.inRange(hsv_cpu, lo, hi)
        return _mask_inplace
    except Exception:
        pass
    return None

gpu_hsv_mask = _make_gpu_hsv_mask() if USE_CUDA else None

def cpu_hsv_mask(roi_bgr, lo, hi):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lo, hi)


# ── Frame processing ────────────────────────────────────────────────
def process_line_frame(frame, s, annotate: bool):
    global _last_steer
    h, w = frame.shape[:2]
    ROI_FRAC = 0.65
    roi_top = int(h * ROI_FRAC)
    
    x_start = int(w * s.get("roi_side_limit", 0.0))
    x_end   = w - x_start
    roi = frame[roi_top:h, x_start:x_end]

    lo = np.array([s["h_lo"], s["s_lo"], s["v_lo"]], dtype=np.uint8)
    hi = np.array([s["h_hi"], s["s_hi"], s["v_hi"]], dtype=np.uint8)

    if gpu_hsv_mask is not None:
        mask = gpu_hsv_mask(roi, lo, hi)
    else:
        mask = cpu_hsv_mask(roi, lo, hi)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL_ELLIPSE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL_ELLIPSE)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = s["min_contour_area"]
    line_cx, line_cy = None, None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x_start
                cy = int(M["m01"] / M["m00"])
                if line_cx is None or cx < line_cx:
                    line_cx = cx
                    line_cy = cy

    line_found = line_cx is not None

    if line_found:
        offset = 140
        target = line_cx + offset
        target = max(60, min(w-60, target))
        error = (target - w / 2.0) / (w / 2.0) * 2

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
        steer = _last_steer

    if annotate:
        annotated = frame.copy()
        cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)
        cv2.line(annotated, (x_start, roi_top), (x_start, h), (255, 0, 255), 1)
        cv2.line(annotated, (x_end, roi_top), (x_end, h), (255, 0, 255), 1)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_3ch[:, :, 0] = 0
        annotated[roi_top:h, x_start:x_end] = cv2.addWeighted(
            annotated[roi_top:h, x_start:x_end], 0.7, mask_3ch, 0.3, 0)

        if line_found:
            cv2.circle(annotated, (line_cx, roi_top + line_cy), 8, (0, 255, 0), -1)
            cv2.circle(annotated, (line_cx, roi_top + line_cy), 8, (255, 255, 255), 2)
    else:
        annotated = frame

    return annotated, error, steer, line_found

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
            target_x = left_x + 140
        elif len(right_pixels) > 10:
            right_x = np.mean(right_pixels) + x_start
            target_x = right_x - 140
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
        steer = max(-1.0, min(1.0, steer))
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


# ── Async JPEG encode ─────────────────────────────────────────────────────────
def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return
    with frame_lock:
        global latest_frame
        latest_frame = jpeg.tobytes()


# ── Firebase Telemetry ────────────────────────────────────────────────────────
def firebase_loop():
    print("[firebase] Telemetry background thread started")
    batch = {}
    last_upload_time = time.time()
    
    while True:
        try:
            test_id, timestamp, data_point = telemetry_queue.get(timeout=0.5)
            if test_id not in batch:
                batch[test_id] = {}
            key = str(int(timestamp * 1000))
            batch[test_id][key] = data_point
        except queue.Empty:
            pass
            
        now = time.time()
        if now - last_upload_time >= 2.0:
            if batch:
                try:
                    for tid, b_data in batch.items():
                        url = f"{FIREBASE_URL}/experiments/{tid}.json"
                        requests.patch(url, json=b_data, timeout=5)
                    batch.clear()
                except Exception as e:
                    print(f"[firebase] upload error: {e}")
            last_upload_time = now

# ── Lidar background thread ───────────────────────────────────────────────────
_lidar_cache      = {"closest": 0.0, "closest_left": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()

def lidar_loop(car: JetRacer):
    print("[lidar] Background safety thread started")
    while True:
        try:
            with state_lock:
                STOP_DISTANCE = state["stop_distance"]

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

        time.sleep(0.05)


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
    last_lvel      = 0
    last_rvel      = 0
    last_time      = time.time()

    SPEED_SCALE = 0.00748

    while True:
        with state_lock:
            if state.get("reset_encoder_dist"):
                total_distance = 0.0
                state["reset_encoder_dist"] = False

        try:
            b = ser.read(1)
            if not b or b[0] != HEAD1:
                continue

            b = ser.read(1)
            if not b or b[0] != HEAD2:
                continue

            b = ser.read(1)
            if not b:
                continue
            frame_size = b[0]

            if frame_size < 5 or frame_size > 50:
                continue

            remaining = frame_size - 3
            rest = ser.read(remaining)
            if len(rest) != remaining:
                continue

            frame = bytes([HEAD1, HEAD2, frame_size]) + rest

            calc_sum = sum(frame[:-1]) & 0xFF
            recv_sum = frame[-1]
            if calc_sum != recv_sum:
                continue

            gx = int.from_bytes(frame[4:6],   'big', signed=True) / 32768 * 2000
            gy = int.from_bytes(frame[6:8],   'big', signed=True) / 32768 * 2000
            gz = int.from_bytes(frame[8:10],  'big', signed=True) / 32768 * 2000
            ax = int.from_bytes(frame[10:12], 'big', signed=True) / 32768 * 2 * 9.8
            ay = int.from_bytes(frame[12:14], 'big', signed=True) / 32768 * 2 * 9.8
            az = int.from_bytes(frame[14:16], 'big', signed=True) / 32768 * 2 * 9.8

            with _imu_cache_lock:
                _imu_cache["ax"] = round(ax, 2)
                _imu_cache["ay"] = round(ay, 2)
                _imu_cache["az"] = round(az, 2)
                _imu_cache["gx"] = round(gx, 1)
                _imu_cache["gy"] = round(gy, 1)
                _imu_cache["gz"] = round(gz, 1)

            lvel = int.from_bytes(frame[34:36], 'big', signed=True)
            rvel = int.from_bytes(frame[36:38], 'big', signed=True)

            now = time.time()
            dt  = now - last_time

            if dt > 0:
                avg_vel   = (lvel + rvel) / 2.0
                speed_ms  = avg_vel * SPEED_SCALE
                d_dist    = speed_ms * dt
                total_distance += d_dist

                with _encoder_cache_lock:
                    _encoder_cache["speed"]    = round(speed_ms, 3)
                    _encoder_cache["distance"] = round(total_distance, 3)

            last_lvel = lvel
            last_rvel = rvel
            last_time = now

        except Exception as e:
            time.sleep(0.1)

def control_loop(car: JetRacer):
    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    frame_idx = 0
    
    yaw = 0.0
    pos_y = 0.0
    last_time = time.time()
    pid_traj = {"integral": 0.0, "last_error": 0.0}
    
    TRAJ_KP = 3.5
    TRAJ_KI = 0.1
    TRAJ_KD = 0.8
    LANE_WIDTH_ACTUAL = 0.28
    LANE_WIDTH = LANE_WIDTH_ACTUAL*0.4
    OVERTAKE_MANEUVER_DIST = 0.30
    MANEUVER_MAX_DIST = OVERTAKE_MANEUVER_DIST * 2.5
    RECOVERY_POS_TOLERANCE = 0.03
    
    print("[loop] Control loop started")

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
        experiment = s_copy.get("experiment", "lanefollow")

        if experiment == "linefollow":
            annotated, error, steer, line_found = process_line_frame(frame, s_copy, do_annotate)
            lane_found = line_found
            left_found = right_found = False
            lane_width = left_x = right_x = 0.0
        else:
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
            lidar_closest_left = _lidar_cache.get("closest_left", 0.0)
            lidar_blocked = _lidar_cache["blocked"]

        with _imu_cache_lock:
            imu_ax = _imu_cache["ax"]
            imu_ay = _imu_cache["ay"]
            imu_az = _imu_cache["az"]
            imu_gx = _imu_cache["gx"]
            imu_gy = _imu_cache["gy"]
            imu_gz = _imu_cache["gz"]

        with _encoder_cache_lock:
            enc_speed = _encoder_cache["speed"]
            enc_dist = _encoder_cache["distance"]

        autonomy_state = s_copy.get("autonomy_state", "FOLLOW")
        now = time.time()

        dt = max(now - last_time, 0.001)
        last_time = now
        
        yaw_rate_rad = math.radians(imu_gz)
        yaw += yaw_rate_rad * dt
        vy = enc_speed * math.sin(yaw)
        pos_y += vy * dt

        if s_copy["enabled"]:
            if not pid_state.get("was_enabled", False):
                pid_state["start_exp_dist"] = enc_dist
                pid_state["was_enabled"] = True

            if autonomy_state == "FOLLOW":
                if experiment == "laneswitch":
                    if (enc_dist - pid_state.get("start_exp_dist", enc_dist)) > 0.5 and not pid_state.get("is_post_overtake", False):
                        autonomy_state = "OVERTAKING"
                        yaw = 0.0; pos_y = 0.0; pid_traj["integral"] = 0.0; pid_traj["last_error"] = 0.0
                        pid_state["start_enc_dist"] = enc_dist; pid_state["start_pos_y"] = 0.0
                        pid_state["lane_change_dist"] = OVERTAKE_MANEUVER_DIST
                        print("[STATE] -> OVERTAKING (laneswitch after 0.5m)", flush=True)
                
                elif experiment in ["overtake_no_recovery", "overtake_with_recovery", "emergencybrake"]:
                    if lidar_blocked and not pid_state.get("is_post_overtake", False):
                        if experiment != "emergencybrake":
                            autonomy_state = "OVERTAKING"
                            yaw = 0.0; pos_y = 0.0; pid_traj["integral"] = 0.0; pid_traj["last_error"] = 0.0
                            pid_state["start_enc_dist"] = enc_dist; pid_state["start_pos_y"] = 0.0
                            pid_state["lane_change_dist"] = OVERTAKE_MANEUVER_DIST
                            print(f"[STATE] -> OVERTAKING. Obstacle at {lidar_closest}mm", flush=True)

                if autonomy_state == "FOLLOW":
                    if pid_state.get("is_post_overtake", False) and experiment == "overtake_with_recovery":
                        s_follow = enc_dist - pid_state.get("post_overtake_enc_dist", enc_dist)
                        if s_follow >= 0.2 and lidar_closest_left > 400.0:
                            print("[STATE] -> RECOVERY (left lane clear)", flush=True)
                            autonomy_state = "RECOVERY"
                            pid_state["is_post_overtake"] = False
                            yaw = 0.0; pos_y = 0.0; pid_traj["integral"] = 0.0; pid_traj["last_error"] = 0.0
                            pid_state["start_enc_dist"] = enc_dist; pid_state["start_pos_y"] = 0.0
                            pid_state["lane_change_dist"] = OVERTAKE_MANEUVER_DIST
                    
                    if experiment == "emergencybrake" and lidar_closest > 0 and lidar_closest < 600.0:
                        car.stop()
                    else:
                        car.steer(steer)
                        car.forward(s_copy["speed"])

            elif autonomy_state == "OVERTAKING":
                s = enc_dist - pid_state.get("start_enc_dist", enc_dist)
                D = pid_state.get("lane_change_dist", OVERTAKE_MANEUVER_DIST)

                if s < D:
                    s_ratio = s / D
                    poly = 10*(s_ratio)**3 - 15*(s_ratio)**4 + 6*(s_ratio)**5
                    kick = (s_ratio ** 0.5) * (1.0 - s_ratio)
                    poly = poly + 0.4 * kick
                    target_y = pid_state.get("start_pos_y", 0.0) + LANE_WIDTH * poly
                else:
                    target_y = pid_state.get("start_pos_y", 0.0) + LANE_WIDTH
                    
                    if experiment == "laneswitch":
                        if s >= MANEUVER_MAX_DIST or s >= D:
                            print(f"[STATE] -> FOLLOW (laneswitch done)", flush=True)
                            autonomy_state = "FOLLOW"
                            pid_state["is_post_overtake"] = True
                            yaw = 0.0; pos_y = 0.0
                    else:
                        if (not lidar_blocked) or (s >= MANEUVER_MAX_DIST):
                            print(f"[STATE] -> FOLLOW (overtake done)", flush=True)
                            autonomy_state = "FOLLOW"
                            pid_state["is_post_overtake"] = True
                            pid_state["post_overtake_enc_dist"] = enc_dist
                            yaw = 0.0; pos_y = 0.0

                traj_error = target_y - pos_y

                pid_traj["integral"] += traj_error * dt
                pid_traj["integral"] = max(-1.0, min(1.0, pid_traj["integral"]))
                derivative = (traj_error - pid_traj["last_error"]) / dt
                pid_traj["last_error"] = traj_error

                steer_cmd = TRAJ_KP * traj_error + TRAJ_KI * pid_traj["integral"] + TRAJ_KD * derivative
                steer_cmd = max(-1.0, min(1.0, steer_cmd))

                car.steer(steer_cmd)
                car.forward(s_copy["speed"])
                steer = steer_cmd

            elif autonomy_state == "RECOVERY":
                s = enc_dist - pid_state.get("start_enc_dist", enc_dist)
                D = pid_state.get("lane_change_dist", OVERTAKE_MANEUVER_DIST)

                if s < D:
                    s_ratio = s / D
                    poly = 10*(s_ratio)**3 - 15*(s_ratio)**4 + 6*(s_ratio)**5
                    kick = (s_ratio ** 0.5) * (1.0 - s_ratio)
                    poly = poly + 0.4 * kick
                    target_y = pid_state.get("start_pos_y", 0.0) - LANE_WIDTH * poly
                else:
                    target_y = pid_state.get("start_pos_y", 0.0) - LANE_WIDTH

                traj_error = target_y - pos_y

                if s >= D and (abs(traj_error) < RECOVERY_POS_TOLERANCE or s >= MANEUVER_MAX_DIST):
                    print("[STATE] -> FOLLOW (recovery complete)", flush=True)
                    autonomy_state = "FOLLOW"
                    yaw = 0.0; pos_y = 0.0

                pid_traj["integral"] += traj_error * dt
                pid_traj["integral"] = max(-1.0, min(1.0, pid_traj["integral"]))
                derivative = (traj_error - pid_traj["last_error"]) / dt
                pid_traj["last_error"] = traj_error

                steer_cmd = TRAJ_KP * traj_error + TRAJ_KI * pid_traj["integral"] + TRAJ_KD * derivative
                steer_cmd = max(-1.0, min(1.0, steer_cmd))

                car.steer(steer_cmd)
                car.forward(s_copy["speed"])
                steer = steer_cmd

        else:
            car.stop()
            autonomy_state = "FOLLOW"
            yaw = 0.0; pos_y = 0.0
            pid_state["is_post_overtake"] = False
            pid_state["was_enabled"] = False

        with state_lock:
            state["error"]         = round(error, 3)
            state["steer"]         = round(steer, 3)
            state["lane_found"]    = lane_found
            state["lidar_closest"] = lidar_closest
            state["lidar_closest_left"] = lidar_closest_left
            state["lidar_blocked"] = lidar_blocked
            state["autonomy_state"] = autonomy_state
            state["imu_ax"] = imu_ax; state["imu_ay"] = imu_ay; state["imu_az"] = imu_az
            state["imu_gx"] = imu_gx; state["imu_gy"] = imu_gy; state["imu_gz"] = imu_gz
            state["enc_speed"] = enc_speed
            state["enc_dist"] = enc_dist

        if s_copy.get("is_testing", False) and s_copy.get("test_id"):
            current_time = time.time()
            data_point = {
                "timestamp": current_time,
                "error": round(error, 4) if 'error' in locals() else 0.0,
                "steer": round(steer, 4) if 'steer' in locals() else 0.0,
                "left_found": left_found,
                "right_found": right_found,
                "lane_width": round(lane_width, 2) if 'lane_width' in locals() else 0.0,
                "lidar_closest": lidar_closest,
                "lidar_closest_left": lidar_closest_left,
                "lidar_blocked": lidar_blocked,
                "imu_ax": imu_ax, "imu_ay": imu_ay, "imu_az": imu_az,
                "imu_gx": imu_gx, "imu_gy": imu_gy, "imu_gz": imu_gz,
                "enc_speed": enc_speed, "enc_dist": enc_dist,
                "autonomy_state": autonomy_state,
                "experiment": experiment,
                "integral": round(pid_state.get("integral", 0.0), 4)
            }
            for k, v in s_copy.items():
                if k not in ["is_testing", "test_id"]:
                    if isinstance(v, float):
                        data_point[k] = round(v, 4)
                    else:
                        data_point[k] = v
            telemetry_queue.put((s_copy["test_id"], current_time, data_point))

        frame_idx += 1

    cap.release()

# ── Flask / dashboard ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetRacer Experiments</title>
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
  .slider-row input[type=range], .slider-row select { flex: 1; accent-color: var(--accent); }
  .slider-row select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 4px; }
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
<h1>&#9675; JetRacer &#183; Experiments</h1>
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
      <div class="stat"><span class="stat-val" id="v-enc-spd">0.00</span>
                        <span class="stat-lbl">enc spd</span></div>
      <div class="stat"><span class="stat-val" id="v-enc-dist">0.00</span>
                        <span class="stat-lbl">enc dist</span></div>
      <div class="stat"><span class="stat-val" id="v-test">--</span>
                        <span class="stat-lbl">test id</span></div>
    </div>
    <div class="error-track" title="Lane error (centre = 0)">
      <div id="error-bar"></div>
    </div>
  </div>
  <div class="card">
    <h2>Experiment Mode</h2>
    <div class="slider-row">
      <label>Mode</label>
      <select id="experiment">
        <option value="lanefollow">Lane Follow</option>
        <option value="laneswitch">Lane Switch (after 0.5m)</option>
        <option value="emergencybrake">Emergency Brake (<600mm)</option>
        <option value="linefollow">Line Follow (HSV)</option>
        <option value="overtake_no_recovery">Overtake No Recovery</option>
        <option value="overtake_with_recovery">Overtake With Recovery</option>
      </select>
    </div>
    <div class="btn-row">
      <button id="btn-go"   onclick="setEnabled(true)">&#9654; GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
      <button id="btn-test" onclick="toggleTest()" style="background: #6b7a99; color: #fff;">&#9654; START TEST</button>
    </div>
    <div class="slider-row" style="margin-top: .75rem;">
      <label>Speed</label>
      <input type="range" id="speed" min="0" max="60" value="15" step="1">
      <span class="val" id="v-speed">0.15</span>
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
    <h2>Line Tracking (HSV)</h2>
    <div class="slider-row">
      <label>Min area</label>
      <input type="range" id="min_contour_area" min="50" max="5000" value="300" step="50">
      <span class="val" id="v-min_contour_area">300</span>
    </div>
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
  "stop_distance",
  "h_lo","h_hi","s_lo","s_hi","v_lo","v_hi","min_contour_area"
];

let isTesting = false;
function toggleTest() {
  isTesting = !isTesting;
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({is_testing: isTesting})});
}

document.getElementById("experiment").addEventListener("change", (e) => {
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({experiment: e.target.value})});
});

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

    const encSpdEl = document.getElementById("v-enc-spd");
    if(encSpdEl) encSpdEl.textContent = d.enc_speed.toFixed(2) + " m/s";
    const encDistEl = document.getElementById("v-enc-dist");
    if(encDistEl) encDistEl.textContent = d.enc_dist.toFixed(2) + " m";

    const testEl = document.getElementById("v-test");
    if(testEl) testEl.textContent = d.is_testing ? d.test_id : "--";
    isTesting = d.is_testing;
    const btn = document.getElementById("btn-test");
    if(btn) {
      if(isTesting) {
        btn.innerHTML = "&#9632; END TEST";
        btn.style.background = "#ffb020";
      } else {
        btn.innerHTML = "&#9654; START TEST";
        btn.style.background = "#6b7a99";
      }
    }

    const expSelect = document.getElementById("experiment");
    if (d.experiment && expSelect.value !== d.experiment && document.activeElement !== expSelect) {
      expSelect.value = d.experiment;
    }

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
                         "lidar_closest", "lidar_closest_left", "lidar_blocked", "autonomy_state", "is_testing", "test_id", "imu_ax", "imu_ay", "imu_az", "imu_gx", "imu_gy", "imu_gz", "enc_speed", "enc_dist", "experiment")})


@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k == "is_testing":
                if v and not state.get("is_testing", False):
                    state["test_id"] = f"test_{int(time.time())}"
                    state["is_testing"] = True
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = JetRacer(init_lidar=True)
    car.arm(delay=3)

    lt = threading.Thread(target=lidar_loop, args=(car,), daemon=True)
    lt.start()

    ft = threading.Thread(target=firebase_loop, daemon=True)
    ft.start()

    st = threading.Thread(target=sensor_loop, daemon=True)
    st.start()

    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
