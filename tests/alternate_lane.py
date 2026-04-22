#!/usr/bin/env python3
"""
lane_following_jetson.py
========================
CUDA-GPU-accelerated lane following for NVIDIA Jetson platforms.
Pipeline: frame → grayscale → Gaussian blur → (Canny + Binary) → OR combine
          → morphology clean → bird-eye → histogram → lane center → PID → motor

Flask server runs on port 5000 for real-time parameter tuning.

Lane type: black road bordered by white lines.

Author: generated for Jetson Nano/Xavier/Orin
Dependencies:
    pip install flask opencv-python-headless numpy cupy-cuda12x
    (use cupy-cuda11x if CUDA 11.x, or cupy-cuda12x for CUDA 12.x)
    Motor control: adapt _send_motor_command() to your HAL (Adafruit, GPIO, ROS, etc.)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import threading
import time
import logging
import sys
import signal
import math
import os

from flask import Flask, jsonify, request, render_template_string
from collections import deque
from jetracer import JetRacer

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

# CuPy for CUDA-side ndarray ops (falls back to NumPy if unavailable)
try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:
    cp = np          # transparent fallback
    _CUPY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("LanePilot")

# ─────────────────────────────────────────────────────────────────────────────
# ── Shared mutable parameter store (thread-safe with a lock) ─────────────────
# ─────────────────────────────────────────────────────────────────────────────
_param_lock = threading.Lock()

PARAMS = {
    # ── Camera ────────────────────────────────────────────────────────────────
    "camera_index":         0,
    "frame_width":          640,
    "frame_height":         480,
    "fps_cap":              30,

    # ── Gaussian blur ─────────────────────────────────────────────────────────
    "blur_ksize":           5,       # must be odd

    # ── Canny ─────────────────────────────────────────────────────────────────
    "canny_lo":             50,
    "canny_hi":             150,

    # ── Binary threshold (for white-line detection) ────────────────────────────
    "binary_thresh":        200,     # 0-255
    "binary_maxval":        255,

    # ── Morphology (post-OR combine) ──────────────────────────────────────────
    "morph_ksize":          5,
    "morph_iterations":     2,

    # ── Bird-eye ROI quad (fractions of frame w/h) ────────────────────────────
    # top-left, top-right, bot-right, bot-left  (y=0 = top)
    "bev_tl_x": 0.40, "bev_tl_y": 0.55,
    "bev_tr_x": 0.60, "bev_tr_y": 0.55,
    "bev_br_x": 0.90, "bev_br_y": 0.90,
    "bev_bl_x": 0.10, "bev_bl_y": 0.90,

    # ── Histogram / lane detection ────────────────────────────────────────────
    "hist_row_start":       0.5,     # fraction of BEV height to start summing
    "lane_search_margin":   80,      # px around prior base
    "lane_nwindows":        9,
    "lane_minpix":          30,

    # ── PID ───────────────────────────────────────────────────────────────────
    "pid_kp":               0.55,
    "pid_ki":               0.002,
    "pid_kd":               0.18,
    "pid_integral_clamp":   300.0,
    "pid_output_clamp":     1.0,     # normalised steering ∈ [-1, 1]

    # ── Motor / drive ─────────────────────────────────────────────────────────
    "base_speed":           0.35,    # normalised [0, 1]
    "speed_reduction_turn": 0.15,    # reduce speed by this fraction when steering

    # ── Fail-safe ─────────────────────────────────────────────────────────────
    "failsafe_no_lane_frames": 10,   # consecutive frames with no lane → stop
    "failsafe_max_error":      300,  # |error| > this → emergency stop

    # ── Debug ─────────────────────────────────────────────────────────────────
    "show_debug_window":    False,
    "log_telemetry":        True,
}


def get(key):
    with _param_lock:
        return PARAMS[key]


def set_param(key, value):
    with _param_lock:
        if key in PARAMS:
            PARAMS[key] = type(PARAMS[key])(value)
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ── CUDA helpers ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _to_gpu(mat: np.ndarray):
    """Upload numpy array to CuPy array (no-op if CuPy absent)."""
    if _CUPY_OK:
        return cp.asarray(mat)
    return mat


def _to_cpu(mat) -> np.ndarray:
    """Download CuPy array to numpy (no-op if CuPy absent)."""
    if _CUPY_OK and isinstance(mat, cp.ndarray):
        return cp.asnumpy(mat)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# ── OpenCV CUDA stream wrapper ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class CUDAStream:
    """Thin wrapper: uses cv2.cuda if available, else plain cv2."""

    HAS_CV_CUDA = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

    def __init__(self):
        if self.HAS_CV_CUDA:
            log.info("OpenCV CUDA enabled – GPU sync mode (no explicit stream).")
        else:
            log.warning("OpenCV CUDA NOT available – falling back to CPU cv2.")

    # ── Upload / Download ────────────────────────────────────────────────────
    def upload(self, mat: np.ndarray):
        if self.HAS_CV_CUDA:
            g = cv2.cuda_GpuMat()
            g.upload(mat)
            return g
        return mat

    def download(self, g) -> np.ndarray:
        if self.HAS_CV_CUDA and isinstance(g, cv2.cuda_GpuMat):
            return g.download()
        return g

    # ── Per-op wrappers ──────────────────────────────────────────────────────
    def cvtColor(self, g, code):
        if self.HAS_CV_CUDA:
            dst = cv2.cuda_GpuMat()
            cv2.cuda.cvtColor(g, code, dst=dst)
            return dst
        return cv2.cvtColor(g, code)

    def gaussianBlur(self, g, ksize, sigma):
        if self.HAS_CV_CUDA:
            filt = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (ksize, ksize), sigma
            )
            dst = cv2.cuda_GpuMat()
            filt.apply(g, dst)
            return dst
        return cv2.GaussianBlur(g, (ksize, ksize), sigma)

    def canny(self, g, lo, hi):
        if self.HAS_CV_CUDA:
            det = cv2.cuda.createCannyEdgeDetector(lo, hi)
            dst = cv2.cuda_GpuMat()
            det.detect(g, dst)
            return dst
        return cv2.Canny(g, lo, hi)

    def threshold(self, g, thresh, maxval):
        if self.HAS_CV_CUDA:
            dst = cv2.cuda_GpuMat()
            cv2.cuda.threshold(g, thresh, maxval, cv2.THRESH_BINARY, dst=dst)
            return dst
        _, out = cv2.threshold(g, thresh, maxval, cv2.THRESH_BINARY)
        return out

    def bitwise_or(self, a, b):
        if self.HAS_CV_CUDA:
            dst = cv2.cuda_GpuMat()
            cv2.cuda.bitwise_or(a, b, dst=dst)
            return dst
        return cv2.bitwise_or(a, b)

    def morphologyEx(self, g, op, kernel, iters):
        """Morphology runs on CPU (cv2.cuda morphology is limited)."""
        cpu = self.download(g)
        out = cv2.morphologyEx(cpu, op, kernel, iterations=iters)
        return self.upload(out)

    def warpPerspective(self, g, M, dsize):
        if self.HAS_CV_CUDA:
            dst = cv2.cuda_GpuMat()
            cv2.cuda.warpPerspective(g, M, dsize, dst=dst)
            return dst
        return cv2.warpPerspective(g, M, dsize)

    def sync(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ── PID Controller ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class PIDController:
    def __init__(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.monotonic()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.monotonic()

    def compute(self, error: float) -> float:
        now = time.monotonic()
        dt = max(now - self._prev_time, 1e-4)
        self._prev_time = now

        kp = get("pid_kp")
        ki = get("pid_ki")
        kd = get("pid_kd")
        i_clamp = get("pid_integral_clamp")
        o_clamp = get("pid_output_clamp")

        self._integral += error * dt
        self._integral = max(-i_clamp, min(i_clamp, self._integral))

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = kp * error + ki * self._integral + kd * derivative
        return max(-o_clamp, min(o_clamp, output))


# ─────────────────────────────────────────────────────────────────────────────
# ── Motor interface (STUB – adapt to your hardware) ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
car = None

def _send_motor_command(speed: float, steering: float):
    """
    Send drive command to motor controller.
    speed    ∈ [0, 1]   (forward)
    steering ∈ [-1, 1]  (negative = left, positive = right)
    """
    if car is not None:
        car.steer(steering)
        car.forward(speed)


def _emergency_stop():
    if car is not None:
        car.stop()
    log.warning("⛔ Emergency stop triggered.")


# ─────────────────────────────────────────────────────────────────────────────
# ── Bird-eye perspective transform (cached, rebuilt on param change) ──────────
# ─────────────────────────────────────────────────────────────────────────────
class BirdEye:
    def __init__(self, stream: CUDAStream):
        self._stream = stream
        self._M = None
        self._Minv = None
        self._last_params = None
        self._dsize = None

    def _build(self, W, H):
        p = {
            k: get(k) for k in (
                "bev_tl_x", "bev_tl_y", "bev_tr_x", "bev_tr_y",
                "bev_br_x", "bev_br_y", "bev_bl_x", "bev_bl_y",
            )
        }
        if p == self._last_params:
            return

        src = np.float32([
            [p["bev_tl_x"] * W, p["bev_tl_y"] * H],
            [p["bev_tr_x"] * W, p["bev_tr_y"] * H],
            [p["bev_br_x"] * W, p["bev_br_y"] * H],
            [p["bev_bl_x"] * W, p["bev_bl_y"] * H],
        ])
        dst = np.float32([
            [0,   0],
            [W,   0],
            [W,   H],
            [0,   H],
        ])
        self._M    = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)
        self._dsize = (W, H)
        self._last_params = p

    def warp(self, g, W, H):
        self._build(W, H)
        return self._stream.warpPerspective(g, self._M, self._dsize)

    @property
    def Minv(self):
        return self._Minv


# ─────────────────────────────────────────────────────────────────────────────
# ── Lane detector (histogram + sliding windows) ───────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class LaneDetector:
    def __init__(self):
        self._left_base_prev  = None
        self._right_base_prev = None

    def detect(self, bev_binary: np.ndarray):
        """
        bev_binary: grayscale/binary BEV image (CPU ndarray, 0/255).
        Returns: (center_x, left_fit, right_fit, debug_img)
                 center_x = None if detection fails.
        """
        H, W = bev_binary.shape
        row_start = int(get("hist_row_start") * H)
        margin    = int(get("lane_search_margin"))
        nwindows  = int(get("lane_nwindows"))
        minpix    = int(get("lane_minpix"))

        # ── Histogram on lower half ───────────────────────────────────────────
        hist = np.sum(bev_binary[row_start:, :].astype(np.int32), axis=0)
        midpoint = W // 2

        # Left / right peaks
        left_base  = int(np.argmax(hist[:midpoint]))
        right_base = int(np.argmax(hist[midpoint:]) + midpoint)

        # Smooth with previous frame (simple IIR)
        alpha = 0.6
        if self._left_base_prev is not None:
            left_base  = int(alpha * left_base  + (1 - alpha) * self._left_base_prev)
            right_base = int(alpha * right_base + (1 - alpha) * self._right_base_prev)
        self._left_base_prev  = left_base
        self._right_base_prev = right_base

        # ── Sliding windows ───────────────────────────────────────────────────
        win_h = H // nwindows
        nonzero   = bev_binary.nonzero()
        nzy, nzx  = np.array(nonzero[0]), np.array(nonzero[1])

        left_cur, right_cur = left_base, right_base
        left_lane_idx, right_lane_idx = [], []

        dbg = cv2.cvtColor(bev_binary, cv2.COLOR_GRAY2BGR)

        for w in range(nwindows):
            y_lo = H - (w + 1) * win_h
            y_hi = H - w * win_h
            xl_lo, xl_hi = left_cur  - margin, left_cur  + margin
            xr_lo, xr_hi = right_cur - margin, right_cur + margin

            cv2.rectangle(dbg, (xl_lo, y_lo), (xl_hi, y_hi), (0, 255, 0), 1)
            cv2.rectangle(dbg, (xr_lo, y_lo), (xr_hi, y_hi), (0, 0, 255), 1)

            good_l = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xl_lo) & (nzx < xl_hi)).nonzero()[0]
            good_r = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xr_lo) & (nzx < xr_hi)).nonzero()[0]

            left_lane_idx.append(good_l)
            right_lane_idx.append(good_r)

            if len(good_l) > minpix:
                left_cur  = int(np.mean(nzx[good_l]))
            if len(good_r) > minpix:
                right_cur = int(np.mean(nzx[good_r]))

        left_idx  = np.concatenate(left_lane_idx)
        right_idx = np.concatenate(right_lane_idx)

        if len(left_idx) < 5 or len(right_idx) < 5:
            return None, None, None, dbg   # detection failed

        # ── Polynomial fit (2nd order) ────────────────────────────────────────
        ly, lx = nzy[left_idx],  nzx[left_idx]
        ry, rx = nzy[right_idx], nzx[right_idx]

        try:
            left_fit  = np.polyfit(ly, lx, 2)
            right_fit = np.polyfit(ry, rx, 2)
        except np.linalg.LinAlgError:
            return None, None, None, dbg

        # ── Lane centre at bottom ─────────────────────────────────────────────
        ploty = H - 1
        left_x  = np.polyval(left_fit,  ploty)
        right_x = np.polyval(right_fit, ploty)
        center_x = (left_x + right_x) / 2.0

        # Draw fitted lines
        ys = np.linspace(0, H - 1, H)
        lxs = np.polyval(left_fit,  ys).astype(int)
        rxs = np.polyval(right_fit, ys).astype(int)
        for y, lx_, rx_ in zip(ys.astype(int), lxs, rxs):
            if 0 <= lx_ < W:
                cv2.circle(dbg, (lx_, y), 1, (255, 200, 0), -1)
            if 0 <= rx_ < W:
                cv2.circle(dbg, (rx_, y), 1, (0, 200, 255), -1)

        return center_x, left_fit, right_fit, dbg


# ─────────────────────────────────────────────────────────────────────────────
# ── Main pipeline ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class LanePilot:
    def __init__(self):
        self._stream   = CUDAStream()
        self._bev      = BirdEye(self._stream)
        self._detector = LaneDetector()
        self._pid      = PIDController()
        self._running  = False

        self._no_lane_count = 0
        self._morph_kernel  = None
        self._morph_ksize   = None

        # Telemetry ring-buffer (for Flask dashboard)
        self._telem_lock = threading.Lock()
        self._telem = deque(maxlen=200)

    # ── Morphology kernel (cached) ────────────────────────────────────────────
    def _get_morph_kernel(self):
        ksize = get("morph_ksize")
        if ksize != self._morph_ksize:
            self._morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (ksize, ksize)
            )
            self._morph_ksize = ksize
        return self._morph_kernel

    # ── Single frame processing ───────────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray):
        H, W = frame.shape[:2]

        # 1. Upload to GPU
        g_bgr = self._stream.upload(frame)

        # 2. Grayscale
        g_gray = self._stream.cvtColor(g_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Gaussian blur
        ksize = get("blur_ksize") | 1    # ensure odd
        g_blur = self._stream.gaussianBlur(g_gray, ksize, 0)

        # 4a. Canny edge
        g_canny = self._stream.canny(g_blur, get("canny_lo"), get("canny_hi"))

        # 4b. Binary threshold (white lines)
        g_bin = self._stream.threshold(g_blur, get("binary_thresh"), get("binary_maxval"))

        # 5. OR combine
        g_combined = self._stream.bitwise_or(g_canny, g_bin)

        # 6. Morphology clean
        g_morph = self._stream.morphologyEx(
            g_combined, cv2.MORPH_CLOSE,
            self._get_morph_kernel(),
            get("morph_iterations")
        )

        # 7. Bird-eye
        g_bev = self._bev.warp(g_morph, W, H)

        # 8. Sync & download for CPU lane algo
        self._stream.sync()
        bev_cpu = self._stream.download(g_bev)

        # 9. Histogram + lane detection
        center_x, left_fit, right_fit, dbg = self._detector.detect(bev_cpu)

        # 10. PID
        if center_x is None:
            self._no_lane_count += 1
            if self._no_lane_count >= get("failsafe_no_lane_frames"):
                _emergency_stop()
            error   = 0.0
            steering = 0.0
        else:
            self._no_lane_count = 0
            frame_cx = W / 2.0
            error    = center_x - frame_cx

            if abs(error) > get("failsafe_max_error"):
                _emergency_stop()
                return None, dbg

            steering = self._pid.compute(error)

        # 11. Motor command
        speed = get("base_speed")
        _send_motor_command(speed, steering)

        # 12. Telemetry
        t = {
            "ts":        time.time(),
            "error":     round(float(error), 2) if center_x is not None else None,
            "steering":  round(float(steering), 4),
            "speed":     round(float(speed), 4),
            "no_lane":   self._no_lane_count,
            "center_x":  round(float(center_x), 1) if center_x is not None else None,
        }
        with self._telem_lock:
            self._telem.append(t)

        if get("log_telemetry"):
            log.debug(
                f"err={t['error']} steer={t['steering']:.3f} cx={t['center_x']}"
            )

        return t, dbg

    # ── Run loop ──────────────────────────────────────────────────────────────
    def run(self):
        w = get("frame_width")
        h = get("frame_height")
        fps = get("fps_cap")
        
        gst = (
            f"nvarguscamerasrc sensor-id={get('camera_index')} ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int)1280, height=(int)720, "
            f"framerate=(fraction)60/1, format=(string)NV12 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){w}, height=(int){h}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink drop=1 max-buffers=1"
        )
        log.info(f"Trying GStreamer pipeline:\n  {gst}")
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            log.warning("GStreamer pipeline failed, trying USB fallback...")
            cap = cv2.VideoCapture(get("camera_index"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS,          fps)

        if not cap.isOpened():
            log.error("Cannot open camera.")
            return

        self._running = True
        log.info("Lane pilot started.")
        fps_cap = get("fps_cap")

        try:
            while self._running:
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    log.warning("Camera read failed.")
                    time.sleep(0.1)
                    continue

                telem, dbg = self._process_frame(frame)

                if get("show_debug_window") and dbg is not None:
                    cv2.imshow("BEV Debug", dbg)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Frame rate cap
                elapsed = time.monotonic() - t0
                sleep_t = (1.0 / fps_cap) - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            self._running = False
            _emergency_stop()
            cap.release()
            cv2.destroyAllWindows()
            log.info("Lane pilot stopped.")

    def stop(self):
        self._running = False

    def get_telemetry(self):
        with self._telem_lock:
            return list(self._telem)


# ─────────────────────────────────────────────────────────────────────────────
# ── Flask tuning server ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>LanePilot · Jetson Tuner</title>
<style>
  :root{
    --bg:#0a0c10;--card:#12151c;--border:#1e2430;
    --accent:#00e5ff;--warn:#ff6b35;--ok:#00e676;
    --text:#c9d1e0;--dim:#556;
    --font:'JetBrains Mono',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;min-height:100vh}
  header{
    background:linear-gradient(90deg,#001a26,#000c14);
    border-bottom:1px solid var(--border);
    padding:14px 24px;display:flex;align-items:center;gap:14px;
  }
  header .logo{font-size:20px;font-weight:700;color:var(--accent);letter-spacing:2px}
  header .sub{color:var(--dim);font-size:11px}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 8px var(--ok);animation:pulse 1.8s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .grid{display:grid;grid-template-columns:300px 1fr;gap:1px;background:var(--border);min-height:calc(100vh - 53px)}
  .panel{background:var(--card);padding:20px;overflow-y:auto}
  h2{color:var(--accent);font-size:11px;letter-spacing:3px;text-transform:uppercase;margin-bottom:16px;border-bottom:1px solid var(--border);padding-bottom:8px}
  .group{margin-bottom:22px}
  .group-label{color:var(--dim);font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}
  .row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
  .row label{color:var(--text);flex:1;font-size:12px}
  .row input[type=range]{flex:1.2;accent-color:var(--accent)}
  .row .val{width:60px;text-align:right;color:var(--accent);font-size:12px}
  .btn{
    background:transparent;border:1px solid var(--accent);color:var(--accent);
    padding:7px 16px;border-radius:3px;cursor:pointer;font-family:var(--font);
    font-size:12px;letter-spacing:1px;transition:.2s;
  }
  .btn:hover{background:var(--accent);color:#000}
  .btn.stop{border-color:var(--warn);color:var(--warn)}
  .btn.stop:hover{background:var(--warn);color:#000}
  .telem-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:20px}
  .card{background:#0d1018;border:1px solid var(--border);border-radius:4px;padding:14px}
  .card .lbl{color:var(--dim);font-size:10px;letter-spacing:2px;text-transform:uppercase}
  .card .big{font-size:26px;font-weight:700;color:var(--accent);margin-top:4px}
  .card .big.warn{color:var(--warn)}
  canvas{width:100%;height:140px;display:block;margin-top:10px;border:1px solid var(--border);border-radius:3px}
  #status{font-size:11px;color:var(--dim);margin-top:14px;min-height:16px}
  @media(max-width:700px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div class="dot"></div>
  <div>
    <div class="logo">LANEPILOT</div>
    <div class="sub">Jetson CUDA lane follower · real-time tuner</div>
  </div>
</header>

<div class="grid">
<!-- ── LEFT: PARAMETER PANEL ──────────────────────────── -->
<div class="panel">
  <h2>Parameters</h2>

  <div class="group">
    <div class="group-label">Vision</div>
    <div class="row"><label>Blur kernel</label><input type="range" id="blur_ksize" min="1" max="21" step="2" value="5"><span class="val" id="v_blur_ksize">5</span></div>
    <div class="row"><label>Canny lo</label><input type="range" id="canny_lo" min="0" max="255" value="50"><span class="val" id="v_canny_lo">50</span></div>
    <div class="row"><label>Canny hi</label><input type="range" id="canny_hi" min="0" max="255" value="150"><span class="val" id="v_canny_hi">150</span></div>
    <div class="row"><label>Binary threshold</label><input type="range" id="binary_thresh" min="0" max="255" value="200"><span class="val" id="v_binary_thresh">200</span></div>
    <div class="row"><label>Morph kernel</label><input type="range" id="morph_ksize" min="1" max="21" step="2" value="5"><span class="val" id="v_morph_ksize">5</span></div>
    <div class="row"><label>Morph iterations</label><input type="range" id="morph_iterations" min="1" max="5" value="2"><span class="val" id="v_morph_iterations">2</span></div>
  </div>

  <div class="group">
    <div class="group-label">PID</div>
    <div class="row"><label>Kp</label><input type="range" id="pid_kp" min="0" max="2" step="0.01" value="0.55"><span class="val" id="v_pid_kp">0.55</span></div>
    <div class="row"><label>Ki</label><input type="range" id="pid_ki" min="0" max="0.1" step="0.001" value="0.002"><span class="val" id="v_pid_ki">0.002</span></div>
    <div class="row"><label>Kd</label><input type="range" id="pid_kd" min="0" max="1" step="0.01" value="0.18"><span class="val" id="v_pid_kd">0.18</span></div>
  </div>

  <div class="group">
    <div class="group-label">Drive</div>
    <div class="row"><label>Base speed</label><input type="range" id="base_speed" min="0" max="1" step="0.01" value="0.35"><span class="val" id="v_base_speed">0.35</span></div>
    <div class="row"><label>Turn reduction</label><input type="range" id="speed_reduction_turn" min="0" max="0.5" step="0.01" value="0.15"><span class="val" id="v_speed_reduction_turn">0.15</span></div>
  </div>

  <div class="group">
    <div class="group-label">Fail-safe</div>
    <div class="row"><label>No-lane frames</label><input type="range" id="failsafe_no_lane_frames" min="1" max="60" value="10"><span class="val" id="v_failsafe_no_lane_frames">10</span></div>
    <div class="row"><label>Max error px</label><input type="range" id="failsafe_max_error" min="50" max="640" value="300"><span class="val" id="v_failsafe_max_error">300</span></div>
  </div>

  <button class="btn stop" onclick="eStop()">⛔ Emergency Stop</button>
  <div id="status"></div>
</div>

<!-- ── RIGHT: TELEMETRY PANEL ─────────────────────────── -->
<div class="panel">
  <h2>Live Telemetry</h2>
  <div class="telem-grid">
    <div class="card"><div class="lbl">Error (px)</div><div class="big" id="t_error">—</div></div>
    <div class="card"><div class="lbl">Steering</div><div class="big" id="t_steer">—</div></div>
    <div class="card"><div class="lbl">Speed</div><div class="big" id="t_speed">—</div></div>
    <div class="card"><div class="lbl">Center X</div><div class="big" id="t_cx">—</div></div>
    <div class="card"><div class="lbl">No-lane frames</div><div class="big" id="t_nolane">0</div></div>
  </div>
  <h2>Error history</h2>
  <canvas id="chart"></canvas>
</div>
</div>

<script>
const PARAMS=[
  'blur_ksize','canny_lo','canny_hi','binary_thresh',
  'morph_ksize','morph_iterations',
  'pid_kp','pid_ki','pid_kd',
  'base_speed','speed_reduction_turn',
  'failsafe_no_lane_frames','failsafe_max_error'
];

// ── sliders ───────────────────────────────────────────────────────────
PARAMS.forEach(k=>{
  const el=document.getElementById(k);
  const vEl=document.getElementById('v_'+k);
  if(!el)return;
  el.addEventListener('input',()=>{
    vEl.textContent=el.value;
    fetch('/api/params',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({[k]:parseFloat(el.value)})})
    .then(r=>r.json()).then(d=>setStatus(d.ok?'✓ '+k+'='+el.value:'✗ error'));
  });
});

function setStatus(msg){
  const s=document.getElementById('status');
  s.textContent=msg;
  setTimeout(()=>s.textContent='',2500);
}

function eStop(){
  fetch('/api/estop',{method:'POST'}).then(()=>setStatus('⛔ stop sent'));
}

// ── load current params from server ──────────────────────────────────
fetch('/api/params').then(r=>r.json()).then(data=>{
  PARAMS.forEach(k=>{
    const el=document.getElementById(k);
    const vEl=document.getElementById('v_'+k);
    if(el&&data[k]!==undefined){el.value=data[k];if(vEl)vEl.textContent=data[k];}
  });
});

// ── telemetry polling ─────────────────────────────────────────────────
const errBuf=new Array(120).fill(null);
const canvas=document.getElementById('chart');
const ctx=canvas.getContext('2d');

function drawChart(){
  const W=canvas.clientWidth,H=80;
  canvas.width=W;canvas.height=H;
  ctx.fillStyle='#0d1018';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#1e2430';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();
  const valid=errBuf.filter(v=>v!==null);
  if(!valid.length)return;
  const mx=Math.max(...valid.map(Math.abs),1);
  ctx.strokeStyle='#00e5ff';ctx.lineWidth=1.5;ctx.beginPath();
  let started=false;
  errBuf.forEach((v,i)=>{
    if(v===null)return;
    const x=(i/errBuf.length)*W;
    const y=H/2-(v/mx)*(H/2-4);
    if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

function pollTelem(){
  fetch('/api/telemetry').then(r=>r.json()).then(data=>{
    if(!data.length)return;
    const last=data[data.length-1];
    const fmt=v=>v===null||v===undefined?'—':typeof v==='number'?v.toFixed(3):v;
    document.getElementById('t_error').textContent=fmt(last.error);
    document.getElementById('t_steer').textContent=fmt(last.steering);
    document.getElementById('t_speed').textContent=fmt(last.speed);
    document.getElementById('t_cx').textContent=fmt(last.center_x);
    document.getElementById('t_nolane').textContent=last.no_lane??0;
    // nolane color
    const nl=document.getElementById('t_nolane');
    nl.className='big'+(last.no_lane>3?' warn':'');
    // chart
    data.slice(-errBuf.length).forEach((d,i)=>{errBuf[i]=d.error;});
    drawChart();
  }).catch(()=>{});
}
setInterval(pollTelem,250);
</script>
</body>
</html>
"""


def create_flask_app(pilot: LanePilot) -> Flask:
    app = Flask(__name__)
    log_ws = logging.getLogger("werkzeug")
    log_ws.setLevel(logging.WARNING)   # silence Flask request logs

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/params", methods=["GET"])
    def get_params():
        with _param_lock:
            return jsonify(dict(PARAMS))

    @app.route("/api/params", methods=["POST"])
    def set_params():
        data = request.get_json(force=True) or {}
        changed = {}
        errors  = {}
        for k, v in data.items():
            if set_param(k, v):
                changed[k] = v
            else:
                errors[k] = "unknown key"
        return jsonify({"ok": not errors, "changed": changed, "errors": errors})

    @app.route("/api/telemetry", methods=["GET"])
    def telemetry():
        return jsonify(pilot.get_telemetry())

    @app.route("/api/estop", methods=["POST"])
    def estop():
        _emergency_stop()
        pilot.stop()
        return jsonify({"ok": True, "message": "emergency stop"})

    return app


# ─────────────────────────────────────────────────────────────────────────────
# ── Entry point ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global car
    car = JetRacer()
    car.arm(delay=3)

    pilot = LanePilot()
    app   = create_flask_app(pilot)

    # Graceful shutdown
    def _sig_handler(sig, frame):
        log.info("Signal received – shutting down…")
        pilot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Flask in background daemon thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False),
        daemon=True,
        name="FlaskServer",
    )
    flask_thread.start()
    log.info("Flask tuner running at http://0.0.0.0:5000")

    # Main pipeline on calling thread (keeps GPU context on main thread)
    pilot.run()


if __name__ == "__main__":
    main()