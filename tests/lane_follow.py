"""
lane_follow.py — Inner-line hugger for JetRacer (Jetson Nano 4 GB)
Track: dark road, white lane lines, green inner island (VIT & ARC mat)

Core philosophy:
  • Always identify the INNER white line (closest to green center).
  • Drive at a fixed pixel offset FROM that inner line toward the road center.
  • BOTH lines visible  → inner = the one geometrically closer to frame center.
  • ONE line visible    → classify by screen position; apply offset accordingly.
  • NO lines visible    → hold last steer, reduce speed.

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
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

app = Flask(__name__)

# ── CUDA check ────────────────────────────────────────────────────────────────
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"[init] {'CUDA active' if USE_CUDA else 'CPU fallback'}")
_gpu_frame = cv2.cuda_GpuMat() if USE_CUDA else None
_gpu_hsv   = cv2.cuda_GpuMat() if USE_CUDA else None

# ── Constants ─────────────────────────────────────────────────────────────────
WIDTH, HEIGHT  = 320, 240
ROI_FRAC       = 0.60       # use bottom 40 % of frame
ENCODE_EVERY   = 3
JPEG_QUALITY   = 30
MJPEG_INTERVAL = 1 / 15
MORPH_KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    # HSV thresholds for white line detection
    "h_lo": 0,   "h_hi": 180,
    "s_lo": 0,   "s_hi": 50,
    "v_lo": 200, "v_hi": 255,

    # PID
    "kp": 0.70, "ki": 0.002, "kd": 0.30,

    # Drive
    "speed": 0.45,
    "enabled": False,

    # Lane geometry
    "lane_width": 300,      # expected pixel distance outer→inner line
    "inner_offset": 60,     # pixels to stay away from inner line (into road)
    "min_contour_area": 400,

    # Telemetry (written by control loop, read by /status)
    "error": 0.0,
    "steer": 0.0,
    "fps": 0,
    "lane_state": "NONE",   # BOTH-IL | BOTH-IR | SINGLE-IL | SINGLE-IR | NONE
}

pid_state = {
    "integral":   0.0,
    "last_error": 0.0,
    "last_time":  time.time(),
    "last_steer": 0.0,
}
state_lock = threading.Lock()

frame_lock   = threading.Lock()
latest_frame = None
stream_clients   = 0
clients_lock = threading.Lock()
_encode_pool = ThreadPoolExecutor(max_workers=1)


# ── Camera ────────────────────────────────────────────────────────────────────
def _gst_pipeline(sensor_id=0, cap_w=1280, cap_h=720,
                  disp_w=WIDTH, disp_h=HEIGHT, fps=60, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={cap_w}, height={cap_h}, "
        f"framerate={fps}/1, format=NV12 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={disp_w}, height={disp_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1"
    )

def open_camera():
    cap = cv2.VideoCapture(_gst_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] CSI {WIDTH}×{HEIGHT} OK")
        return cap
    print("[camera] GStreamer failed → USB fallback")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[camera] USB {WIDTH}×{HEIGHT} OK")
        return cap
    raise RuntimeError("No camera found (CSI and USB both failed)")


# ── HSV masking (GPU or CPU) ──────────────────────────────────────────────────
def _build_gpu_mask_fn():
    probe = cv2.cuda_GpuMat()
    probe.upload(np.zeros((1, 1, 3), dtype=np.uint8))
    try:
        result = cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2HSV)
        if result is not None and not result.empty():
            def fn(roi, lo, hi):
                _gpu_frame.upload(roi)
                hsv = cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV).download()
                return cv2.inRange(hsv, lo, hi)
            return fn
    except Exception:
        pass
    # fallback in-place API
    def fn(roi, lo, hi):
        _gpu_frame.upload(roi)
        cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV, _gpu_hsv)
        return cv2.inRange(_gpu_hsv.download(), lo, hi)
    return fn

_gpu_mask_fn = _build_gpu_mask_fn() if USE_CUDA else None

def get_white_mask(roi, lo, hi):
    if _gpu_mask_fn:
        return _gpu_mask_fn(roi, lo, hi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lo, hi)


# ── Line detection ────────────────────────────────────────────────────────────
def find_line_centroids(mask, min_area):
    """
    Return list of (cx, cy, area) for valid white blobs in mask,
    sorted left-to-right by cx.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        lines.append((cx, cy, area))
    lines.sort(key=lambda x: x[0])
    return lines


def pick_best_left_right(lines):
    """
    Split blobs into left-of-center and right-of-center pools.
    Return the largest blob from each pool as (cx, cy) or None.
    """
    mid = WIDTH // 2
    left_pool  = [(cx, cy, a) for cx, cy, a in lines if cx <  mid]
    right_pool = [(cx, cy, a) for cx, cy, a in lines if cx >= mid]

    left  = max(left_pool,  key=lambda x: x[2])[:2] if left_pool  else None
    right = max(right_pool, key=lambda x: x[2])[:2] if right_pool else None
    return left, right          # each is (cx, cy) or None


# ── Core processing ───────────────────────────────────────────────────────────
def process_frame(frame, s, annotate: bool):
    """
    Detect white lines in ROI.
    Always compute a target_x that is offset_px AWAY from the inner line
    toward the road centre.  Feed into PID to produce a steer value.
    """
    h, w = frame.shape[:2]
    roi_top = int(h * ROI_FRAC)
    roi = frame[roi_top:h, :]

    lo = np.array([s["h_lo"], s["s_lo"], s["v_lo"]], dtype=np.uint8)
    hi = np.array([s["h_hi"], s["s_hi"], s["v_hi"]], dtype=np.uint8)

    # White mask + morphology
    mask = get_white_mask(roi, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    lines       = find_line_centroids(mask, s["min_contour_area"])
    left, right = pick_best_left_right(lines)

    offset_px = s["inner_offset"]
    cx_mid    = w // 2

    # ── Identify inner line and set target_x ──────────────────────────────────
    #
    # On this oval track the green island is always at the CENTER.
    # The inner white line is therefore always CLOSER to the frame center than
    # the outer white line.
    #
    # "Hug inner" means:
    #   • inner line is to the LEFT  → target = inner_cx + offset  (stay just right of it)
    #   • inner line is to the RIGHT → target = inner_cx - offset  (stay just left of it)

    if left and right:
        left_cx,  left_cy  = left
        right_cx, right_cy = right

        dist_left  = abs(left_cx  - cx_mid)
        dist_right = abs(right_cx - cx_mid)

        if dist_left < dist_right:
            # Left line is INNER (closer to center)
            inner_cx, inner_cy = left_cx,  left_cy
            target_x   = inner_cx + offset_px   # stay right of inner-left line
            lane_state = "BOTH-IL"
        else:
            # Right line is INNER
            inner_cx, inner_cy = right_cx, right_cy
            target_x   = inner_cx - offset_px   # stay left of inner-right line
            lane_state = "BOTH-IR"

        inner_pt = (inner_cx, inner_cy)

    elif left:
        left_cx, left_cy = left
        inner_pt = (left_cx, left_cy)
        if left_cx >= cx_mid:
            # Blob is on the right half → inner right line
            target_x   = left_cx - offset_px
            lane_state = "SINGLE-IR"
        else:
            # Blob is on the left half → inner left line
            target_x   = left_cx + offset_px
            lane_state = "SINGLE-IL"

    elif right:
        right_cx, right_cy = right
        inner_pt = (right_cx, right_cy)
        if right_cx < cx_mid:
            # Blob is on the left half → inner left line
            target_x   = right_cx + offset_px
            lane_state = "SINGLE-IL"
        else:
            # Blob is on the right half → inner right line
            target_x   = right_cx - offset_px
            lane_state = "SINGLE-IR"

    else:
        inner_pt   = None
        target_x   = cx_mid      # dummy
        lane_state = "NONE"

    # Clamp target_x to frame bounds
    target_x = max(0, min(w - 1, target_x))

    # ── PID ───────────────────────────────────────────────────────────────────
    lane_found = lane_state != "NONE"

    if lane_found:
        # error: positive = target is right of centre → steer right
        error = (target_x - cx_mid) / cx_mid
        error = max(-1.0, min(1.0, error))

        now = time.time()
        dt  = max(now - pid_state["last_time"], 0.001)

        pid_state["integral"] += error * dt
        pid_state["integral"]  = max(-1.0, min(1.0, pid_state["integral"]))
        derivative             = (error - pid_state["last_error"]) / dt

        pid_state["last_error"] = error
        pid_state["last_time"]  = now

        steer = (s["kp"] * error
               + s["ki"] * pid_state["integral"]
               + s["kd"] * derivative)
        steer = max(-1.0, min(1.0, steer))
        pid_state["last_steer"] = steer

    else:
        error = 0.0
        steer = pid_state["last_steer"]   # hold last known steer

    # ── Annotation ────────────────────────────────────────────────────────────
    if annotate:
        vis = frame.copy()

        # ROI boundary
        cv2.line(vis, (0, roi_top), (w, roi_top), (255, 255, 0), 1)

        # Mask overlay (green tint in ROI)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_3ch[:, :, [0, 2]] = 0
        vis[roi_top:h] = cv2.addWeighted(vis[roi_top:h], 0.7, mask_3ch, 0.3, 0)

        # Left blob → blue dot, right blob → orange dot
        if left:
            cv2.circle(vis, (left[0],  roi_top + left[1]),  9, (255, 80,  0),   -1)
            cv2.circle(vis, (left[0],  roi_top + left[1]),  9, (255,255,255),    2)
        if right:
            cv2.circle(vis, (right[0], roi_top + right[1]), 9, (0,   80, 255),  -1)
            cv2.circle(vis, (right[0], roi_top + right[1]), 9, (255,255,255),    2)

        # Inner line (diamond marker)
        if inner_pt:
            ix = inner_pt[0]
            iy = roi_top + inner_pt[1]
            pts = np.array([[ix, iy-10],[ix+7,iy],[ix,iy+10],[ix-7,iy]], np.int32)
            cv2.fillPoly(vis, [pts], (0, 230, 120))
            cv2.polylines(vis, [pts], True, (255,255,255), 1)

        # Target X line (cyan)
        if lane_found:
            cy_mid = roi_top + (h - roi_top) // 2
            cv2.line(vis, (target_x, roi_top), (target_x, h), (0, 220, 220), 1)
            cv2.circle(vis, (target_x, cy_mid), 6, (0, 220, 220), -1)

        # Frame centre line (grey)
        cv2.line(vis, (cx_mid, roi_top), (cx_mid, h), (80, 80, 80), 1)

        # Steer arrow at top
        arrow_x = int(cx_mid + steer * (w // 3))
        cv2.arrowedLine(vis, (cx_mid, 22), (arrow_x, 22), (0, 160, 255), 2, tipLength=0.35)

        # Text
        status = "DRIVE" if s["enabled"] else "STOP"
        col    = (0, 220, 60) if s["enabled"] else (60, 60, 220)
        cv2.putText(vis, status, (5, 18),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(vis, f"e{error:+.2f} s{steer:+.2f}", (5, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, f"fps {s['fps']}", (5, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        lc = (0, 220, 60) if lane_found else (0, 60, 220)
        cv2.putText(vis, lane_state, (w - 95, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, lc, 1)
    else:
        vis = frame

    return vis, error, steer, lane_state


# ── Async JPEG encode ─────────────────────────────────────────────────────────
def _do_encode(img):
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ret:
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
            s = dict(state)

        with clients_lock:
            do_annotate = stream_clients > 0 and (frame_idx % ENCODE_EVERY == 0)

        vis, error, steer, lane_state = process_frame(frame, s, do_annotate)

        if do_annotate:
            _encode_pool.submit(_do_encode, vis)

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        if s["enabled"]:
            speed = s["speed"]
            if lane_state == "NONE":
                speed *= 0.5    # coast slowly when blind
            car.steer(steer)
            car.forward(speed)
        else:
            car.stop()

        with state_lock:
            state["error"]      = round(error, 3)
            state["steer"]      = round(steer, 3)
            state["lane_state"] = lane_state

        frame_idx += 1

    cap.release()


# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetRacer · Inner Line Hugger</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;500&display=swap');
  :root {
    --bg: #080c12; --surface: #0f1520; --surface2: #161e2e;
    --border: #1e2d42; --accent: #00e5c0; --warn: #f5a623;
    --danger: #ff3d5a; --text: #d8e4f0; --muted: #4d6680;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh; padding: 1.25rem;
    background-image:
      radial-gradient(ellipse at 15% 0%, #001a2e55 0%, transparent 55%),
      radial-gradient(ellipse at 85% 100%, #00e5c010 0%, transparent 55%);
  }
  header { display: flex; align-items: center; gap: .75rem; margin-bottom: 1.25rem; }
  .pulse {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 10px var(--accent);
    animation: blink 2s infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
  h1 { font-family:'Space Mono',monospace; font-size:.82rem;
       letter-spacing:.22em; color:var(--accent); text-transform:uppercase; }

  .grid { display:grid; grid-template-columns:1fr 310px; gap:1rem;
          max-width:1060px; margin:0 auto; }
  .card { background:var(--surface); border:1px solid var(--border);
          border-radius:12px; padding:1rem; }
  .card-title { font-family:'Space Mono',monospace; font-size:.58rem;
                letter-spacing:.15em; color:var(--muted); text-transform:uppercase;
                margin-bottom:.75rem; }

  img#feed { width:100%; border-radius:8px; display:block;
             background:#000; min-height:150px; border:1px solid var(--border); }

  .stats { display:flex; gap:.75rem; flex-wrap:wrap; margin-top:.85rem; }
  .stat { background:var(--surface2); border-radius:8px; padding:.45rem .7rem; min-width:60px; }
  .stat-val { font-family:'Space Mono',monospace; font-size:1.2rem;
              font-weight:700; color:var(--accent); line-height:1; }
  .stat-lbl { font-size:.58rem; color:var(--muted); margin-top:2px;
              text-transform:uppercase; letter-spacing:.08em; }

  .err-track { position:relative; height:6px; background:var(--border);
               border-radius:3px; margin-top:.85rem; overflow:hidden; }
  #err-bar { position:absolute; height:100%; width:6px; background:var(--accent);
             left:50%; transform:translateX(-50%);
             transition:left .08s, background .2s; border-radius:3px; }
  .err-track::after { content:''; position:absolute; left:50%; top:0;
                      width:1px; height:100%; background:var(--muted);
                      transform:translateX(-50%); }

  .legend { display:flex; gap:.75rem; margin-top:.6rem; flex-wrap:wrap; }
  .legend-item { display:flex; align-items:center; gap:.3rem;
                 font-size:.62rem; color:var(--muted); }
  .dot { width:8px; height:8px; border-radius:50%; }

  .section-label { font-family:'Space Mono',monospace; font-size:.58rem;
                   color:var(--muted); letter-spacing:.12em; text-transform:uppercase;
                   margin:.85rem 0 .5rem; padding-bottom:.35rem;
                   border-bottom:1px solid var(--border); }
  .row { display:flex; align-items:center; gap:.5rem; margin-bottom:.42rem; }
  .row label { font-size:.66rem; color:var(--muted); width:68px; flex-shrink:0; }
  .row input[type=range] { flex:1; accent-color:var(--accent); cursor:pointer; }
  .row .val { font-family:'Space Mono',monospace; font-size:.68rem;
              width:46px; text-align:right; color:var(--text); }

  .btn-row { display:flex; gap:.5rem; margin-top:.5rem; }
  button { flex:1; padding:.55rem; border:none; border-radius:8px; cursor:pointer;
           font-family:'Space Mono',monospace; font-size:.75rem; font-weight:700;
           letter-spacing:.05em; transition:filter .15s, transform .1s; }
  button:active { transform:scale(.97); }
  #btn-go   { background:var(--accent); color:#021a14; }
  #btn-stop { background:var(--danger); color:#fff; }
  #btn-go:hover, #btn-stop:hover { filter:brightness(1.15); }

  @media(max-width:700px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div class="pulse"></div>
  <h1>JetRacer &middot; Inner Line Hugger</h1>
</header>

<div class="grid">
  <!-- Camera card -->
  <div class="card">
    <div class="card-title">Camera feed (annotated)</div>
    <img id="feed" src="/video_feed" alt="feed">
    <div class="stats">
      <div class="stat"><div class="stat-val" id="v-fps">0</div><div class="stat-lbl">fps</div></div>
      <div class="stat"><div class="stat-val" id="v-err">0.00</div><div class="stat-lbl">error</div></div>
      <div class="stat"><div class="stat-val" id="v-str">0.00</div><div class="stat-lbl">steer</div></div>
      <div class="stat"><div class="stat-val" id="v-lane" style="font-size:.8rem">—</div><div class="stat-lbl">lane</div></div>
    </div>
    <div class="err-track" title="error: left=-1  centre=0  right=+1">
      <div id="err-bar"></div>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#ff5020"></div>Left blob</div>
      <div class="legend-item"><div class="dot" style="background:#0050ff"></div>Right blob</div>
      <div class="legend-item"><div class="dot" style="background:#00e060"></div>Inner line</div>
      <div class="legend-item"><div class="dot" style="background:#00e5c0"></div>Target</div>
    </div>
  </div>

  <!-- Controls card -->
  <div class="card">
    <div class="card-title">Controls</div>

    <div class="section-label">Drive</div>
    <div class="row">
      <label>Speed</label>
      <input type="range" id="speed" min="0" max="70" value="45" step="1">
      <span class="val" id="v-speed">0.45</span>
    </div>
    <div class="btn-row">
      <button id="btn-go"   onclick="setEnabled(true)">&#9654; GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
    </div>

    <div class="section-label">PID Gains</div>
    <div class="row"><label>Kp</label>
      <input type="range" id="kp" min="0" max="1.5" value="0.70" step="0.01">
      <span class="val" id="v-kp">0.70</span></div>
    <div class="row"><label>Ki</label>
      <input type="range" id="ki" min="0" max="0.05" value="0.002" step="0.001">
      <span class="val" id="v-ki">0.002</span></div>
    <div class="row"><label>Kd</label>
      <input type="range" id="kd" min="0" max="0.8" value="0.30" step="0.01">
      <span class="val" id="v-kd">0.30</span></div>

    <div class="section-label">Inner Hug</div>
    <div class="row"><label>Offset px</label>
      <input type="range" id="inner_offset" min="10" max="160" value="60" step="5">
      <span class="val" id="v-inner_offset">60</span></div>
    <div class="row"><label>Lane W px</label>
      <input type="range" id="lane_width" min="50" max="500" value="300" step="5">
      <span class="val" id="v-lane_width">300</span></div>
    <div class="row"><label>Min area</label>
      <input type="range" id="min_contour_area" min="50" max="3000" value="400" step="50">
      <span class="val" id="v-min_contour_area">400</span></div>

    <div class="section-label">HSV (white lines)</div>
    <div class="row"><label>H lo</label>
      <input type="range" id="h_lo" min="0" max="179" value="0" step="1">
      <span class="val" id="v-h_lo">0</span></div>
    <div class="row"><label>H hi</label>
      <input type="range" id="h_hi" min="0" max="179" value="180" step="1">
      <span class="val" id="v-h_hi">180</span></div>
    <div class="row"><label>S lo</label>
      <input type="range" id="s_lo" min="0" max="255" value="0" step="1">
      <span class="val" id="v-s_lo">0</span></div>
    <div class="row"><label>S hi</label>
      <input type="range" id="s_hi" min="0" max="255" value="50" step="1">
      <span class="val" id="v-s_hi">50</span></div>
    <div class="row"><label>V lo</label>
      <input type="range" id="v_lo" min="0" max="255" value="200" step="1">
      <span class="val" id="v-v_lo">200</span></div>
    <div class="row"><label>V hi</label>
      <input type="range" id="v_hi" min="0" max="255" value="255" step="1">
      <span class="val" id="v-v_hi">255</span></div>
  </div>
</div>

<script>
const PARAMS = [
  {id:"speed",           fmt: v => (v/100).toFixed(2), tx: v => v/100},
  {id:"kp",              fmt: v => v.toFixed(2)},
  {id:"ki",              fmt: v => v.toFixed(3)},
  {id:"kd",              fmt: v => v.toFixed(2)},
  {id:"inner_offset",    fmt: v => parseInt(v)},
  {id:"lane_width",      fmt: v => parseInt(v)},
  {id:"min_contour_area",fmt: v => parseInt(v)},
  {id:"h_lo"},{id:"h_hi"},{id:"s_lo"},{id:"s_hi"},{id:"v_lo"},{id:"v_hi"},
];
PARAMS.forEach(p => {
  const el = document.getElementById(p.id);
  const dv = document.getElementById("v-" + p.id);
  if (!el) return;
  el.addEventListener("input", () => {
    const raw = parseFloat(el.value);
    dv.textContent = p.fmt ? p.fmt(raw) : raw;
    const sv = p.tx ? p.tx(raw) : raw;
    fetch("/set",{method:"POST",headers:{"Content-Type":"application/json"},
                  body:JSON.stringify({[p.id]: parseFloat(sv)})});
  });
});
function setEnabled(v){
  fetch("/set",{method:"POST",headers:{"Content-Type":"application/json"},
                body:JSON.stringify({enabled:v})});
}
const LANE_COLOR = {
  "BOTH-IL":"#00e5c0","BOTH-IR":"#00e5c0",
  "SINGLE-IL":"#f5a623","SINGLE-IR":"#f5a623","NONE":"#ff3d5a"
};
const LANE_SHORT = {
  "BOTH-IL":"BOTH","BOTH-IR":"BOTH",
  "SINGLE-IL":"SNGL","SINGLE-IR":"SNGL","NONE":"NONE"
};
async function poll(){
  try{
    const d = await (await fetch("/status")).json();
    document.getElementById("v-fps").textContent = d.fps;
    document.getElementById("v-err").textContent = d.error.toFixed(2);
    document.getElementById("v-str").textContent = d.steer.toFixed(2);
    const lel = document.getElementById("v-lane");
    lel.textContent  = LANE_SHORT[d.lane_state] || d.lane_state;
    lel.style.color  = LANE_COLOR[d.lane_state] || "#fff";
    const pct = (d.error + 1) / 2 * 100;
    const bar = document.getElementById("err-bar");
    bar.style.left       = pct + "%";
    bar.style.background = Math.abs(d.error) > 0.6 ? "#ff3d5a" : "#00e5c0";
  }catch(e){}
  setTimeout(poll, 200);
}
poll();
</script>
</body>
</html>"""


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

def _generate_mjpeg():
    global stream_clients
    with clients_lock:
        stream_clients += 1
    try:
        while True:
            with frame_lock:
                f = latest_frame
            if f is None:
                time.sleep(0.02)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(MJPEG_INTERVAL)
    finally:
        with clients_lock:
            stream_clients -= 1

@app.route("/video_feed")
def video_feed():
    return Response(_generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with state_lock:
        return jsonify({k: state[k] for k in
                        ("fps","error","steer","lane_state","enabled")})

@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k in state:
                state[k] = v
                if k in ("kp","ki","kd"):
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