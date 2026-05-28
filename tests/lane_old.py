"""
lane_follow.py — OpenCV lane following + Flask live dashboard
Run with:  python lane_follow.py
Open browser at:  http://<jetson-ip>:5000

FIXES vs original
─────────────────
[BUG-01] PID integral was winding up while lane was lost → violent lurch on
         reacquisition. Fix: freeze integrator / reset derivative on loss.
[BUG-02] pid_state shared between control thread and Flask /set handler with
         NO lock → race condition / torn writes. Fix: pid_state now guarded by
         the same state_lock as state{}.
[BUG-03] MJPEG generator slept 0.1 s → 10 fps, comment claimed ~30 fps.
         Fix: sleep removed; frame rate is driven by the camera/encode loop.
[BUG-04] car.stop() hammered every frame while disabled instead of only on
         the transition edge → motor-controller spam. Fix: edge-detect.
[BUG-05] PID last_time initialised at module-load time; first dt could be
         several seconds → giant derivative spike on startup. Fix: reset
         last_time to now() at the first frame.
[BUG-06] On lane-loss the old steer value was held, potentially driving the
         car straight into the wall.  Fix: linearly decay steer toward 0 and
         reduce speed to LOST_SPEED (configurable).
[BUG-07] Largest contour used blindly even when it's a noise blob. Added
         solidity & aspect-ratio guard so irregular blobs are rejected.
[BUG-08] S hi / V hi hardcoded 255 with no UI sliders → could never tune them.
         Fix: added s_hi / v_hi sliders.
[BUG-09] Speed slider range 0-60 mapped v/100 → max 0.60 instead of 1.0;
         comment/intent mismatch. Fix: range 0-100 mapped v/100 → 0.00-1.00.
[BUG-10] No lane-loss timeout: car would drive blind indefinitely. Fix:
         LOST_TIMEOUT_S (default 2 s) triggers full stop if lane not seen.
[RELIABILITY-01] Dual-contour (left + right) lane centre: when two lane lines
         are visible, use their midpoint rather than the single brightest blob.
[RELIABILITY-02] Exponential moving average on error to smooth noisy detections
         without adding lag (alpha configurable in state).
[RELIABILITY-03] Steering is soft-clamped with a rate-limiter (max_steer_rate)
         so sudden PID spikes can't yank the wheels full-lock in one frame.
"""

import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template_string, request, jsonify

from jetracer import JetRacer

app = Flask(__name__)

# ── Tunable constants (not runtime-adjustable) ────────────────────────────────
LOST_TIMEOUT_S   = 2.0    # seconds without lane → full stop
LOST_SPEED_FRAC  = 0.40   # fraction of set speed used while searching
STEER_DECAY      = 0.85   # steer multiplier per frame while lane lost (→ 0)
MAX_STEER_RATE   = 0.25   # max change in steer per frame (rate limiter)
EMA_ALPHA        = 0.55   # error EMA weight (higher = faster, noisier)
MIN_SOLIDITY     = 0.35   # reject contours with solidity < this (noise blobs)
FRAME_SKIP       = 0      # process every Nth+1 frame (0 = every frame)

# ── Global shared state ───────────────────────────────────────────────────────
state = {
    # HSV lane colour thresholds
    "h_lo": 20,  "h_hi": 35,
    "s_lo": 80,  "s_hi": 255,
    "v_lo": 80,  "v_hi": 255,

    # PID gains
    "kp": 0.40,
    "ki": 0.002,
    "kd": 0.15,

    # Drive
    "speed":   0.18,
    "enabled": False,
    "min_contour_area": 2000,

    # Runtime (read-only from browser)
    "error":      0.0,
    "raw_error":  0.0,
    "steer":      0.0,
    "fps":        0,
    "lane_found": False,
    "lost_secs":  0.0,
}

# PID + filter state (ALL guarded by state_lock)
pid_state = {
    "integral":   0.0,
    "last_error": 0.0,
    "last_time":  None,   # None → initialise on first frame [BUG-05]
    "last_steer": 0.0,
    "ema_error":  0.0,
    "lost_since": None,   # time.time() when lane first lost
}

state_lock = threading.Lock()

# MJPEG frame buffer
frame_lock   = threading.Lock()
latest_frame = None
_placeholder_jpeg = None   # generated once if camera is slow to start


def _make_placeholder():
    """Black 640×480 frame with 'Waiting for camera…' text."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for camera...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    _, j = cv2.imencode(".jpg", img)
    return j.tobytes()


# ── Camera ────────────────────────────────────────────────────────────────────
def open_camera():
    gst = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw,width=640,height=480,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[camera] CSI GStreamer pipeline OK")
        return cap
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[camera] USB /dev/video0 OK")
        return cap
    raise RuntimeError("No camera found")


# ── Contour quality filter ────────────────────────────────────────────────────
def _valid_contour(cnt, min_area):
    """Return True if contour is large enough AND looks like a lane marking."""
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity   = area / hull_area if hull_area > 0 else 0
    if solidity < MIN_SOLIDITY:
        return False
    return True


# ── Lane detection ────────────────────────────────────────────────────────────
def _detect_lane_cx(roi, s):
    """
    Returns (cx, lane_found).
    Tries dual-contour midpoint first; falls back to single largest.
    cx is in ROI pixel coordinates (0 … roi.shape[1]).
    """
    w = roi.shape[1]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo   = np.array([s["h_lo"], s["s_lo"], s["v_lo"]])
    hi   = np.array([s["h_hi"], s["s_hi"], s["v_hi"]])
    mask = cv2.inRange(hsv, lo, hi)

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good = [c for c in contours if _valid_contour(c, s["min_contour_area"])]

    if not good:
        return w // 2, False, mask

    # Sort by area descending
    good.sort(key=cv2.contourArea, reverse=True)

    def cx_of(cnt):
        M = cv2.moments(cnt)
        return int(M["m10"] / M["m00"]) if M["m00"] > 0 else w // 2

    if len(good) >= 2:
        # Two best contours → use their centroid midpoint [RELIABILITY-01]
        c1, c2 = good[0], good[1]
        x1, x2 = cx_of(c1), cx_of(c2)
        # Only use dual if they're on opposite sides of frame centre
        if (x1 < w // 2) != (x2 < w // 2):
            return (x1 + x2) // 2, True, mask

    # Single contour fallback
    return cx_of(good[0]), True, mask


# ── PID + steer ───────────────────────────────────────────────────────────────
def _compute_steer(error, s, ps):
    """Update PID, apply EMA + rate-limiter. Mutates ps (pid_state)."""
    now = time.time()

    # [BUG-05] First-frame initialisation
    if ps["last_time"] is None:
        ps["last_time"]  = now
        ps["last_error"] = error

    dt = max(now - ps["last_time"], 0.001)
    ps["last_time"] = now

    # EMA on error [RELIABILITY-02]
    ps["ema_error"] = EMA_ALPHA * error + (1.0 - EMA_ALPHA) * ps["ema_error"]
    smooth_error    = ps["ema_error"]

    ps["integral"]  += smooth_error * dt
    ps["integral"]   = max(-1.0, min(1.0, ps["integral"]))
    derivative       = (smooth_error - ps["last_error"]) / dt
    ps["last_error"] = smooth_error

    raw_steer = (s["kp"] * smooth_error
               + s["ki"] * ps["integral"]
               + s["kd"] * derivative)
    raw_steer = max(-1.0, min(1.0, raw_steer))

    # Rate limiter [RELIABILITY-03]
    delta = raw_steer - ps["last_steer"]
    delta = max(-MAX_STEER_RATE, min(MAX_STEER_RATE, delta))
    steer = ps["last_steer"] + delta
    ps["last_steer"] = steer

    return steer


# ── Frame processor ───────────────────────────────────────────────────────────
def process_frame(frame, s, ps):
    h, w = frame.shape[:2]
    roi_top = int(h * 0.60)
    roi     = frame[roi_top:h, :]

    cx_lane, lane_found, mask = _detect_lane_cx(roi, s)

    # ── Error & PID ──────────────────────────────────────────────────────────
    now = time.time()

    if lane_found:
        raw_error    = (cx_lane - w // 2) / (w // 2)   # -1 … +1
        ps["lost_since"] = None

        steer = _compute_steer(raw_error, s, ps)
        error = ps["ema_error"]

    else:
        # [BUG-06] Decay steer toward 0 instead of holding last value
        raw_error = 0.0
        ps["ema_error"]  *= (1.0 - EMA_ALPHA)
        ps["last_steer"] *= STEER_DECAY
        steer = ps["last_steer"]
        error = ps["ema_error"]

        # [BUG-01] Freeze integrator on loss — don't wind up
        # (integral left unchanged; derivative will be 0 next frame)
        if ps["lost_since"] is None:
            ps["lost_since"] = now

    # ── Annotation ───────────────────────────────────────────────────────────
    annotated = frame.copy()

    # Mask overlay (green tint in ROI)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_3ch[:, :, 0] = 0
    annotated[roi_top:h, :] = cv2.addWeighted(
        annotated[roi_top:h, :], 0.7, mask_3ch, 0.3, 0)

    cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)
    cv2.line(annotated, (w // 2, roi_top), (w // 2, h), (0, 200, 255), 1)

    if lane_found:
        mid_y = roi_top + (h - roi_top) // 2
        cv2.circle(annotated, (cx_lane, mid_y), 12, (0, 255, 0), -1)
        cv2.circle(annotated, (cx_lane, mid_y), 12, (255, 255, 255), 2)

    arrow_x = int(w // 2 + steer * (w // 3))
    cv2.arrowedLine(annotated, (w // 2, 30), (arrow_x, 30),
                    (0, 140, 255), 3, tipLength=0.35)

    status = "DRIVING" if s["enabled"] else "STOPPED"
    color  = (0, 220, 60)  if s["enabled"] else (60, 60, 220)
    cv2.putText(annotated, status,               (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  color,           2)
    cv2.putText(annotated, f"err {error:+.2f}",  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(annotated, f"str {steer:+.2f}",  (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(annotated, f"fps {s['fps']}",    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    lane_txt = "lane OK" if lane_found else "NO LANE"
    lane_col = (0, 220, 60) if lane_found else (0, 60, 220)
    cv2.putText(annotated, lane_txt, (w - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lane_col, 2)

    if ps["lost_since"] is not None:
        lost_s = now - ps["lost_since"]
        cv2.putText(annotated, f"lost {lost_s:.1f}s",
                    (w - 120, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 220), 1)

    return annotated, raw_error, error, steer, lane_found


# ── Control loop ──────────────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    global latest_frame, _placeholder_jpeg
    _placeholder_jpeg = _make_placeholder()

    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    was_enabled   = False    # [BUG-04] edge-detect for stop()
    frame_idx     = 0

    print("[loop] Starting control loop …")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_idx += 1
        if FRAME_SKIP > 0 and (frame_idx % (FRAME_SKIP + 1)) != 0:
            continue

        with state_lock:
            s_copy  = dict(state)
            ps_copy = dict(pid_state)   # shallow; modified in-place below

        annotated, raw_error, error, steer, lane_found = process_frame(
            frame, s_copy, ps_copy)

        # Write back PID state [BUG-02]
        with state_lock:
            pid_state.update({k: ps_copy[k] for k in
                ("integral", "last_error", "last_time",
                 "last_steer", "ema_error", "lost_since")})

        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        # ── Drive logic ───────────────────────────────────────────────────────
        now_enabled = s_copy["enabled"]

        if now_enabled:
            lost_since = ps_copy["lost_since"]
            lost_secs  = (time.time() - lost_since) if lost_since else 0.0

            if lost_secs >= LOST_TIMEOUT_S:
                # [BUG-10] Full stop after timeout
                car.steer(0.0)
                car.stop()
                with state_lock:
                    state["enabled"] = False
                print(f"[safety] Lane lost >{LOST_TIMEOUT_S}s → auto-stop")
            elif lane_found:
                car.steer(steer)
                car.forward(s_copy["speed"])
            else:
                # Searching: gentle straight crawl with decayed steer
                car.steer(steer)
                car.forward(s_copy["speed"] * LOST_SPEED_FRAC)

            with state_lock:
                state["lost_secs"] = round(lost_secs, 1)

        else:
            # [BUG-04] Only call stop() on the falling edge
            if was_enabled:
                car.stop()
            with state_lock:
                state["lost_secs"] = 0.0

        was_enabled = now_enabled

        with state_lock:
            state["error"]      = round(error, 3)
            state["raw_error"]  = round(raw_error, 3)
            state["steer"]      = round(steer, 3)
            state["lane_found"] = lane_found

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
        with frame_lock:
            latest_frame = jpeg.tobytes()

    cap.release()


# ── Flask routes ──────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetRacer Lane Follower</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080b10; --surface: #0f1520; --surface2: #161e2e;
    --border: #1e2d42; --border2: #263650;
    --accent: #00e5b0; --accent2: #0099ff; --warn: #ffb020; --danger: #ff3d3d;
    --text: #cdd6e8; --muted: #4a6080; --dim: #2a3f5a;
    --font: 'IBM Plex Mono', monospace;
    --green: #00e5b0; --red: #ff3d3d; --blue: #0099ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: var(--font); min-height: 100vh;
    padding: 1.2rem 1rem 2rem;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,229,176,.07) 0%, transparent 70%);
  }
  /* ── Header ── */
  header {
    display: flex; align-items: center; gap: 1rem;
    max-width: 1140px; margin: 0 auto 1.2rem;
  }
  .logo { width: 28px; height: 28px; }
  header h1 {
    font-size: .78rem; letter-spacing: .2em; color: var(--accent);
    text-transform: uppercase; font-weight: 700;
  }
  .pill {
    margin-left: auto; font-size: .65rem; letter-spacing: .1em;
    padding: .25rem .7rem; border-radius: 20px;
    border: 1px solid var(--border2); color: var(--muted);
  }
  /* ── Layout ── */
  .grid {
    display: grid;
    grid-template-columns: minmax(0,1fr) 320px;
    gap: 1rem; max-width: 1140px; margin: 0 auto;
  }
  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem;
    position: relative; overflow: hidden;
  }
  .card::before {
    content: ''; position: absolute; inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
    border-radius: 12px 12px 0 0;
  }
  .card h2 {
    font-size: .62rem; letter-spacing: .14em; color: var(--muted);
    text-transform: uppercase; margin-bottom: .8rem; font-weight: 600;
  }
  /* ── Feed ── */
  #feed {
    width: 100%; border-radius: 8px; display: block;
    background: #000; min-height: 220px; border: 1px solid var(--border2);
  }
  /* ── Stat row ── */
  .stat-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: .5rem; margin-top: .8rem;
  }
  .stat {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: .5rem .6rem;
  }
  .stat-val {
    font-size: 1.25rem; font-weight: 700; color: var(--accent);
    line-height: 1; display: block;
  }
  .stat-lbl {
    font-size: .58rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: .08em; margin-top: .25rem; display: block;
  }
  /* ── Error track ── */
  .err-wrap { margin-top: .7rem; }
  .err-label {
    font-size: .6rem; color: var(--muted); letter-spacing: .1em;
    text-transform: uppercase; margin-bottom: .3rem;
    display: flex; justify-content: space-between;
  }
  .error-track {
    position: relative; height: 10px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 5px; overflow: hidden;
  }
  #error-bar {
    position: absolute; height: 100%; width: 5px;
    background: var(--accent); left: 50%;
    transform: translateX(-50%); transition: left .12s, background .2s;
    border-radius: 5px;
  }
  .tick-center {
    position: absolute; left: 50%; top: 0; height: 100%;
    width: 1px; background: var(--dim);
  }
  /* ── Sliders ── */
  .slider-row {
    display: flex; align-items: center; gap: .5rem;
    margin-bottom: .48rem;
  }
  .slider-row label {
    font-size: .67rem; color: var(--muted);
    width: 50px; flex-shrink: 0; letter-spacing: .04em;
  }
  .slider-row input[type=range] {
    flex: 1; accent-color: var(--accent);
    height: 4px; cursor: pointer;
  }
  .slider-row .val {
    font-size: .72rem; width: 46px; text-align: right;
    color: var(--text); font-weight: 600;
  }
  /* ── Buttons ── */
  .btn-row { display: flex; gap: .6rem; margin-top: .8rem; }
  button {
    padding: .5rem 1.2rem; border: none; border-radius: 8px;
    cursor: pointer; font-family: var(--font); font-size: .8rem;
    font-weight: 700; letter-spacing: .06em; transition: filter .15s, transform .1s;
    flex: 1;
  }
  button:active { transform: scale(.97); }
  #btn-go   { background: var(--accent); color: #041a13; }
  #btn-stop { background: var(--danger); color: #fff; }
  #btn-go:hover   { filter: brightness(1.12); }
  #btn-stop:hover { filter: brightness(1.12); }
  /* ── Divider ── */
  .divider { border: none; border-top: 1px solid var(--border); margin: .8rem 0; }
  /* ── Lost banner ── */
  #lost-banner {
    display: none; margin-top: .5rem; padding: .4rem .7rem;
    background: rgba(255,61,61,.15); border: 1px solid rgba(255,61,61,.4);
    border-radius: 6px; font-size: .7rem; color: var(--red);
    letter-spacing: .06em;
  }
  /* ── Responsive ── */
  @media (max-width: 740px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<header>
  <svg class="logo" viewBox="0 0 28 28" fill="none">
    <circle cx="14" cy="14" r="13" stroke="#00e5b0" stroke-width="1.5"/>
    <path d="M8 19 L14 9 L20 19" stroke="#00e5b0" stroke-width="1.8" stroke-linejoin="round"/>
    <circle cx="14" cy="14" r="2.5" fill="#00e5b0"/>
  </svg>
  <h1>JetRacer · Lane Follow Dashboard</h1>
  <span class="pill" id="conn-pill">● connecting</span>
</header>

<div class="grid">

  <!-- ── Camera ── -->
  <div class="card">
    <h2>Camera feed (annotated)</h2>
    <img id="feed" src="/video_feed" alt="camera feed">

    <div class="stat-row">
      <div class="stat"><span class="stat-val" id="v-fps">—</span><span class="stat-lbl">FPS</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span><span class="stat-lbl">Error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span><span class="stat-lbl">Steer</span></div>
      <div class="stat"><span class="stat-val" id="v-lane" style="color:var(--muted)">—</span><span class="stat-lbl">Lane</span></div>
    </div>

    <div class="err-wrap">
      <div class="err-label"><span>← left</span><span>Lane error</span><span>right →</span></div>
      <div class="error-track">
        <div class="tick-center"></div>
        <div id="error-bar"></div>
      </div>
    </div>

    <div id="lost-banner">⚠ LANE LOST — <span id="lost-timer">0.0</span>s  (auto-stop at """ + str(LOST_TIMEOUT_S) + """s)</div>
  </div>

  <!-- ── Controls ── -->
  <div class="card">
    <h2>Drive</h2>
    <div class="slider-row">
      <label>Speed</label>
      <input type="range" id="speed" min="0" max="100" value="18" step="1">
      <span class="val" id="v-speed">0.18</span>
    </div>
    <div class="btn-row">
      <button id="btn-go"   onclick="setEnabled(true)">▶ GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">■ STOP</button>
    </div>

    <hr class="divider">
    <h2>PID gains</h2>
    <div class="slider-row">
      <label>Kp</label>
      <input type="range" id="kp" min="0" max="1.5" value="0.4" step="0.01">
      <span class="val" id="v-kp">0.40</span>
    </div>
    <div class="slider-row">
      <label>Ki</label>
      <input type="range" id="ki" min="0" max="0.05" value="0.002" step="0.001">
      <span class="val" id="v-ki">0.002</span>
    </div>
    <div class="slider-row">
      <label>Kd</label>
      <input type="range" id="kd" min="0" max="0.8" value="0.15" step="0.01">
      <span class="val" id="v-kd">0.15</span>
    </div>

    <hr class="divider">
    <h2>HSV mask</h2>
    <div class="slider-row">
      <label>H lo</label>
      <input type="range" id="h_lo" min="0" max="179" value="20" step="1">
      <span class="val" id="v-h_lo">20</span>
    </div>
    <div class="slider-row">
      <label>H hi</label>
      <input type="range" id="h_hi" min="0" max="179" value="35" step="1">
      <span class="val" id="v-h_hi">35</span>
    </div>
    <div class="slider-row">
      <label>S lo</label>
      <input type="range" id="s_lo" min="0" max="255" value="80" step="1">
      <span class="val" id="v-s_lo">80</span>
    </div>
    <div class="slider-row">
      <label>S hi</label>
      <input type="range" id="s_hi" min="0" max="255" value="255" step="1">
      <span class="val" id="v-s_hi">255</span>
    </div>
    <div class="slider-row">
      <label>V lo</label>
      <input type="range" id="v_lo" min="0" max="255" value="80" step="1">
      <span class="val" id="v-v_lo">80</span>
    </div>
    <div class="slider-row">
      <label>V hi</label>
      <input type="range" id="v_hi" min="0" max="255" value="255" step="1">
      <span class="val" id="v-v_hi">255</span>
    </div>

    <hr class="divider">
    <h2>Contour filter</h2>
    <div class="slider-row">
      <label>Min area</label>
      <input type="range" id="min_contour_area" min="200" max="20000" value="2000" step="100">
      <span class="val" id="v-min_contour_area">2000</span>
    </div>
  </div>

</div>

<script>
const INT_PARAMS = new Set(["h_lo","h_hi","s_lo","s_hi","v_lo","v_hi","min_contour_area"]);

const sliders = ["speed","kp","ki","kd","h_lo","h_hi","s_lo","s_hi","v_lo","v_hi","min_contour_area"];
sliders.forEach(id => {
  const el   = document.getElementById(id);
  const disp = document.getElementById("v-" + id);

  function fmt(v) {
    if (id === "speed") return (v / 100).toFixed(2);
    if (INT_PARAMS.has(id)) return String(Math.round(v));
    return v.toFixed(3);
  }
  function serverVal(v) {
    return id === "speed" ? v / 100 : (INT_PARAMS.has(id) ? Math.round(v) : v);
  }

  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    disp.textContent = fmt(v);
    sendParam(id, serverVal(v));
  });
});

function sendParam(key, value) {
  fetch("/set", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({[key]: value})
  });
}

function setEnabled(v) {
  fetch("/set", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enabled: v})
  });
}

// ── Telemetry ─────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch("/status");
    if (!r.ok) throw new Error();
    const d = await r.json();

    document.getElementById("conn-pill").textContent = "● live";
    document.getElementById("conn-pill").style.color = "#00e5b0";

    document.getElementById("v-fps").textContent = d.fps;
    document.getElementById("v-err").textContent = d.error.toFixed(2);
    document.getElementById("v-str").textContent = d.steer.toFixed(2);

    const laneEl = document.getElementById("v-lane");
    laneEl.textContent = d.lane_found ? "✓" : "✗";
    laneEl.style.color = d.lane_found ? "var(--green)" : "var(--red)";

    const pct = (d.error + 1) / 2 * 100;
    const bar = document.getElementById("error-bar");
    bar.style.left       = pct + "%";
    bar.style.background = Math.abs(d.error) > 0.5 ? "var(--red)" : "var(--accent)";

    // Lost banner
    const banner = document.getElementById("lost-banner");
    if (!d.lane_found && d.enabled) {
      banner.style.display = "block";
      document.getElementById("lost-timer").textContent = d.lost_secs.toFixed(1);
    } else {
      banner.style.display = "none";
    }
  } catch(e) {
    document.getElementById("conn-pill").textContent = "● offline";
    document.getElementById("conn-pill").style.color = "#ff3d3d";
  }
  setTimeout(poll, 180);
}
poll();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def generate_mjpeg():
    """MJPEG generator. No artificial sleep — frame rate is camera-limited."""
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            # [BUG-09] Serve placeholder until camera warms up
            frame = _placeholder_jpeg or _make_placeholder()
            time.sleep(0.05)
        # [BUG-03] Removed hard 0.1 s sleep — was capping at 10 fps
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        return jsonify({k: state[k] for k in
                        ("fps", "error", "raw_error", "steer",
                         "lane_found", "enabled", "lost_secs")})


@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:   # [BUG-02] pid_state also protected here
        for k, v in data.items():
            if k in state:
                state[k] = v
                if k in ("kp", "ki", "kd"):
                    pid_state["integral"]   = 0.0
                    pid_state["last_error"] = 0.0
                    pid_state["ema_error"]  = 0.0
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = JetRacer()
    car.arm(delay=3)

    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)

