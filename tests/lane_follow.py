"""
lane_follow_optimized.py — GPU-accelerated OpenCV lane following + Flask dashboard
Optimised for Jetson Nano 4 GB:
  • CUDA HSV masking (GPU)
  • Reduced resolution 320×240
  • Frame-skip JPEG encoding (every Nth frame)
  • Async encode via ThreadPoolExecutor
  • Annotation skipped when no browser connected
  • GStreamer pipeline kept in NVMM until final BGR conversion

Run:   python lane_follow_optimized.py
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
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"        # Disable MSMF (Windows)
# GStreamer MUST stay enabled — CSI camera outputs raw Bayer (RG10)
# and needs nvarguscamerasrc for ISP debayering
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"      # Disable FFMPEG

app = Flask(__name__)

# ── CUDA availability check ───────────────────────────────────────────────────
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_CUDA:
    print("[init] CUDA device found — GPU path active")
    _gpu_frame = cv2.cuda_GpuMat()
    _gpu_hsv   = cv2.cuda_GpuMat()
    _gpu_mask  = cv2.cuda_GpuMat()
else:
    print("[init] No CUDA device — falling back to CPU")
    _gpu_frame = _gpu_hsv = _gpu_mask = None

# ── Config constants ──────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240          # lower res = less GPU/CPU work
ENCODE_EVERY    = 3                  # encode JPEG only every Nth frame
JPEG_QUALITY    = 30                 # lower = smaller payload, less CPU
MJPEG_INTERVAL  = 1 / 15            # 15 fps to browser
ROI_FRAC        = 0.60               # bottom 40% used as ROI

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "h_lo": 0,  "h_hi": 180,
    "s_lo": 0,  "s_hi": 50,
    "v_lo": 200,  "v_hi": 255,
    "kp": 0.4,   "ki": 0.002,  "kd": 0.15,
    "speed": 0.18,
    "enabled": False,
    "min_contour_area": 500,   # scaled down for 320×240 (was 2000 @ 640×480)
    # telemetry (read-only from browser)
    "error": 0.0,  "steer": 0.0,  "fps": 0,  "lane_found": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()

# Latest JPEG bytes for MJPEG stream
frame_lock   = threading.Lock()
latest_frame = None

# Count of active MJPEG clients — skip annotation when 0
stream_clients = 0
clients_lock   = threading.Lock()

# Async JPEG encoder (1 worker is enough; encoding is sequential)
_encode_pool = ThreadPoolExecutor(max_workers=1)


# ── Camera ────────────────────────────────────────────────────────────────────
def _gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280, capture_height=720,
    display_width=WIDTH, display_height=HEIGHT,
    framerate=60, flip_method=0,
):
    """
    Build a GStreamer pipeline string for nvarguscamerasrc (Jetson CSI cameras).
    The ISP handles debayering RG10 → NV12 in hardware, then we convert to BGR
    for OpenCV.
    """
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
    """
    Open the CSI camera via nvarguscamerasrc GStreamer pipeline.
    Falls back to USB /dev/video0 if GStreamer pipeline fails.
    """
    # --- Try CSI camera via GStreamer (nvarguscamerasrc) ---
    gst = _gstreamer_pipeline()
    print(f"[camera] Trying GStreamer pipeline:\n  {gst}")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] CSI camera via nvarguscamerasrc {WIDTH}×{HEIGHT} OK")
        return cap
    print("[camera] GStreamer pipeline failed, trying USB fallback...")

    # --- Fallback: USB camera ---
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[camera] USB /dev/video0 {WIDTH}×{HEIGHT} OK")
        return cap

    raise RuntimeError("No camera found (CSI and USB both failed)")


# ── GPU HSV mask ──────────────────────────────────────────────────────────────
# Detect which cv2.cuda.cvtColor API is available at startup.
# OCV 4.5.x  → returns a new GpuMat  (functional style)
# OCV 4.6+   → writes into dst arg   (in-place style)
def _make_gpu_hsv_mask():
    """
    Factory: returns the correct gpu_hsv_mask implementation for this
    OpenCV build so the if-branch is paid once, not every frame.
    """
    # Probe with a 1×1 dummy to see which API works
    probe = cv2.cuda_GpuMat()
    probe.upload(np.zeros((1, 1, 3), dtype=np.uint8))
    try:
        result = cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2HSV)
        if result is not None and not result.empty():
            # Functional API (OCV 4.5.x) — cvtColor returns the output GpuMat
            print("[cuda] cvtColor: functional API (returns GpuMat)")
            def _mask_functional(roi_bgr, lo, hi):
                _gpu_frame.upload(roi_bgr)
                hsv_gpu = cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV)
                hsv_cpu = hsv_gpu.download()
                return cv2.inRange(hsv_cpu, lo, hi)
            return _mask_functional
    except Exception:
        pass

    # In-place API (OCV 4.6+) — cvtColor writes into dst
    print("[cuda] cvtColor: in-place API (dst arg)")
    def _mask_inplace(roi_bgr, lo, hi):
        _gpu_frame.upload(roi_bgr)
        cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV, _gpu_hsv)
        hsv_cpu = _gpu_hsv.download()
        return cv2.inRange(hsv_cpu, lo, hi)
    return _mask_inplace


def cpu_hsv_mask(roi_bgr, lo, hi):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lo, hi)


# Build the correct GPU mask fn for this OpenCV version (probed once at startup)
gpu_hsv_mask = _make_gpu_hsv_mask() if USE_CUDA else None


# ── Lane detection + PID ──────────────────────────────────────────────────────
def process_frame(frame, s, annotate: bool):
    h, w = frame.shape[:2]
    roi_top = int(h * ROI_FRAC)
    roi = frame[roi_top:h, :]

    lo = np.array([s["h_lo"], s["s_lo"], s["v_lo"]], dtype=np.uint8)
    hi = np.array([s["h_hi"], s["s_hi"], s["v_hi"]], dtype=np.uint8)

    # --- Mask: gpu_hsv_mask is None when CUDA unavailable ---
    mask = gpu_hsv_mask(roi, lo, hi) if gpu_hsv_mask is not None else cpu_hsv_mask(roi, lo, hi)

    # --- Morphological cleanup (small kernel for speed) ---
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # --- Contours (CPU — fast enough at 320×240) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lane_found, cx_lane, error = False, w // 2, 0.0

    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > s["min_contour_area"]:
            M = cv2.moments(big)
            if M["m00"] > 0:
                cx_lane    = int(M["m10"] / M["m00"])
                error      = (cx_lane - w // 2) / (w // 2)
                lane_found = True

    # --- PID ---
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

    # --- Annotate only when a browser is watching ---
    if annotate:
        annotated = frame.copy()
        cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_3ch[:, :, 0] = 0
        annotated[roi_top:h, :] = cv2.addWeighted(
            annotated[roi_top:h, :], 0.7, mask_3ch, 0.3, 0)

        if lane_found:
            cy = roi_top + (h - roi_top) // 2
            cv2.circle(annotated, (cx_lane, cy), 8, (0, 255, 0), -1)
            cv2.circle(annotated, (cx_lane, cy), 8, (255, 255, 255), 2)

        cv2.line(annotated, (w // 2, roi_top), (w // 2, h), (0, 200, 255), 1)

        arrow_x = int(w // 2 + steer * (w // 3))
        cv2.arrowedLine(annotated, (w // 2, 22), (arrow_x, 22),
                        (0, 140, 255), 2, tipLength=0.35)

        status = "DRIVING" if s["enabled"] else "STOPPED"
        color  = (0, 220, 60) if s["enabled"] else (60, 60, 220)
        cv2.putText(annotated, status,               (5, 18),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(annotated, f"e{error:+.2f} s{steer:+.2f}", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(annotated, f"fps {s['fps']}",    (5, 52),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        lane_col = (0, 220, 60) if lane_found else (0, 60, 220)
        cv2.putText(annotated, "OK" if lane_found else "NO",
                    (w - 28, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_col, 1)
    else:
        annotated = frame   # no copy needed

    return annotated, error, steer, lane_found


# ── Async JPEG encode ─────────────────────────────────────────────────────────
def _do_encode(img):
    print(f"[debug] Encoding image mean: {np.mean(img):.2f}")
    cv2.imwrite("debug_frame.jpg", img)
    print(f"[debug] Encoding JPEG: shape={img.shape}, dtype={img.dtype}")
    ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        print("[error] JPEG encoding failed!")
        return
    print(f"[debug] JPEG encoding success, size={len(jpeg.tobytes())} bytes")
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
            
        # Ensure frame is 3D (BGR)
        if len(frame.shape) != 3:
            print("[warn] Frame is not 3D, skipping.")
            continue
            
        if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with state_lock:
            s_copy = dict(state)

        with clients_lock:
            has_clients = stream_clients > 0

        # Annotate only if someone is watching; encode every Nth frame
        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        print(f"[debug] Frame mean: {np.mean(frame):.2f}")
        annotated, error, steer, lane_found = process_frame(frame, s_copy, do_annotate)

        # Async encode (fire-and-forget; previous encode may still be running — that's fine)
        if do_annotate:
            _encode_pool.submit(_do_encode, annotated)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        # Drive
        if s_copy["enabled"]:
            if lane_found:
                car.steer(steer)
                car.forward(s_copy["speed"])
            else:
                car.steer(0.0)
                car.forward(s_copy["speed"] * 0.5)
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
<title>JetRacer Lane Follower</title>
<style>
  :root {
    --bg: #0e1117; --surface: #161b27; --border: #2a3040;
    --accent: #00d4aa; --warn: #ffb020; --danger: #ff4d4d;
    --text: #e8ecf1; --muted: #6b7a99;
    --font: 'JetBrains Mono', 'Fira Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 1rem; }
  h1 { font-size: 1rem; letter-spacing: .15em; color: var(--accent);
       text-transform: uppercase; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 340px; gap: 1rem; width: 100%; max-width: 1100px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
  .card h2 { font-size: .7rem; letter-spacing: .12em; color: var(--muted); text-transform: uppercase;
             margin-bottom: .75rem; }
  img#feed { width: 100%; border-radius: 6px; display: block; background: #000; min-height: 180px; }
  .status-bar { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: .75rem; }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
  .slider-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .55rem; }
  .slider-row label { font-size: .7rem; color: var(--muted); width: 52px; flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val { font-size: .75rem; width: 40px; text-align: right; color: var(--text); }
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; }
  button { padding: .45rem 1.1rem; border: none; border-radius: 6px; cursor: pointer;
           font-family: var(--font); font-size: .8rem; font-weight: 600; letter-spacing: .04em; }
  #btn-go   { background: var(--accent); color: #061612; }
  #btn-stop { background: var(--danger); color: #fff; }
  #btn-go:hover   { filter: brightness(1.1); }
  #btn-stop:hover { filter: brightness(1.1); }
  .error-track { position: relative; height: 18px; background: var(--border);
                 border-radius: 9px; margin-top: .5rem; overflow: hidden; }
  #error-bar { position: absolute; height: 100%; width: 4px; background: var(--accent);
               left: 50%; transform: translateX(-50%); transition: left .1s; border-radius: 9px; }
  .divider { border: none; border-top: 1px solid var(--border); margin: .75rem 0; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>&#9675; JetRacer &#183; Lane Follow Dashboard</h1>
<div class="grid">
  <div class="card">
    <h2>Camera feed (annotated)</h2>
    <img id="feed" src="/video_feed" alt="camera">
    <div class="status-bar" style="margin-top:.75rem">
      <div class="stat"><span class="stat-val" id="v-fps">0</span><span class="stat-lbl">fps</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span><span class="stat-lbl">error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span><span class="stat-lbl">steer</span></div>
      <div class="stat"><span class="stat-val" id="v-lane">—</span><span class="stat-lbl">lane</span></div>
    </div>
    <div class="error-track" title="Lane error (centre = 0)">
      <div id="error-bar"></div>
    </div>
  </div>
  <div class="card">
    <h2>Drive</h2>
    <div class="slider-row">
      <label>Speed</label>
      <input type="range" id="speed" min="0" max="60" value="18" step="1">
      <span class="val" id="v-speed">0.18</span>
    </div>
    <div class="btn-row">
      <button id="btn-go"   onclick="setEnabled(true)">&#9654; GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
    </div>
    <hr class="divider">
    <h2>PID gains</h2>
    <div class="slider-row">
      <label>Kp</label>
      <input type="range" id="kp" min="0" max="1" value="0.4" step="0.01">
      <span class="val" id="v-kp">0.40</span>
    </div>
    <div class="slider-row">
      <label>Ki</label>
      <input type="range" id="ki" min="0" max="0.05" value="0.002" step="0.001">
      <span class="val" id="v-ki">0.002</span>
    </div>
    <div class="slider-row">
      <label>Kd</label>
      <input type="range" id="kd" min="0" max="0.5" value="0.15" step="0.01">
      <span class="val" id="v-kd">0.15</span>
    </div>
    <hr class="divider">
    <h2>HSV mask — hue</h2>
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
      <label>V hi</label>
      <input type="range" id="v_hi" min="0" max="255" value="255" step="1">
      <span class="val" id="v-v_hi">255</span>
    </div>
    <div class="slider-row">
      <label>V lo</label>
      <input type="range" id="v_lo" min="0" max="255" value="80" step="1">
      <span class="val" id="v-v_lo">80</span>
    </div>
    <hr class="divider">
    <h2>Min contour area</h2>
    <div class="slider-row">
      <label>Area</label>
      <input type="range" id="min_contour_area" min="50" max="5000" value="500" step="50">
      <span class="val" id="v-min_contour_area">500</span>
    </div>
  </div>
</div>
<script>
const sliders = ["speed","kp","ki","kd","h_lo","h_hi","s_lo","s_hi","v_lo","v_hi","min_contour_area"];
sliders.forEach(id => {
  const el = document.getElementById(id);
  const disp = document.getElementById("v-"+id);
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    disp.textContent = id === "speed" ? (v/100).toFixed(2) : (Number.isInteger(v) ? v : v.toFixed(3));
    sendParam(id, id === "speed" ? v/100 : v);
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
    # use_reloader=False is critical — reloader forks and doubles CPU load
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)