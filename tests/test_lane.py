"""
lane_follow.py — GPU-accelerated lane following + Flask live dashboard
Optimised for Jetson Nano 4 GB (CUDA cv2.cuda pipeline, low CPU footprint).
Run with:  python test_lane.py
Open browser at:  http://<jetson-ip>:5000
"""

import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template_string, request, jsonify

# ── Import your JetRacer class ────────────────────────────────────────────────
from jetracer import JetRacer

app = Flask(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
_CAM_W, _CAM_H = 320, 240          # halved from 640×480 → 4× fewer pixels
_JPEG_QUALITY   = 30                # lower = smaller + faster encode
_STREAM_FPS_CAP = 8                 # dashboard doesn't need 30 fps
_MORPH_KERNEL   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ── Detect CUDA availability ─────────────────────────────────────────────────
try:
    _USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
except (AttributeError, cv2.error):
    _USE_GPU = False

if _USE_GPU:
    print("[gpu] CUDA device found — GPU-accelerated pipeline active")
    # Pre-allocate GpuMat buffers to avoid per-frame GPU malloc
    _gpu_src     = cv2.cuda_GpuMat()
    _gpu_hsv     = cv2.cuda_GpuMat()
    _gpu_mask    = cv2.cuda_GpuMat()
    _gpu_tmp     = cv2.cuda_GpuMat()
    _gpu_ch_h    = cv2.cuda_GpuMat()
    _gpu_ch_s    = cv2.cuda_GpuMat()
    _gpu_ch_v    = cv2.cuda_GpuMat()
    _gpu_m0      = cv2.cuda_GpuMat()
    _gpu_m1      = cv2.cuda_GpuMat()
    _gpu_m2      = cv2.cuda_GpuMat()
    _gpu_m3      = cv2.cuda_GpuMat()
    _gpu_m4      = cv2.cuda_GpuMat()
    _gpu_m5      = cv2.cuda_GpuMat()
    # CUDA morphology filter — created once, reused every frame
    _gpu_morph = cv2.cuda.createMorphologyFilter(
        cv2.MORPH_CLOSE, cv2.CV_8UC1, _MORPH_KERNEL)
    # CUDA stream for async pipelining
    _gpu_stream = cv2.cuda.Stream()
else:
    print("[gpu] No CUDA — falling back to CPU-only pipeline")

# ── Global shared state ───────────────────────────────────────────────────────
state = {
    # HSV lane colour thresholds (tune in browser)
    "h_lo": 20,  "h_hi": 35,   # hue  (20-35 = yellow lane markings)
    "s_lo": 80,  "s_hi": 255,
    "v_lo": 80,  "v_hi": 255,

    # PID gains
    "kp": 0.4,
    "ki": 0.002,
    "kd": 0.15,

    # Drive settings
    "speed": 0.18,         # forward throttle 0–1
    "enabled": False,       # master enable/disable
    "min_contour_area": 500,  # smaller because image is 320×240 now

    # Runtime (read-only from browser)
    "error":      0.0,
    "steer":      0.0,
    "fps":        0,
    "lane_found": False,
}

# PID integral / last-error accumulators (protected by lock)
pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()

# Latest annotated frame for MJPEG stream
frame_lock    = threading.Lock()
latest_frame  = None   # raw bytes (JPEG)
_has_clients  = False  # set True while a browser is streaming


# ── Camera setup ─────────────────────────────────────────────────────────────
def open_camera():
    """Try GStreamer CSI pipeline (Jetson, GPU-accelerated), fall back to USB."""
    # nvvidconv does scaling + colour conversion on GPU.
    gst = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw,width={_CAM_W},height={_CAM_H},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[camera] CSI GStreamer pipeline OK")
        return cap

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_H)
        print("[camera] USB /dev/video0 OK")
        return cap

    raise RuntimeError("No camera found")


# ── GPU-accelerated lane detection ────────────────────────────────────────────
def detect_lane_gpu(roi, w, hsv_lo, hsv_hi, min_area):
    """
    Runs cvtColor, inRange (via split+threshold+AND), and morphology on GPU.
    Downloads only the final single-channel mask to CPU for contour finding.
    Returns (error, lane_found, cx_lane).
    """
    # Upload ROI to GPU (one 320×96×3 block ≈ 90 KB)
    _gpu_src.upload(roi, stream=_gpu_stream)

    # BGR → HSV entirely on GPU
    cv2.cuda.cvtColor(_gpu_src, cv2.COLOR_BGR2HSV, _gpu_hsv, stream=_gpu_stream)

    # Split HSV channels on GPU
    channels = cv2.cuda.split(_gpu_hsv, stream=_gpu_stream)  # [H, S, V] GpuMats
    _gpu_ch_h = channels[0]
    _gpu_ch_s = channels[1]
    _gpu_ch_v = channels[2]

    # --- GPU inRange via per-channel threshold + bitwise AND ---
    # H channel:  h_lo <= H <= h_hi
    cv2.cuda.threshold(_gpu_ch_h, float(hsv_lo[0]), 255,
                       cv2.THRESH_BINARY, dst=_gpu_m0, stream=_gpu_stream)
    cv2.cuda.threshold(_gpu_ch_h, float(hsv_hi[0]), 255,
                       cv2.THRESH_BINARY_INV, dst=_gpu_m1, stream=_gpu_stream)
    # S channel:  s_lo <= S <= s_hi
    cv2.cuda.threshold(_gpu_ch_s, float(hsv_lo[1]), 255,
                       cv2.THRESH_BINARY, dst=_gpu_m2, stream=_gpu_stream)
    cv2.cuda.threshold(_gpu_ch_s, float(hsv_hi[1]), 255,
                       cv2.THRESH_BINARY_INV, dst=_gpu_m3, stream=_gpu_stream)
    # V channel:  v_lo <= V <= v_hi
    cv2.cuda.threshold(_gpu_ch_v, float(hsv_lo[2]), 255,
                       cv2.THRESH_BINARY, dst=_gpu_m4, stream=_gpu_stream)
    cv2.cuda.threshold(_gpu_ch_v, float(hsv_hi[2]), 255,
                       cv2.THRESH_BINARY_INV, dst=_gpu_m5, stream=_gpu_stream)

    # Combine all 6 masks:  mask = m0 & m1 & m2 & m3 & m4 & m5
    cv2.cuda.bitwise_and(_gpu_m0, _gpu_m1, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m2, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m3, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m4, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m5, _gpu_mask, stream=_gpu_stream)

    # Morphological close on GPU (fills gaps in lane markings)
    _gpu_morph.apply(_gpu_mask, _gpu_mask, stream=_gpu_stream)

    # Synchronise stream before downloading
    _gpu_stream.waitForCompletion()

    # Download mask to CPU (single-channel 320×96 ≈ 30 KB — very cheap)
    mask = _gpu_mask.download()

    # --- Contour finding (CPU only — no CUDA equivalent) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > min_area:
            M = cv2.moments(big)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                error = (cx - w // 2) / (w // 2)
                return error, True, cx

    return 0.0, False, w // 2


# ── CPU-only lane detection (fallback) ────────────────────────────────────────
def detect_lane_cpu(roi, w, hsv_lo, hsv_hi, min_area):
    """
    Pure-CPU fallback for non-CUDA OpenCV builds.
    Returns (error, lane_found, cx_lane).
    """
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > min_area:
            M = cv2.moments(big)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                error = (cx - w // 2) / (w // 2)
                return error, True, cx

    return 0.0, False, w // 2


# Select the right detection function once at import time
_detect_lane = detect_lane_gpu if _USE_GPU else detect_lane_cpu


# ── Cheap annotator (only runs when dashboard is open) ────────────────────────
_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]

def annotate_and_encode(frame, error, steer, lane_found, cx_lane, enabled, fps):
    """Draw minimal HUD. Modifies frame in-place (no copy)."""
    h, w = frame.shape[:2]
    roi_top = int(h * 0.6)

    # ROI line
    cv2.line(frame, (0, roi_top), (w, roi_top), (255, 255, 0), 1)

    # Centre reference
    cv2.line(frame, (w // 2, roi_top), (w // 2, h), (0, 200, 255), 1)

    # Lane centroid
    if lane_found:
        cy = roi_top + (h - roi_top) // 2
        cv2.circle(frame, (cx_lane, cy), 6, (0, 255, 0), -1)

    # Compact HUD
    status = "GO" if enabled else "STOP"
    colour = (0, 220, 60) if enabled else (60, 60, 220)
    cv2.putText(frame, f"{status} e:{error:+.1f} s:{steer:+.1f} {fps}fps",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

    _, buf = cv2.imencode(".jpg", frame, _ENCODE_PARAMS)
    return buf.tobytes()


# ── Control loop thread ───────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    global latest_frame
    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    fps_val = 0
    frame_tick = 0            # counts frames for stream-skip

    print("[loop] Starting control loop …")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        h, w = frame.shape[:2]
        roi_top = int(h * 0.6)
        roi = frame[roi_top:, :]

        # ── Snapshot tunables under lock ──────────────────────────────────
        with state_lock:
            kp, ki, kd = state["kp"], state["ki"], state["kd"]
            speed       = state["speed"]
            enabled     = state["enabled"]
            min_area    = state["min_contour_area"]
            hsv_lo = np.array([state["h_lo"], state["s_lo"], state["v_lo"]],
                              dtype=np.uint8)
            hsv_hi = np.array([state["h_hi"], state["s_hi"], state["v_hi"]],
                              dtype=np.uint8)

        # ── Detection (GPU or CPU) ───────────────────────────────────────
        error, lane_found, cx_lane = _detect_lane(roi, w, hsv_lo, hsv_hi, min_area)

        # ── PID ──────────────────────────────────────────────────────────
        now = time.time()
        dt  = max(now - pid_state["last_time"], 0.001)

        pid_state["integral"]  += error * dt
        pid_state["integral"]   = max(-1.0, min(1.0, pid_state["integral"]))
        derivative              = (error - pid_state["last_error"]) / dt
        pid_state["last_error"] = error
        pid_state["last_time"]  = now

        steer = kp * error + ki * pid_state["integral"] + kd * derivative
        steer = max(-1.0, min(1.0, steer))

        # ── Drive ────────────────────────────────────────────────────────
        if enabled:
            car.steer(steer)
            car.forward(speed if lane_found else speed * 0.5)
        else:
            car.stop()

        # ── FPS counter ──────────────────────────────────────────────────
        fps_counter += 1
        if now - fps_time >= 1.0:
            fps_val = fps_counter
            fps_counter = 0
            fps_time = now

        # ── Publish to state (cheap dict writes) ─────────────────────────
        with state_lock:
            state["error"]      = round(error, 3)
            state["steer"]      = round(steer, 3)
            state["lane_found"] = lane_found
            state["fps"]        = fps_val

        # ── Stream annotation (skip frames if nobody is watching) ────────
        frame_tick += 1
        if _has_clients and (frame_tick % 3 == 0):   # annotate every 3rd frame
            jpeg_bytes = annotate_and_encode(
                frame, error, steer, lane_found, cx_lane, enabled, fps_val)
            with frame_lock:
                latest_frame = jpeg_bytes

    cap.release()


# ── Flask routes ──────────────────────────────────────────────────────────────
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
  img#feed { width: 100%; border-radius: 6px; display: block; background: #000; min-height: 240px; }

  /* Status bar */
  .status-bar { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: .75rem; }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }

  /* Sliders */
  .slider-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .55rem; }
  .slider-row label { font-size: .7rem; color: var(--muted); width: 52px; flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val { font-size: .75rem; width: 40px; text-align: right; color: var(--text); }

  /* Buttons */
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; }
  button { padding: .45rem 1.1rem; border: none; border-radius: 6px; cursor: pointer;
           font-family: var(--font); font-size: .8rem; font-weight: 600; letter-spacing: .04em; }
  #btn-go   { background: var(--accent); color: #061612; }
  #btn-stop { background: var(--danger); color: #fff; }
  #btn-go:hover   { filter: brightness(1.1); }
  #btn-stop:hover { filter: brightness(1.1); }

  /* Error bar */
  .error-track { position: relative; height: 18px; background: var(--border); border-radius: 9px; margin-top: .5rem; overflow: hidden; }
  #error-bar { position: absolute; height: 100%; width: 4px; background: var(--accent);
               left: 50%; transform: translateX(-50%); transition: left .1s; border-radius: 9px; }

  /* Section divider */
  .divider { border: none; border-top: 1px solid var(--border); margin: .75rem 0; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>&#9675; JetRacer &#183; Lane Follow Dashboard</h1>
<div class="grid">

  <!-- Camera feed -->
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

  <!-- Controls -->
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
    <h2>HSV mask — hue (lane colour)</h2>
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
// ── Slider live update ────────────────────────────────────────────────────────
const sliders = ["speed","kp","ki","kd","h_lo","h_hi","s_lo","v_lo","min_contour_area"];
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

// ── Telemetry polling ─────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    document.getElementById("v-fps").textContent  = d.fps;
    document.getElementById("v-err").textContent  = d.error.toFixed(2);
    document.getElementById("v-str").textContent  = d.steer.toFixed(2);
    document.getElementById("v-lane").textContent = d.lane_found ? "✓" : "✗";
    document.getElementById("v-lane").style.color = d.lane_found ? "#00d4aa" : "#ff4d4d";
    // Error bar: map -1…+1 to 0%…100%
    const pct = (d.error + 1) / 2 * 100;
    document.getElementById("error-bar").style.left = pct + "%";
    document.getElementById("error-bar").style.background = Math.abs(d.error) > 0.5 ? "#ff4d4d" : "#00d4aa";
  } catch(e) {}
  setTimeout(poll, 300);
}
poll();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def generate_mjpeg():
    """MJPEG generator — yields JPEG frames wrapped in multipart boundary."""
    global _has_clients
    _has_clients = True
    interval = 1.0 / _STREAM_FPS_CAP
    try:
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(interval)
    finally:
        _has_clients = False


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
                # Reset PID integrator whenever gains change
                if k in ("kp", "ki", "kd"):
                    pid_state["integral"]  = 0.0
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
