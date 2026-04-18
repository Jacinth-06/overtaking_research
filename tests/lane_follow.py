"""
lane_follow.py — GPU-accelerated dual-lane follower + Flask dashboard
Flow: Frame → Gaussian Blur → HLS + Sobel → Morphology → Lookahead → Width Check → Center Est. → Temporal Smoothing → Jump Filter → Curvature Awareness → PID
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

# ── CUDA availability check & Failsafe ─────────────────────────────────────────
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_CUDA:
    print("[init] CUDA device found — GPU path active")
    _gpu_frame = cv2.cuda_GpuMat()
    _gpu_hls   = cv2.cuda_GpuMat()
else:
    print("[init] No CUDA device — falling back to CPU")
    _gpu_frame = _gpu_hls = None

# ── Config constants ──────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240
ENCODE_EVERY    = 3
JPEG_QUALITY    = 30
MJPEG_INTERVAL  = 1 / 15
ROI_FRAC        = 0.65

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    # Masks
    "blur_k": 5, "sobel_min": 20, "sobel_max": 255,
    "h_lo": 0, "h_hi": 180,
    "l_lo": 0, "l_hi": 255,
    "s_lo": 40, "s_hi": 255,
    
    # PID & Drive
    "kp": 0.75, "ki": 0.003, "kd": 0.35, "speed": 0.46, "enabled": False,
    
    # Detection constraints
    "min_contour_area": 300,
    "lane_width": 325,
    "roi_side_limit": 0.0,
    
    # Telemetry
    "error": 0.0, "steer": 0.0, "fps": 0, "lane_found": False,
}

pid_state = {
    "integral": 0.0, 
    "last_error": 0.0, 
    "last_time": time.time(),
    "last_center": WIDTH // 2,
    "stable_center": WIDTH // 2
}
state_lock = threading.Lock()

_last_steer = 0.0
frame_lock = threading.Lock()
latest_frame = None
stream_clients = 0
clients_lock = threading.Lock()
_encode_pool = ThreadPoolExecutor(max_workers=1)

# ── Camera ────────────────────────────────────────────────────────────────────
def _gstreamer_pipeline(
    sensor_id=0, capture_width=1280, capture_height=720,
    display_width=WIDTH, display_height=HEIGHT, framerate=60, flip_method=0
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1"
    )

def open_camera():
    gst = _gstreamer_pipeline()
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened(): return cap
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        return cap
    raise RuntimeError("No camera found")

# ── Pipeline Algorithm ────────────────────────────────────────────────────────
def compute_mask(roi_bgr, s):
    # 1. Gaussian Blur (CPU acts as great standard)
    k = max(3, int(s["blur_k"]) | 1)
    blurred = cv2.GaussianBlur(roi_bgr, (k, k), 0)

    # 2. Convert to HLS via CUDA if available
    hls_cpu = None
    if USE_CUDA:
        try:
            _gpu_frame.upload(blurred)
            cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HLS, _gpu_hls)
            hls_cpu = _gpu_hls.download()
        except: pass
    if hls_cpu is None:
        hls_cpu = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        
    lower_hls = np.array([s["h_lo"], s["l_lo"], s["s_lo"]], dtype=np.uint8)
    upper_hls = np.array([s["h_hi"], s["l_hi"], s["s_hi"]], dtype=np.uint8)
    color_mask = cv2.inRange(hls_cpu, lower_hls, upper_hls)

    # 3. Sobel Edge Detection (L channel for best contrast)
    l_channel = hls_cpu[:, :, 1]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) or 1))
    
    sobel_mask = np.zeros_like(scaled_sobel)
    sobel_mask[(scaled_sobel >= s["sobel_min"]) & (scaled_sobel <= s["sobel_max"])] = 255

    # 4. HLS + Sobel combination
    return cv2.bitwise_or(color_mask, sobel_mask)


def process_frame(frame, s, annotate: bool):
    global _last_steer
    h, w = frame.shape[:2]
    
    # --- Lookahead detection (ROI cropping) ---
    roi_top = int(h * ROI_FRAC)
    x_start = int(w * s["roi_side_limit"])
    x_end   = w - x_start
    roi = frame[roi_top:h, x_start:x_end]

    # --- HLS + Sobel Mask ---
    mask = compute_mask(roi, s)

    # --- Morphology (open + close) ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    # 1. Pixel spreading boundary extraction
    nonzeros = cv2.findNonZero(mask)
    lane_width_px = s["lane_width"]
    
    l_det, r_det = False, False
    left_cx, right_cx = None, None
    left_cy = right_cy = mask.shape[0] // 2
    turn_severity = 0.0
    curve_diff = 0.0

    if nonzeros is not None:
        xs = nonzeros[:, 0, 0] + x_start
        leftmost = int(np.min(xs))
        rightmost = int(np.max(xs))
        pixel_spread = rightmost - leftmost

        # Lookahead shaping: Near vs Far curvature geometry
        h_roi = mask.shape[0]
        top_nz = cv2.findNonZero(mask[0:h_roi//2, :])
        bot_nz = cv2.findNonZero(mask[h_roi//2:, :])
        
        if top_nz is not None and bot_nz is not None:
            top_x = np.mean(top_nz[:, 0, 0])
            bot_x = np.mean(bot_nz[:, 0, 0])
            curve_diff = top_x - bot_x
            turn_severity = min(1.0, abs(curve_diff) / 80.0)

        # 2. Compute corridor
        if pixel_spread > lane_width_px * 0.5:
            l_det = r_det = True
            left_cx, right_cx = leftmost, rightmost
            raw_center = (leftmost + rightmost) // 2
            
            with state_lock:
                state["lane_width"] = int(lane_width_px * 0.95 + pixel_spread * 0.05)
        else:
            line_cx = (leftmost + rightmost) // 2
            if line_cx < pid_state["last_center"]:
                l_det = True
                left_cx = line_cx
                raw_center = line_cx + lane_width_px // 2
            else:
                r_det = True
                right_cx = line_cx
                raw_center = line_cx - lane_width_px // 2
                
        # 1. Bias target inward 
        # (left turn = negative diff => shift target slightly left, right turn => shift right)
        raw_center += int(turn_severity * 20 * np.sign(curve_diff))
    else:
        raw_center = w // 2

    lane_found = l_det or r_det

    # 3. Temporal smoothing & Jump filtering
    if lane_found:
        last_smooth = pid_state["stable_center"]
        if abs(raw_center - last_smooth) > 60:
            raw_center = last_smooth + int(np.sign(raw_center - last_smooth)) * 60

        smooth_center = int(0.6 * raw_center + 0.4 * pid_state["last_center"])
    else:
        smooth_center = pid_state["last_center"]
        
    pid_state["last_center"] = smooth_center
    pid_state["stable_center"] = smooth_center

    # 4. PID Following
    if lane_found:
        error = (smooth_center - w // 2) / (w // 2) *3
        now = time.time()
        dt = max(now - pid_state["last_time"], 0.001)
        pid_state["integral"] += error * dt
        pid_state["integral"] = max(-1.0, min(1.0, pid_state["integral"]))
        derivative = (error - pid_state["last_error"]) / dt

        pid_state["last_error"] = error
        pid_state["last_time"] = now

        effective_kp = s["kp"] * (1.0 - 0.4 * turn_severity)
        
        # 2. Penalize outward drift
        # Left curve (diff<0) and drifted right (center on left, error<0): boost gain
        # Right curve (diff>0) and drifted left (center on right, error>0): boost gain
        if curve_diff * error > 0:
            effective_kp *= 1.6  # Correct hard!
        
        steer = (effective_kp * error + s["ki"] * pid_state["integral"] + s["kd"] * derivative)
        steer = max(-1.0, min(1.0, steer))
        _last_steer = steer
    else:
        error = 0.0
        steer = _last_steer # Hold last steer precisely

    # --- Diagnostics Output ---
    if annotate:
        annotated = frame.copy()
        cv2.line(annotated, (x_start, roi_top), (x_end, roi_top), (0, 0, 255), 1)
        
        # Visualize Sobel/HLS mask cleanly on top left
        mask_debug = cv2.resize(mask, (w//3, h//3))
        annotated[0:h//3, 0:w//3] = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)

        if l_det: cv2.circle(annotated, (left_cx, roi_top + left_cy), 8, (255, 0, 255), 2)
        if r_det: cv2.circle(annotated, (right_cx, roi_top + right_cy), 8, (255, 255, 255), 2)

        if lane_found:
            cy_mid = roi_top + (h - roi_top) // 2
            cv2.circle(annotated, (smooth_center, cy_mid), 6, (0, 255, 255), -1) # smoothed center
            cv2.circle(annotated, (raw_center, cy_mid+10), 4, (0, 100, 255), -1) # raw geometry center

        cv2.line(annotated, (w//2, h), (w//2, h - 30), (0, 255, 0), 2)
        cv2.putText(annotated, f"e{error:+.2f} s{steer:+.2f}", (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        lane_col = (0, 220, 60) if lane_found else (0, 60, 220)
        cv2.putText(annotated, "OK" if lane_found else "NO", (w - 30, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_col, 1)
    else:
        annotated = frame

    return annotated, error, steer, lane_found


# ── Web UI ────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head><title>JetRacer Advanced Lane Follower</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg: #0b0f19; --card: #151b2b; --text: #f0f4f8; --accent: #00d4aa; --muted: #707b90; --border: #232c40; --danger: #ff4d4d; --font: 'Inter', system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 10px; background: var(--bg); color: var(--text); font-family: var(--font); }
  h1 { font-size: 1.2rem; margin: 0 0 1rem; color: var(--accent); letter-spacing: 0.5px; }
  h2 { font-size: 0.8rem; margin: 0 0 .5rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .grid { display: grid; grid-template-columns: minmax(auto, 480px) 300px; gap: 1rem; align-items: start; }
  .card { background: var(--card); padding: 1rem; border-radius: 12px; border: 1px solid var(--border); }
  #feed { width: 100%; border-radius: 8px; background: #000; aspect-ratio: 4/3; object-fit: cover; border: 1px solid var(--border); }
  .status-bar { display: flex; justify-content: space-between; background: #06090f; padding: .6rem; border-radius: 8px; border: 1px solid var(--border); }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; }
  .slider-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .55rem; }
  .slider-row label { font-size: .7rem; color: var(--muted); width: 65px; flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val { font-size: .75rem; width: 45px; text-align: right; color: var(--text); }
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; }
  button { padding: .45rem 1.1rem; border: none; border-radius: 6px; cursor: pointer; font-family: var(--font); font-size: .8rem; font-weight: 600; }
  #btn-go   { background: var(--accent); color: #061612; }
  #btn-stop { background: var(--danger); color: #fff; }
  .divider { border: none; border-top: 1px solid var(--border); margin: .75rem 0; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>&#9675; Advanced Lane Pipeline</h1>
<div class="grid">
  <div class="card">
    <img id="feed" src="/video_feed" alt="camera">
    <div class="status-bar" style="margin-top:.75rem">
      <div class="stat"><span class="stat-val" id="v-fps">0</span><span class="stat-lbl">fps</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span><span class="stat-lbl">error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span><span class="stat-lbl">steer</span></div>
      <div class="stat"><span class="stat-val" id="v-lane">—</span><span class="stat-lbl">line</span></div>
    </div>
  </div>
  <div class="card">
    <h2>Drive</h2>
    <div class="slider-row">
      <label>Speed</label>
      <input type="range" id="speed" min="0" max="60" value="46" step="1">
      <span class="val" id="v-speed">0.46</span>
    </div>
    <div class="btn-row">
      <button id="btn-go" onclick="setEnabled(true)">&#9654; GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
    </div>
    <hr class="divider">
    <h2>PID Tracking</h2>
    <div class="slider-row"><label>Kp</label><input type="range" id="kp" min="0" max="1" value="0.75" step="0.01"><span class="val" id="v-kp">0.75</span></div>
    <div class="slider-row"><label>Ki</label><input type="range" id="ki" min="0" max="0.05" value="0.003" step="0.001"><span class="val" id="v-ki">0.003</span></div>
    <div class="slider-row"><label>Kd</label><input type="range" id="kd" min="0" max="0.5" value="0.35" step="0.01"><span class="val" id="v-kd">0.35</span></div>
    <hr class="divider">
    <h2>Geometry & Filtering</h2>
    <div class="slider-row"><label>Lane W</label><input type="range" id="lane_width" min="50" max="700" value="325" step="5"><span class="val" id="v-lane_width">325</span></div>
    <div class="slider-row"><label>Blur K</label><input type="range" id="blur_k" min="1" max="15" value="5" step="2"><span class="val" id="v-blur_k">5</span></div>
    <div class="slider-row"><label>Side Crop</label><input type="range" id="roi_side_limit" min="0" max="0.45" value="0.0" step="0.01"><span class="val" id="v-roi_side_limit">0.0</span></div>
    <hr class="divider">
    <h2>HLS + Sobel Thresholds</h2>
    <div class="slider-row"><label>Sob Min</label><input type="range" id="sobel_min" min="0" max="255" value="20" step="1"><span class="val" id="v-sobel_min">20</span></div>
    <div class="slider-row"><label>Sob Max</label><input type="range" id="sobel_max" min="0" max="255" value="255" step="1"><span class="val" id="v-sobel_max">255</span></div>
    <div class="slider-row"><label>L lo</label><input type="range" id="l_lo" min="0" max="255" value="0" step="1"><span class="val" id="v-l_lo">0</span></div>
    <div class="slider-row"><label>L hi</label><input type="range" id="l_hi" min="0" max="255" value="255" step="1"><span class="val" id="v-l_hi">255</span></div>
    <div class="slider-row"><label>S lo</label><input type="range" id="s_lo" min="0" max="255" value="40" step="1"><span class="val" id="v-s_lo">40</span></div>
    <div class="slider-row"><label>S hi</label><input type="range" id="s_hi" min="0" max="255" value="255" step="1"><span class="val" id="v-s_hi">255</span></div>
  </div>
</div>
<script>
const sliders = ["speed","kp","ki","kd","blur_k","sobel_min","sobel_max","l_lo","l_hi","s_lo","s_hi","lane_width","roi_side_limit"];
sliders.forEach(id => {
  const el = document.getElementById(id);
  if(!el) return;
  const disp = document.getElementById("v-"+id);
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (id === "speed") { disp.textContent = (v/100).toFixed(2); sendParam(id, v/100); }
    else if (id === "roi_side_limit") { disp.textContent = v.toFixed(2); sendParam(id, v); }
    else { disp.textContent = Number.isInteger(v) ? v : v.toFixed(3); sendParam(id, v); }
  });
});
function sendParam(key, value) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({[key]: value})}); }
function setEnabled(v) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled: v})}); }
async function poll() {
  try {
    const r = await fetch("/status"); const d = await r.json();
    document.getElementById("v-fps").textContent = d.fps;
    document.getElementById("v-err").textContent = d.error.toFixed(2);
    document.getElementById("v-str").textContent = d.steer.toFixed(2);
    const laneEl = document.getElementById("v-lane");
    laneEl.textContent = d.lane_found ? "✓" : "✗";
    laneEl.style.color = d.lane_found ? "#00d4aa" : "#ff4d4d";
  } catch(e) {}
  setTimeout(poll, 250);
}
poll();
</script>
</body>
</html>
"""

@app.route("/")
def index(): return render_template_string(DASHBOARD_HTML)

def generate_mjpeg():
    global stream_clients
    with clients_lock: stream_clients += 1
    try:
        while True:
            with frame_lock: frame = latest_frame
            if frame is None:
                time.sleep(0.02); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(MJPEG_INTERVAL)
    finally:
        with clients_lock: stream_clients -= 1

@app.route("/video_feed")
def video_feed(): return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with state_lock: return jsonify({ "fps": state["fps"], "error": state["error"], "steer": state["steer"], "lane_found": state["lane_found"] })

@app.route("/set", methods=["POST"])
def set_param():
    data = request.json
    with state_lock:
        for k, v in data.items():
            if k in state: state[k] = v
    return jsonify({"status": "ok"})

def _do_encode(annotated_bgr):
    global latest_frame
    _, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    with frame_lock: latest_frame = buf.tobytes()

def control_loop():
    print("[control] Starting camera...")
    cap = open_camera()
    car = JetRacer()
    frame_idx = 0
    t_last = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[control] Failed to read frame!"); time.sleep(0.1); continue
        
        with state_lock: s_copy = dict(state)
        
        has_clients = False
        with clients_lock: has_clients = stream_clients > 0
        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        
        annotated, error, steer, lane_found = process_frame(frame, s_copy, do_annotate)
        
        if do_annotate: _encode_pool.submit(_do_encode, annotated)
        
        if s_copy["enabled"]:
            car.steer(steer)
            car.forward(s_copy["speed"] if lane_found else s_copy["speed"] * 0.5)
        else:
            car.stop()
            
        with state_lock:
            state["error"], state["steer"], state["lane_found"] = round(error, 3), round(steer, 3), lane_found
        
        frames += 1
        now = time.time()
        if now - t_last >= 1.0:
            with state_lock: state["fps"] = frames
            frames, t_last = 0, now
        frame_idx += 1

if __name__ == "__main__":
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()
    print("[flask] Starting UI on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)