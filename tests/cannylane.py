"""
cannylane.py — Jetson-optimised dual-lane follower using Canny edges + Perspective Warp.
Structure based on line_follow.py with custom logic from user's request.
"""

import cv2
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Response, render_template_string, request, jsonify

from jetracer import JetRacer
import os

# Performance optimisations
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

app = Flask(__name__)

# ── Config constants ──────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240
ENCODE_EVERY    = 3
JPEG_QUALITY    = 30
MJPEG_INTERVAL  = 1 / 15

# Perspective Transform (scaled for 320x240)
# Original (approx 640x480): [[50, 480], [590, 480], [240, 300], [400, 300]]
SRC_POINTS = np.float32([[25, 240], [295, 240], [120, 150], [200, 150]])
DST_POINTS = np.float32([[75, 240], [245, 240], [75, 0], [245, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "speed": 0.46,
    "kp": 0.7, "ki": 0.0, "kd": 0.15,
    "enabled": False,
    "lane_width": 170, # scaled EXPECTED_LANE_WIDTH (340 -> 170)
    "canny_low": 50, "canny_high": 150,
    "tape_width_px": 7, # (14 -> 7)
    "confidence_threshold": 250, # scaled down for lower res
    
    # telemetry
    "error": 0.0, "steer": 0.0, "fps": 0, "lane_found": False
}

pid_state = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    raise RuntimeError("No camera found")

# ── Detection Logic ───────────────────────────────────────────────────────────
def find_tape_center(histogram, start_x, end_x, tape_w, thresh):
    best_center = None
    max_score = 0
    # Search within bounds
    start_x = int(max(0, start_x))
    end_x = int(min(len(histogram) - tape_w - 3, end_x))
    
    for x in range(start_x, end_x):
        score = histogram[x] + histogram[x + tape_w]
        if score > max_score and score > thresh:
            max_score = score
            best_center = x + (tape_w // 2)
    return best_center

def process_frame(frame, s, annotate: bool):
    global _last_steer
    h, w = frame.shape[:2]
    img_center = w // 2

    # 1. Edge Detection & Warp
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, s["canny_low"], s["canny_high"])
    
    # Perspectives
    warped = cv2.warpPerspective(edges, M, (w, h))

    # 2. Histogram Analysis
    # We look at the bottom 40% of the image
    roi_top = int(h * 0.6)
    hist = np.sum(warped[roi_top:, :], axis=0)
    
    # 3. Search Zones (Left side 35%, Right side from 65%)
    tape_w = int(s["tape_width_px"])
    thresh = s["confidence_threshold"]
    left_x = find_tape_center(hist, 0, int(w * 0.35), tape_w, thresh)
    right_x = find_tape_center(hist, int(w * 0.65), w - 1, tape_w, thresh)

    # 4. Lane Reconstruction
    l_valid, r_valid = left_x is not None, right_x is not None
    lane_width = s["lane_width"]
    
    if l_valid and r_valid:
        lane_center = (left_x + right_x) // 2
    elif l_valid:
        lane_center = left_x + (lane_width // 2)
        right_x = left_x + lane_width
    elif r_valid:
        lane_center = right_x - (lane_width // 2)
        left_x = right_x - lane_width
    else:
        # Totally lost
        lane_found = False
        return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), 0.0, _last_steer, False

    lane_found = True
    error = (lane_center - img_center) / (w // 2)

    # 5. PID Steering
    now = time.time()
    dt = max(now - pid_state["last_time"], 0.001)
    pid_state["integral"] += error * dt
    pid_state["integral"] = max(-1.0, min(1.0, pid_state["integral"]))
    derivative = (error - pid_state["last_error"]) / dt
    
    pid_state["last_error"] = error
    pid_state["last_time"] = now

    steer = (s["kp"] * error + s["ki"] * pid_state["integral"] + s["kd"] * derivative)
    steer = max(-1.0, min(1.0, steer))
    _last_steer = steer

    # 6. Annotation
    if annotate:
        viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 1)
        if l_valid: cv2.circle(viz, (left_x, h - 20), 10, (0, 255, 0), -1)
        if r_valid: cv2.circle(viz, (right_x, h - 20), 10, (0, 255, 0), -1)
        if lane_found:
            cv2.circle(viz, (lane_center, h - 40), 8, (255, 120, 0), -1)
        
        cv2.putText(viz, f"E{error:+.2f} S{steer:+.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return viz, error, steer, lane_found
    else:
        return frame, error, steer, lane_found

# ── Web UI ────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head><title>JetRacer Canny Lane</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg: #0b111a; --surface: #151d29; --accent: #00d4aa; --danger: #ff4d4d; --text: #e1e7ef; --muted: #6b7a90; --font: 'Inter', system-ui, sans-serif; }
  body { margin: 0; padding: 15px; background: var(--bg); color: var(--text); font-family: var(--font); }
  h1 { font-size: 1.1rem; color: var(--accent); margin-bottom: 1rem; letter-spacing: 0.5px; }
  .grid { display: grid; grid-template-columns: 1fr 320px; gap: 1rem; max-width: 1000px; margin: 0 auto; }
  .card { background: var(--surface); border: 1px solid #2a3444; border-radius: 12px; padding: 1rem; }
  img#feed { width: 100%; border-radius: 8px; background: #000; min-height: 200px; display: block; border: 1px solid #2a3444; }
  .status-bar { display: flex; gap: 1.5rem; margin-top: 1rem; justify-content: space-around; }
  .stat { display: flex; flex-direction: column; align-items: center; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; }
  .slider-row { display: flex; align-items: center; gap: .75rem; margin-bottom: .6rem; }
  .slider-row label { font-size: .7rem; width: 80px; color: var(--muted); }
  input[type=range] { flex: 1; accent-color: var(--accent); cursor: pointer; }
  .val { font-size: .75rem; width: 40px; text-align: right; }
  .btn-row { display: flex; gap: .75rem; margin-top: 1rem; }
  button { flex: 1; padding: .6rem; border: none; border-radius: 6px; font-weight: 700; cursor: pointer; color: #fff; transition: opacity 0.2s; }
  #btn-go { background: var(--accent); color: #000; }
  #btn-stop { background: var(--danger); }
  button:active { opacity: 0.8; }
</style>
</head>
<body>
<h1>&bull; JetRacer CannyLane Pipeline</h1>
<div class="grid">
  <div class="card">
    <img id="feed" src="/video_feed">
    <div class="status-bar">
      <div class="stat"><span class="stat-val" id="v-fps">0</span><span class="stat-lbl">FPS</span></div>
      <div class="stat"><span class="stat-val" id="v-err">0.00</span><span class="stat-lbl">Error</span></div>
      <div class="stat"><span class="stat-val" id="v-str">0.00</span><span class="stat-lbl">Steer</span></div>
      <div class="stat"><span class="stat-val" id="v-lane">-</span><span class="stat-lbl">Lane</span></div>
    </div>
  </div>
  <div class="card">
    <div class="btn-row">
      <button id="btn-go" onclick="setEnabled(true)">START</button>
      <button id="btn-stop" onclick="setEnabled(false)">STOP</button>
    </div>
    <hr style="border:0; border-top:1px solid #2a3444; margin:1.2rem 0">
    <div class="slider-row"><label>Speed</label><input type="range" id="speed" min="0" max="100" value="46"><span class="val" id="d-speed">0.46</span></div>
    <div class="slider-row"><label>Kp</label><input type="range" id="kp" min="0" max="2" step="0.01" value="0.7"><span class="val" id="d-kp">0.70</span></div>
    <div class="slider-row"><label>Kd</label><input type="range" id="kd" min="0" max="0.5" step="0.01" value="0.15"><span class="val" id="d-kd">0.15</span></div>
    <hr style="border:0; border-top:1px solid #2a3444; margin:1.2rem 0">
    <div class="slider-row"><label>Lane Width</label><input type="range" id="lane_width" min="50" max="300" value="170"><span class="val" id="d-lane_width">170</span></div>
    <div class="slider-row"><label>Canny Lo</label><input type="range" id="canny_low" min="0" max="255" value="50"><span class="val" id="d-canny_low">50</span></div>
    <div class="slider-row"><label>Canny Hi</label><input type="range" id="canny_high" min="0" max="255" value="150"><span class="val" id="d-canny_high">150</span></div>
    <div class="slider-row"><label>Threshold</label><input type="range" id="confidence_threshold" min="50" max="1000" value="250"><span class="val" id="d-confidence_threshold">250</span></div>
  </div>
</div>
<script>
const params = ["speed","kp","kd","lane_width","canny_low","canny_high","confidence_threshold"];
params.forEach(id => {
  const el = document.getElementById(id);
  const disp = document.getElementById("d-"+id);
  el.addEventListener("input", () => {
    let val = parseFloat(el.value);
    if(id === 'speed') val /= 100;
    disp.textContent = (id==='speed'||id==='kp'||id==='kd') ? val.toFixed(2) : val;
    fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({[id]: val})});
  });
});
function setEnabled(v) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled: v})}); }
async function poll() {
  try {
    const r = await fetch("/status"); const d = await r.json();
    document.getElementById("v-fps").textContent = d.fps;
    document.getElementById("v-err").textContent = d.error.toFixed(2);
    document.getElementById("v-str").textContent = d.steer.toFixed(2);
    document.getElementById("v-lane").textContent = d.lane_found ? "FIX" : "LOST";
    document.getElementById("v-lane").style.color = d.lane_found ? "#00d4aa" : "#ff4d4d";
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

def _do_encode(img):
    global latest_frame
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    with frame_lock: latest_frame = buf.tobytes()

# ── Control Loop ──────────────────────────────────────────────────────────────
def control_loop():
    cap = open_camera()
    car = JetRacer()
    frame_idx = 0
    t_last = time.time()
    frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.01); continue
        
        with state_lock: s = dict(state)
        
        has_clients = False
        with clients_lock: has_clients = stream_clients > 0
        do_annotate = has_clients and (frame_idx % ENCODE_EVERY == 0)
        
        viz, error, steer, lane_found = process_frame(frame, s, do_annotate)
        
        if do_annotate: _encode_pool.submit(_do_encode, viz)
        
        if s["enabled"]:
            car.steer(steer)
            car.forward(s["speed"])
        else:
            car.stop()
            
        with state_lock:
            state["error"], state["steer"], state["lane_found"] = error, steer, lane_found
            
        frames += 1
        now = time.time()
        if now - t_last >= 1.0:
            with state_lock: state["fps"] = frames
            frames, t_last = 0, now
        frame_idx += 1

if __name__ == "__main__":
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
