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
print(f"[debug] OpenCV version: {cv2.__version__}")
try:
    _USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if not _USE_GPU:
        print("[debug] CUDA device count is 0")
except Exception as e:
    _USE_GPU = False
    print(f"[debug] CUDA detection failed: {e}")

if _USE_GPU:
    print("[gpu] CUDA device found — GPU-accelerated pipeline active")
    # ... rest of GPU setup ...
# (lines continue below, I will replace the whole block)


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
