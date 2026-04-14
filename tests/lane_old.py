"""
lane_follow.py — OpenCV lane following + Flask live dashboard
Run with:  python lane_follow.py
Open browser at:  http://<jetson-ip>:5000
"""

import cv2
import numpy as np
import threading
import time
import os
from flask import Flask, Response, render_template_string, request, jsonify

# ── Import your JetRacer class ────────────────────────────────────────────────
from jetracer import JetRacer

app = Flask(__name__)

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
    "min_contour_area": 2000,

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
frame_lock   = threading.Lock()
latest_frame = None   # raw bytes (JPEG)


# ── Camera setup ─────────────────────────────────────────────────────────────
def open_camera():
    """Try GStreamer CSI pipeline (Jetson), fall back to /dev/video0."""
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


# ── Lane detection & PID ──────────────────────────────────────────────────────
def process_frame(frame, s):
    """
    1. Crop to bottom ROI
    2. HSV threshold → mask
    3. Find largest contour → centroid → error
    4. PID → steer value
    Returns (annotated_frame, error, steer, lane_found)
    """
    h, w = frame.shape[:2]

    # --- ROI: bottom 40% of frame ---
    roi_top = int(h * 0.60)
    roi = frame[roi_top:h, :]

    # --- HSV mask ---
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo   = np.array([s["h_lo"], s["s_lo"], s["v_lo"]])
    hi   = np.array([s["h_hi"], s["s_hi"], s["v_hi"]])
    mask = cv2.inRange(hsv, lo, hi)

    # --- Morphological cleanup ---
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # --- Contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lane_found  = False
    cx_lane     = w // 2
    error       = 0.0

    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > s["min_contour_area"]:
            M = cv2.moments(big)
            if M["m00"] > 0:
                cx_lane    = int(M["m10"] / M["m00"])
                error      = (cx_lane - w // 2) / (w // 2)   # -1 … +1
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

    # --- Annotate frame ---
    annotated = frame.copy()

    # ROI boundary line
    cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 255, 0), 1)

    # Coloured mask overlay in ROI region
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_3ch[:, :, 0] = 0   # zero blue → green channel highlight
    annotated[roi_top:h, :] = cv2.addWeighted(
        annotated[roi_top:h, :], 0.7, mask_3ch, 0.3, 0
    )

    # Lane centroid marker
    if lane_found:
        cv2.circle(annotated, (cx_lane, roi_top + (h - roi_top) // 2), 12, (0, 255, 0), -1)
        cv2.circle(annotated, (cx_lane, roi_top + (h - roi_top) // 2), 12, (255, 255, 255), 2)

    # Frame centre reference
    cv2.line(annotated, (w // 2, roi_top), (w // 2, h), (0, 200, 255), 1)

    # Steering arrow
    arrow_x = int(w // 2 + steer * (w // 3))
    cv2.arrowedLine(annotated, (w // 2, 30), (arrow_x, 30), (0, 140, 255), 3, tipLength=0.35)

    # HUD text
    status = "DRIVING" if s["enabled"] else "STOPPED"
    color  = (0, 220, 60) if s["enabled"] else (60, 60, 220)
    cv2.putText(annotated, status,          (10, 24),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(annotated, f"err {error:+.2f}", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(annotated, f"str {steer:+.2f}", (10, 72),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(annotated, f"fps {s['fps']}",   (10, 94),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    lane_txt = "lane OK" if lane_found else "NO LANE"
    lane_col = (0, 220, 60) if lane_found else (0, 60, 220)
    cv2.putText(annotated, lane_txt, (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lane_col, 2)

    return annotated, error, steer, lane_found


# ── Control loop thread ───────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    global latest_frame
    cap = open_camera()
    fps_counter, fps_time = 0, time.time()

    print("[loop] Starting control loop …")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        with state_lock:
            s_copy = dict(state)

        annotated, error, steer, lane_found = process_frame(frame, s_copy)

        # FPS
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
                car.forward(s_copy["speed"] * 0.5)   # slow, straight
        else:
            car.stop()

        with state_lock:
            state["error"]      = round(error, 3)
            state["steer"]      = round(steer, 3)
            state["lane_found"] = lane_found

        # Encode and publish frame
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 40])
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
      <input type="range" id="min_contour_area" min="200" max="20000" value="2000" step="100">
      <span class="val" id="v-min_contour_area">2000</span>
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
  setTimeout(poll, 200);
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
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.1)   # ~30 fps cap


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


