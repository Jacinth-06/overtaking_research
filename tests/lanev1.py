#!/usr/bin/env python3
"""
lane_follow.py — GPU-accelerated centre-lane follower + Flask dashboard
Target: Jetson Nano 4 GB  |  Straight-line road  |  Single left + right lane

Detection pipeline:
  frame → GPU grayscale → GPU Gaussian blur
  → CPU Canny + CPU Binary threshold → OR combine
  → morphology clean → find left & right lane edges
  → compute lane centre → error = centre - frame_mid → PID → steer

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

# ── CUDA availability check ───────────────────────────────────────────────────
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_CUDA:
    print("[init] CUDA device found — GPU path active")
    _gpu_frame  = cv2.cuda_GpuMat()
    _gpu_gray   = cv2.cuda_GpuMat()
    _gpu_blur   = cv2.cuda_GpuMat()
    # Persistent CUDA filter handles (created once, reused every frame)
    _cuda_gaussian = None   # built after we know ksize
else:
    print("[init] No CUDA device — falling back to CPU")
    _gpu_frame = _gpu_gray = _gpu_blur = None

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 320, 240
ENCODE_EVERY    = 3          # encode every Nth frame for MJPEG
JPEG_QUALITY    = 30
MJPEG_INTERVAL  = 1 / 15    # ~15 fps to browser

# ── Shared state (browser-tunable params + telemetry) ────────────────────────
state = {
    # Vision
    "canny_lo":      50,
    "canny_hi":      150,
    "binary_thresh": 200,
    "blur_ksize":    5,      # must be odd
    "morph_ksize":   5,      # must be odd
    "morph_iters":   2,
    "roi_top_frac":  0.55,   # discard top X% of frame (sky / far background)
    # PID
    "kp": 0.55,
    "ki": 0.003,
    "kd": 0.25,
    # Drive
    "speed":   0.15,
    "enabled": False,
    # Telemetry (read-only from browser)
    "error": 0.0,
    "steer": 0.0,
    "fps":   0,
    "lane_found": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()
_last_steer = 0.0           # hold-last steering when lane lost

# ── MJPEG state ───────────────────────────────────────────────────────────────
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
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[camera] CSI via nvarguscamerasrc {WIDTH}×{HEIGHT} OK")
        return cap
    print("[camera] GStreamer failed — trying USB /dev/video0")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[camera] USB /dev/video0 {WIDTH}×{HEIGHT} OK")
        return cap
    raise RuntimeError("No camera found (CSI and USB both failed)")


# ── GPU helpers ───────────────────────────────────────────────────────────────
def _probe_gpu_grayscale():
    """Return a callable bgr→gray (GPU) or None if CUDA cvtColor broken."""
    probe = cv2.cuda_GpuMat()
    probe.upload(np.zeros((1, 1, 3), dtype=np.uint8))
    # Try functional API first
    try:
        result = cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        if result is not None and not result.empty():
            def _gray(bgr):
                _gpu_frame.upload(bgr)
                return cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2GRAY).download()
            print("[cuda] cvtColor BGR→GRAY: functional API")
            return _gray
    except Exception:
        pass
    # Try in-place API
    try:
        cv2.cuda.cvtColor(probe, cv2.COLOR_BGR2GRAY, _gpu_gray)
        def _gray_ip(bgr):
            _gpu_frame.upload(bgr)
            cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2GRAY, _gpu_gray)
            return _gpu_gray.download()
        print("[cuda] cvtColor BGR→GRAY: in-place API")
        return _gray_ip
    except Exception:
        pass
    print("[cuda] cvtColor BGR→GRAY: unavailable — using CPU")
    return None

gpu_grayscale = _probe_gpu_grayscale() if USE_CUDA else None


def _make_cuda_gaussian(ksize: int):
    """Create (or recreate) a persistent CUDA Gaussian filter."""
    global _cuda_gaussian
    try:
        _cuda_gaussian = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (ksize, ksize), 0)
        return True
    except Exception:
        _cuda_gaussian = None
        return False

# Build with default ksize at startup
if USE_CUDA:
    _make_cuda_gaussian(state["blur_ksize"] | 1)

_last_blur_ksize = state["blur_ksize"] | 1


def gpu_blur(gray_cpu: np.ndarray, ksize: int) -> np.ndarray:
    """
    GPU Gaussian blur.  Recreates the filter only when ksize changes.
    Falls back to CPU if CUDA filter unavailable.
    """
    global _last_blur_ksize, _cuda_gaussian
    ksize = ksize | 1   # ensure odd
    if _cuda_gaussian is None or ksize != _last_blur_ksize:
        if not _make_cuda_gaussian(ksize):
            return cv2.GaussianBlur(gray_cpu, (ksize, ksize), 0)
        _last_blur_ksize = ksize
    _gpu_gray.upload(gray_cpu)
    _cuda_gaussian.apply(_gpu_gray, _gpu_blur)
    return _gpu_blur.download()


# ── Lane detection (centre of left+right edges) ───────────────────────────────
def find_lane_centre(binary_roi: np.ndarray) -> tuple[float | None, float, float]:
    """
    Given a binary ROI (white pixels = lane markings):
      - split frame into left half and right half
      - find the x-centroid of white pixels in each half
      - lane centre = average of left_x and right_x
    Returns: (centre_x or None, left_x, right_x)
             centre_x is None when fewer than one lane edge is found.
    """
    h, w = binary_roi.shape
    left_half  = binary_roi[:, :w // 2]
    right_half = binary_roi[:, w // 2:]

    left_pts  = np.column_stack(np.where(left_half  > 0))  # rows, cols
    right_pts = np.column_stack(np.where(right_half > 0))

    has_left  = len(left_pts)  > 30
    has_right = len(right_pts) > 30

    left_x  = float(np.mean(left_pts[:,  1]))           if has_left  else None
    right_x = float(np.mean(right_pts[:, 1])) + w // 2  if has_right else None

    if has_left and has_right:
        centre_x = (left_x + right_x) / 2.0
    elif has_left:
        centre_x = left_x + w * 0.25   # guess centre from left edge
    elif has_right:
        centre_x = right_x - w * 0.25  # guess centre from right edge
    else:
        centre_x = None

    return centre_x, left_x, right_x


# ── Full processing pipeline ───────────────────────────────────────────────────
def process_frame(frame: np.ndarray, s: dict, annotate: bool):
    global _last_steer
    h, w = frame.shape[:2]
    roi_top = int(h * s["roi_top_frac"])
    roi = frame[roi_top:, :]          # crop bottom portion of frame

    # 1. Grayscale  (GPU if available)
    if gpu_grayscale is not None:
        gray = gpu_grayscale(roi)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur  (GPU if available)
    blurred = gpu_blur(gray, s["blur_ksize"])

    # 3a. Canny edges
    edges = cv2.Canny(blurred, s["canny_lo"], s["canny_hi"])

    # 3b. Binary threshold (bright lane lines)
    _, binary = cv2.threshold(blurred, s["binary_thresh"], 255, cv2.THRESH_BINARY)

    # 4. Combine: keep pixels that are BOTH an edge AND bright
    #    (reduces noise from reflections / shadows)
    combined = cv2.bitwise_and(edges, binary)

    # 5. Morphology clean  (close small gaps, remove speckle)
    mk = s["morph_ksize"] | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel,
                                iterations=s["morph_iters"])
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel, iterations=1)

    # 6. Lane centre detection
    centre_x, left_x, right_x = find_lane_centre(cleaned)
    lane_found = centre_x is not None

    # 7. PID
    if lane_found:
        # Normalise error to [-1, 1]:  positive = car is left of centre
        error = (centre_x - w / 2.0) / (w / 2.0)

        now = time.time()
        dt  = max(now - pid_state["last_time"], 0.001)
        pid_state["integral"]  += error * dt
        pid_state["integral"]   = np.clip(pid_state["integral"], -1.0, 1.0)
        derivative              = (error - pid_state["last_error"]) / dt
        pid_state["last_error"] = error
        pid_state["last_time"]  = now

        steer = (s["kp"] * error
               + s["ki"] * pid_state["integral"]
               + s["kd"] * derivative)
        steer = float(np.clip(steer, -1.0, 1.0))
        _last_steer = steer
    else:
        error = 0.0
        steer = _last_steer   # hold last steer when lane lost

    # 8. Annotation (only when a browser client is connected)
    if annotate:
        annotated = frame.copy()
        cv2.line(annotated, (0, roi_top), (w, roi_top), (255, 220, 0), 1)

        # Overlay cleaned mask in green tint
        mask_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        mask_bgr[:, :, 0] = 0   # zero out B channel → green/red only
        mask_bgr[:, :, 2] = 0
        annotated[roi_top:, :] = cv2.addWeighted(
            annotated[roi_top:, :], 0.65, mask_bgr, 0.35, 0)

        # Draw detected edges
        rx = roi_top
        if left_x is not None:
            cv2.circle(annotated, (int(left_x),  rx + 10), 6, (0, 80, 255),  -1)
        if right_x is not None:
            cv2.circle(annotated, (int(right_x), rx + 10), 6, (255, 80, 0),  -1)
        if lane_found:
            cx = int(centre_x)
            cv2.circle(annotated, (cx, rx + 10), 8, (0, 255, 120), -1)
            cv2.line(annotated, (cx, rx), (cx, h), (0, 255, 120), 1)
        # Draw frame midline
        cv2.line(annotated, (w // 2, roi_top), (w // 2, h), (200, 200, 200), 1)
    else:
        annotated = frame

    return annotated, error, steer, lane_found


# ── Async JPEG encode ─────────────────────────────────────────────────────────
def _do_encode(img: np.ndarray):
    ret, jpeg = cv2.imencode(".jpg", img,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return
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
        if frame.ndim != 3:
            continue
        if frame.shape[:2] != (HEIGHT, WIDTH):
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        with state_lock:
            s = dict(state)

        with clients_lock:
            do_annotate = (stream_clients > 0) and (frame_idx % ENCODE_EVERY == 0)

        annotated, error, steer, lane_found = process_frame(frame, s, do_annotate)

        if do_annotate:
            _encode_pool.submit(_do_encode, annotated)

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with state_lock:
                state["fps"] = fps_counter
            fps_counter, fps_time = 0, time.time()

        if s["enabled"]:
            car.steer(steer)
            car.forward(s["speed"])
        else:
            car.stop()

        with state_lock:
            state["error"]      = round(error, 3)
            state["steer"]      = round(steer, 3)
            state["lane_found"] = lane_found

        frame_idx += 1

    cap.release()


# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lane Follower</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
  :root {
    --bg:#080c10; --surface:#0f1520; --border:#1e2a3a;
    --accent:#00e5b0; --danger:#ff4560; --warn:#ffd166;
    --text:#d4dce8; --muted:#4a5a72;
    --font:'IBM Plex Mono', monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--font);
       display:flex;flex-direction:column;align-items:center;
       min-height:100vh;padding:1.2rem}
  header{display:flex;align-items:center;gap:.75rem;margin-bottom:1.2rem;
         border-bottom:1px solid var(--border);padding-bottom:.8rem;width:100%;max-width:1100px}
  header h1{font-size:.8rem;letter-spacing:.2em;text-transform:uppercase;
             color:var(--accent)}
  #dot{width:8px;height:8px;border-radius:50%;background:var(--muted);
       transition:background .3s}
  #dot.live{background:var(--accent);box-shadow:0 0 8px var(--accent)}

  .grid{display:grid;grid-template-columns:1fr 340px;gap:1rem;
         width:100%;max-width:1100px}
  .card{background:var(--surface);border:1px solid var(--border);
        border-radius:8px;padding:1rem}
  .card h2{font-size:.62rem;letter-spacing:.14em;color:var(--muted);
            text-transform:uppercase;margin-bottom:.8rem}

  img#feed{width:100%;border-radius:4px;display:block;background:#000;
            min-height:200px;image-rendering:pixelated}

  /* Telemetry row */
  .tele{display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.9rem}
  .stat{display:flex;flex-direction:column}
  .stat-v{font-size:1.35rem;font-weight:600;color:var(--accent);
           transition:color .2s}
  .stat-l{font-size:.58rem;color:var(--muted);text-transform:uppercase;
           letter-spacing:.1em;margin-top:2px}

  /* Error bar */
  .ebar-wrap{position:relative;height:14px;background:var(--border);
              border-radius:7px;margin-top:.7rem;overflow:hidden}
  #ebar{position:absolute;height:100%;width:4px;background:var(--accent);
        left:50%;transform:translateX(-50%);transition:left .08s linear;
        border-radius:7px}

  /* Sliders */
  .s-row{display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem}
  .s-row label{font-size:.65rem;color:var(--muted);width:90px;flex-shrink:0}
  .s-row input[type=range]{flex:1;accent-color:var(--accent)}
  .s-row .val{font-size:.7rem;width:42px;text-align:right;color:var(--text)}

  /* Buttons */
  .btn-row{display:flex;gap:.5rem;margin-top:.75rem}
  button{padding:.4rem 1rem;border:none;border-radius:5px;cursor:pointer;
         font-family:var(--font);font-size:.78rem;font-weight:600;
         letter-spacing:.05em;transition:filter .15s}
  #btn-go  {background:var(--accent);color:#020d09}
  #btn-stop{background:var(--danger);color:#fff}
  button:hover{filter:brightness(1.12)}

  hr.div{border:none;border-top:1px solid var(--border);margin:.75rem 0}
  @media(max-width:700px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div id="dot"></div>
  <h1>JetRacer &mdash; Lane Follower</h1>
</header>
<div class="grid">
  <!-- LEFT: feed + telemetry -->
  <div class="card">
    <h2>Camera feed</h2>
    <img id="feed" src="/video_feed" alt="feed">
    <div class="tele">
      <div class="stat"><span class="stat-v" id="v-fps">0</span>
                        <span class="stat-l">fps</span></div>
      <div class="stat"><span class="stat-v" id="v-err">0.00</span>
                        <span class="stat-l">error</span></div>
      <div class="stat"><span class="stat-v" id="v-str">0.00</span>
                        <span class="stat-l">steer</span></div>
      <div class="stat"><span class="stat-v" id="v-lane">&mdash;</span>
                        <span class="stat-l">lane</span></div>
    </div>
    <div class="ebar-wrap" title="Lane error  (centre = 0)">
      <div id="ebar"></div>
    </div>
  </div>

  <!-- RIGHT: controls -->
  <div class="card">
    <h2>Drive</h2>
    <div class="s-row">
      <label>Speed</label>
      <input type="range" id="speed" min="0" max="60" value="15" step="1">
      <span class="val" id="v-speed">0.15</span>
    </div>
    <div class="btn-row">
      <button id="btn-go"   onclick="setEnabled(true)">&#9654; GO</button>
      <button id="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
    </div>

    <hr class="div">
    <h2>PID gains</h2>
    <div class="s-row"><label>Kp</label>
      <input type="range" id="kp" min="0" max="2"    value="0.55"  step="0.01">
      <span class="val" id="v-kp">0.55</span></div>
    <div class="s-row"><label>Ki</label>
      <input type="range" id="ki" min="0" max="0.05" value="0.003" step="0.001">
      <span class="val" id="v-ki">0.003</span></div>
    <div class="s-row"><label>Kd</label>
      <input type="range" id="kd" min="0" max="1"    value="0.25"  step="0.01">
      <span class="val" id="v-kd">0.25</span></div>

    <hr class="div">
    <h2>Vision pipeline</h2>
    <div class="s-row"><label>Canny lo</label>
      <input type="range" id="canny_lo" min="0" max="255" value="50" step="1">
      <span class="val" id="v-canny_lo">50</span></div>
    <div class="s-row"><label>Canny hi</label>
      <input type="range" id="canny_hi" min="0" max="255" value="150" step="1">
      <span class="val" id="v-canny_hi">150</span></div>
    <div class="s-row"><label>Binary thr</label>
      <input type="range" id="binary_thresh" min="0" max="255" value="200" step="1">
      <span class="val" id="v-binary_thresh">200</span></div>
    <div class="s-row"><label>Blur ksize</label>
      <input type="range" id="blur_ksize" min="1" max="21" value="5" step="2">
      <span class="val" id="v-blur_ksize">5</span></div>
    <div class="s-row"><label>Morph ksize</label>
      <input type="range" id="morph_ksize" min="1" max="21" value="5" step="2">
      <span class="val" id="v-morph_ksize">5</span></div>
    <div class="s-row"><label>Morph iters</label>
      <input type="range" id="morph_iters" min="1" max="5" value="2" step="1">
      <span class="val" id="v-morph_iters">2</span></div>

    <hr class="div">
    <h2>ROI</h2>
    <div class="s-row"><label>Top frac</label>
      <input type="range" id="roi_top_frac" min="0.1" max="0.9" value="0.55" step="0.05">
      <span class="val" id="v-roi_top_frac">0.55</span></div>
  </div>
</div>

<script>
const sliders = [
  "speed","kp","ki","kd",
  "canny_lo","canny_hi","binary_thresh","blur_ksize",
  "morph_ksize","morph_iters","roi_top_frac"
];
sliders.forEach(id => {
  const el   = document.getElementById(id);
  const disp = document.getElementById("v-" + id);
  if (!el) return;
  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    if (id === "speed") {
      disp.textContent = (v / 100).toFixed(2);
      send(id, v / 100);
    } else {
      disp.textContent = Number.isInteger(v) ? v : v.toFixed(3);
      send(id, v);
    }
  });
});

function send(key, value) {
  fetch("/set", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({[key]: value})
  });
}
function setEnabled(v) {
  send("enabled", v);
  document.getElementById("dot").classList.toggle("live", v);
}

async function poll() {
  try {
    const d = await (await fetch("/status")).json();
    document.getElementById("v-fps").textContent = d.fps;
    document.getElementById("v-err").textContent = d.error.toFixed(3);
    document.getElementById("v-str").textContent = d.steer.toFixed(3);

    const laneEl = document.getElementById("v-lane");
    laneEl.textContent  = d.lane_found ? "OK" : "LOST";
    laneEl.style.color  = d.lane_found ? "var(--accent)" : "var(--danger)";

    document.getElementById("dot").classList.toggle("live", d.enabled);

    const pct = (d.error + 1) / 2 * 100;
    const bar = document.getElementById("ebar");
    bar.style.left       = pct + "%";
    bar.style.background = Math.abs(d.error) > 0.5
                           ? "var(--danger)" : "var(--accent)";
  } catch(e) {}
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
    return Response(_generate_mjpeg(),
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
                if k in ("kp", "ki", "kd"):     # reset integrator on gain change
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