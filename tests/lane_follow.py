"""
lane_follow.py — Lightweight GPU-accelerated lane following
Optimised for Jetson Nano 4 GB.
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
_CAM_W, _CAM_H = 320, 240
_JPEG_QUALITY   = 30
_STREAM_FPS_CAP = 10
_MORPH_KERNEL   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ── Detect CUDA availability & Diagnostics ───────────────────────────────────
print(f"[debug] OpenCV version: {cv2.__version__}")
try:
    _USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if not _USE_GPU:
        print("[debug] CUDA device count is 0. Check if OpenCV was built with CUDA.")
except Exception as e:
    _USE_GPU = False
    print(f"[debug] CUDA detection failed: {e}")

if _USE_GPU:
    print("[gpu] CUDA device found — GPU-accelerated pipeline active")
    _gpu_src = cv2.cuda_GpuMat()
    _gpu_hsv = cv2.cuda_GpuMat()
    _gpu_mask = cv2.cuda_GpuMat()
    _gpu_m0 = cv2.cuda_GpuMat(); _gpu_m1 = cv2.cuda_GpuMat()
    _gpu_m2 = cv2.cuda_GpuMat(); _gpu_m3 = cv2.cuda_GpuMat()
    _gpu_m4 = cv2.cuda_GpuMat(); _gpu_m5 = cv2.cuda_GpuMat()
    _gpu_morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, _MORPH_KERNEL)
    _gpu_stream = cv2.cuda.Stream()
else:
    print("[gpu] No CUDA — falling back to CPU-only pipeline")

# ── Global shared state ───────────────────────────────────────────────────────
state = {
    "h_lo": 20, "h_hi": 35,
    "s_lo": 80, "s_hi": 255,
    "v_lo": 80, "v_hi": 255,
    "kp": 0.4, "ki": 0.002, "kd": 0.15,
    "speed": 0.25, "enabled": False, "min_contour_area": 500,  # speed up to 0.25 from 0.18
    "error": 0.0, "steer": 0.0, "fps": 0, "lane_found": False,
}

pid_state  = {"integral": 0.0, "last_error": 0.0, "last_time": time.time()}
state_lock = threading.Lock()
frame_lock = threading.Lock()
latest_frame = None
_has_clients = False

# ── Camera setup ─────────────────────────────────────────────────────────────
def open_camera():
    """Optimised GStreamer string for IMX219 (Jetson Nano)."""
    # Unrestricted framerate to avoid locking nvarguscamerasrc exposure adjustments
    gst = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=1280,height=720 ! "
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_H)
        print("[camera] USB /dev/video0 OK")
        return cap
    raise RuntimeError("No camera found")

# ── Lane detection (GPU/CPU) ──────────────────────────────────────────────────
def detect_lane_gpu(roi, w, hsv_lo, hsv_hi, min_area):
    _gpu_src.upload(roi, stream=_gpu_stream)
    cv2.cuda.cvtColor(_gpu_src, cv2.COLOR_BGR2HSV, _gpu_hsv, stream=_gpu_stream)
    channels = cv2.cuda.split(_gpu_hsv, stream=_gpu_stream)
    
    cv2.cuda.threshold(channels[0], float(hsv_lo[0]), 255, cv2.THRESH_BINARY, dst=_gpu_m0, stream=_gpu_stream)
    cv2.cuda.threshold(channels[0], float(hsv_hi[0]), 255, cv2.THRESH_BINARY_INV, dst=_gpu_m1, stream=_gpu_stream)
    cv2.cuda.threshold(channels[1], float(hsv_lo[1]), 255, cv2.THRESH_BINARY, dst=_gpu_m2, stream=_gpu_stream)
    cv2.cuda.threshold(channels[1], float(hsv_hi[1]), 255, cv2.THRESH_BINARY_INV, dst=_gpu_m3, stream=_gpu_stream)
    cv2.cuda.threshold(channels[2], float(hsv_lo[2]), 255, cv2.THRESH_BINARY, dst=_gpu_m4, stream=_gpu_stream)
    cv2.cuda.threshold(channels[2], float(hsv_hi[2]), 255, cv2.THRESH_BINARY_INV, dst=_gpu_m5, stream=_gpu_stream)
    
    cv2.cuda.bitwise_and(_gpu_m0, _gpu_m1, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m2, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m3, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m4, _gpu_mask, stream=_gpu_stream)
    cv2.cuda.bitwise_and(_gpu_mask, _gpu_m5, _gpu_mask, stream=_gpu_stream)
    
    _gpu_morph.apply(_gpu_mask, _gpu_mask, stream=_gpu_stream)
    _gpu_stream.waitForCompletion()
    mask = _gpu_mask.download()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > min_area:
            M = cv2.moments(big)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                error = (cx - w // 2) / (w // 2)
                return error, True, cx
    return 0.0, False, w // 2

def detect_lane_cpu(roi, w, hsv_lo, hsv_hi, min_area):
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > min_area:
            M = cv2.moments(big)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                error = (cx - w // 2) / (w // 2)
                return error, True, cx
    return 0.0, False, w // 2

_detect_lane = detect_lane_gpu if _USE_GPU else detect_lane_cpu

# ── Control loop ─────────────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    global latest_frame
    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    last_enable_state = False
    
    print("[loop] Starting control loop …")
    print("[loop] Auto-drive is currently DISABLED. Open the dashboard and click 'GO' to drive.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        
        h, w = frame.shape[:2]
        roi = frame[int(h*0.6):, :]
        
        with state_lock:
            s = dict(state)
            hsv_lo = np.array([s["h_lo"], s["s_lo"], s["v_lo"]], dtype=np.uint8)
            hsv_hi = np.array([s["h_hi"], s["s_hi"], s["v_hi"]], dtype=np.uint8)

        error, lane_found, cx_lane = _detect_lane(roi, w, hsv_lo, hsv_hi, s["min_contour_area"])
        
        now = time.time()
        dt = max(now - pid_state["last_time"], 0.001)
        pid_state["integral"] = max(-1.0, min(1.0, pid_state["integral"] + error * dt))
        derivative = (error - pid_state["last_error"]) / dt
        pid_state["last_error"] = error
        pid_state["last_time"] = now
        steer = max(-1.0, min(1.0, s["kp"] * error + s["ki"] * pid_state["integral"] + s["kd"] * derivative))

        # Drive control and smart 'Stopped' logging
        if s["enabled"]:
            car.steer(steer)
            car.forward(s["speed"] if lane_found else s["speed"] * 0.5)
            last_enable_state = True
        elif last_enable_state:
            # Only call stop once when transitioning to disabled
            car.stop()
            last_enable_state = False
            print("[loop] Auto-drive disabled. Waiting for 'GO' from dashboard...")

        fps_counter += 1
        if now - fps_time >= 1.0:
            with state_lock: state["fps"] = fps_counter
            fps_counter, fps_time = 0, now
            
        with state_lock:
            state["error"], state["steer"], state["lane_found"] = round(error, 2), round(steer, 2), lane_found

        if _has_clients and (fps_counter % 3 == 0):
            # Compact annotation
            cv2.line(frame, (0, int(h*0.6)), (w, int(h*0.6)), (255, 255, 0), 1)
            if lane_found: cv2.circle(frame, (cx_lane, int(h*0.8)), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"FPS:{state['fps']} E:{error:+.2f} S:{steer:+.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            with frame_lock: latest_frame = buf.tobytes()

    cap.release()

# ── Flask ────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>JetRacer</title>
<style>body{background:#0e1117;color:#e8ecf1;font-family:monospace;display:flex;flex-direction:column;align-items:center;}
.grid{display:grid;grid-template-columns:1fr 300px;gap:1rem;max-width:1000px;width:100%;}
.card{background:#161b27;border:1px solid #2a3040;padding:1rem;border-radius:8px;}
img{width:100%;border-radius:4px;}
input[type=range]{width:100%;}
button{width:100%;padding:15px;margin-top:10px;cursor:pointer;border:none;border-radius:4px;font-weight:bold;font-size:1.5rem;}
#go{background:#00d4aa;color:#000;} #stop{background:#ff4d4d;color:#fff;}</style></head>
<body><h1>JetRacer GPU Dash</h1><div class="grid"><div class="card">
<img id="f" src="/v"><div id="stats"></div></div><div class="card">
Speed: <span id="v-speed">0.25</span><input type="range" id="speed" min="0" max="60" value="25" oninput="document.getElementById('v-speed').innerText=(this.value/100).toFixed(2);s('speed',this.value/100)">
<button id="go" onclick="s('enabled',true)">GO</button><button id="stop" onclick="s('enabled',false)">STOP</button>
<hr>H-LO<input type="range" id="h_lo" min="0" max="179" value="20" oninput="s('h_lo',this.value)">
H-HI<input type="range" id="h_hi" min="0" max="179" value="35" oninput="s('h_hi',this.value)"></div></div>
<script>
function s(k,v){fetch('/s',{method:'POST',body:JSON.stringify({[k]:v})});}
async function p(){const r=await fetch('/st');const d=await r.json();document.getElementById('stats').innerText=JSON.stringify(d);setTimeout(p,500);}
p();</script></body></html>"""

@app.route("/")
def index(): return render_template_string(DASHBOARD_HTML)
@app.route("/v")
def v():
    global _has_clients; _has_clients = True
    def g():
        while True:
            with frame_lock: f = latest_frame
            if f: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+f+b"\r\n")
            time.sleep(0.1)
    return Response(g(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/st")
def st():
    with state_lock: return jsonify(state)
@app.route("/s", methods=["POST"])
def set_p():
    with state_lock:
        for k,v in request.get_json(force=True).items():
            if k in state: state[k]=v
    return jsonify({"ok":True})

if __name__ == "__main__":
    car = JetRacer()
    car.arm(3)
    # Important: Set to true once initially so it'll print "Stopped" once and not spam.
    # Control loop will handle toggling appropriately.
    threading.Thread(target=control_loop, args=(car,), daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
