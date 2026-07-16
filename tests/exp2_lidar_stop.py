#!/usr/bin/env python3
"""
exp2_lidar_stop.py — Straight drive with LiDAR safety stop
===========================================================
Car drives straight at a set speed. If the front LiDAR cone detects an obstacle
closer than stop_distance, the car stops. It resumes when the obstacle clears.

No camera, no vision pipeline. Includes real-time graphing and CSV download.

Run:   python exp2_lidar_stop.py
Open:  http://<jetson-ip>:5000
"""

import threading
import time
import serial
import csv
import io
from flask import Flask, Response, render_template_string, request, jsonify

from jetracer import JetRacer

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "speed": 0.15,
    "enabled": False,
    "stop_distance": 400.0,   # mm
    "lidar_closest": 0.0,
    "lidar_blocked": False,
    "enc_speed": 0.0, "enc_dist": 0.0,
    "imu_ax": 0, "imu_ay": 0, "imu_az": 0,
    "imu_gx": 0, "imu_gy": 0, "imu_gz": 0,
    "is_testing": False,
    "test_id": "",
    "reset_encoder_dist": False,
}

state_lock = threading.Lock()

# ── Sensor caches ─────────────────────────────────────────────────────────────
_imu_cache = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}
_imu_cache_lock = threading.Lock()

_encoder_cache = {"speed": 0.0, "distance": 0.0}
_encoder_cache_lock = threading.Lock()

_lidar_cache = {"closest": 0.0, "blocked": False}
_lidar_cache_lock = threading.Lock()

# ── Data Logging ──────────────────────────────────────────────────────────────
data_log = []
data_lock = threading.Lock()

def telemetry_loop():
    print("[telemetry] Local data logging loop started (20Hz)")
    while True:
        with state_lock:
            s_copy = dict(state)
        with _encoder_cache_lock:
            enc_copy = dict(_encoder_cache)

        if s_copy.get("is_testing"):
            dp = {
                "timestamp": round(time.time(), 3),
                "speed_cmd": s_copy["speed"],
                "lidar_closest": s_copy["lidar_closest"],
                "lidar_blocked": 1 if s_copy["lidar_blocked"] else 0,
                "enc_speed": enc_copy["speed"],
                "enc_dist": enc_copy["distance"],
            }
            with data_lock:
                data_log.append(dp)
        
        time.sleep(0.05)


# ── Lidar background thread ──────────────────────────────────────────────────
def lidar_loop(car: JetRacer):
    print("[lidar] Background safety thread started")
    while True:
        try:
            with state_lock:
                STOP_DISTANCE = state["stop_distance"]

            scan = car.lidar_scan(samples=150)
            front_distances = [dist for ang, dist in scan.items()
                               if (ang >= 320 or ang <= 40) and dist > 10]

            closest_front = min(front_distances) if front_distances else 0.0
            is_blocked = closest_front > 0 and closest_front < STOP_DISTANCE

            with _lidar_cache_lock:
                _lidar_cache["closest"] = round(closest_front, 1)
                _lidar_cache["blocked"] = is_blocked

        except Exception as e:
            print(f"[lidar] scan error: {e}")
            with _lidar_cache_lock:
                _lidar_cache["closest"] = 0.0
                _lidar_cache["blocked"] = True

        time.sleep(0.05)


# ── Sensor loop (IMU + encoder) ──────────────────────────────────────────────
def sensor_loop():
    print("[sensors] Background thread started")
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    except Exception as e:
        print(f"[sensors] Failed to open serial: {e}")
        return

    HEAD1 = 0xAA
    HEAD2 = 0x55
    total_distance = 0.0
    last_time = time.time()
    SPEED_SCALE = 0.00748

    while True:
        with state_lock:
            if state.get("reset_encoder_dist"):
                total_distance = 0.0
                state["reset_encoder_dist"] = False

        try:
            b = ser.read(1)
            if not b or b[0] != HEAD1: continue
            b = ser.read(1)
            if not b or b[0] != HEAD2: continue
            b = ser.read(1)
            if not b: continue
            frame_size = b[0]
            if frame_size < 5 or frame_size > 50: continue
            remaining = frame_size - 3
            rest = ser.read(remaining)
            if len(rest) != remaining: continue
            frame = bytes([HEAD1, HEAD2, frame_size]) + rest
            if (sum(frame[:-1]) & 0xFF) != frame[-1]: continue

            # Parse Encoder
            lvel = int.from_bytes(frame[34:36], 'big', signed=True)
            rvel = int.from_bytes(frame[36:38], 'big', signed=True)
            now = time.time()
            dt = now - last_time
            if dt > 0:
                speed_ms = ((lvel + rvel) / 2.0) * SPEED_SCALE
                total_distance += speed_ms * dt
                with _encoder_cache_lock:
                    _encoder_cache["speed"] = round(speed_ms, 3)
                    _encoder_cache["distance"] = round(total_distance, 3)
            last_time = now

        except Exception as e:
            print(f"[sensors] error: {e}")
            time.sleep(0.1)


# ── Control loop ──────────────────────────────────────────────────────────────
def control_loop(car: JetRacer):
    print("[loop] LiDAR stop control loop started")
    while True:
        with state_lock:
            s_copy = dict(state)
        with _lidar_cache_lock:
            lidar_closest = _lidar_cache["closest"]
            lidar_blocked = _lidar_cache["blocked"]
        with _encoder_cache_lock:
            enc_speed = _encoder_cache["speed"]
            enc_dist = _encoder_cache["distance"]

        if s_copy["enabled"]:
            if lidar_blocked:
                car.stop()
            else:
                car.steer(0.0)
                car.forward(s_copy["speed"])
        else:
            car.stop()

        with state_lock:
            state["lidar_closest"] = lidar_closest
            state["lidar_blocked"] = lidar_blocked
            state["enc_speed"] = enc_speed
            state["enc_dist"] = enc_dist

        time.sleep(0.05)


# ── Flask / dashboard ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 2 — LiDAR Stop</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg: #0b0f19; --surface: #111827; --surface-border: #1f2937;
    --accent: #3b82f6; --accent-hover: #2563eb; 
    --success: #10b981; --danger: #ef4444; --warn: #f59e0b;
    --text-main: #f3f4f6; --text-muted: #9ca3af;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text-main); font-family: var(--font); padding: 2rem; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; border-bottom: 1px solid var(--surface-border); padding-bottom: 1rem; }
  .header h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: 0.05em; }
  .badge { background: var(--surface-border); padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-family: monospace; color: var(--text-muted); }
  
  .dashboard { display: grid; grid-template-columns: 350px 1fr; gap: 2rem; }
  .panel { background: var(--surface); border: 1px solid var(--surface-border); border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
  .panel-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; font-weight: 600; }
  
  .slider-group { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1.2rem; }
  .slider-group label { font-size: 0.85rem; color: var(--text-muted); display: flex; justify-content: space-between; }
  .slider-group input[type=range] { width: 100%; accent-color: var(--accent); }
  
  .btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1.5rem; }
  button { padding: 0.75rem 1rem; border: none; border-radius: 8px; font-family: var(--font); font-weight: 600; font-size: 0.9rem; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { background: var(--accent-hover); }
  .btn-danger { background: var(--surface-border); color: var(--danger); }
  .btn-danger:hover { background: var(--danger); color: white; }
  
  .test-controls { margin-top: 2rem; border-top: 1px solid var(--surface-border); padding-top: 1.5rem; }
  .btn-test { background: var(--success); color: white; width: 100%; margin-bottom: 0.5rem; }
  .btn-test.active { background: var(--warn); color: #000; }
  .btn-download { background: var(--surface-border); color: var(--text-main); width: 100%; text-decoration: none; display: inline-block; text-align: center; font-weight: 600; font-size: 0.9rem; padding: 0.75rem 1rem; border-radius: 8px; }
  .btn-download:hover { background: #374151; }
  
  .charts-grid { display: grid; grid-template-columns: 1fr; gap: 1.5rem; }
  .chart-container { position: relative; height: 300px; width: 100%; background: #182235; border-radius: 8px; padding: 1rem; border: 1px solid var(--surface-border); }

  .status-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
  .stat-box { background: #182235; padding: 1rem; border-radius: 8px; border: 1px solid var(--surface-border); }
  .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem; }
  .stat-value { font-size: 1.25rem; font-weight: 600; font-family: monospace; color: var(--accent); }
  .stat-value.blocked { color: var(--danger); }

  @media (max-width: 900px) { .dashboard { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>Exp 2: Straight Drive + LiDAR Stop</h1>
  <div class="badge">Telemetry Mode: <span id="mode-badge">IDLE</span></div>
</div>

<div class="dashboard">
  <!-- Control Panel -->
  <div class="panel">
    <div class="panel-title">Vehicle Control</div>
    <div class="btn-group">
      <button class="btn-primary" onclick="setEnabled(true)">Enable Drive</button>
      <button class="btn-danger" onclick="setEnabled(false)">Stop / Disable</button>
    </div>

    <div class="slider-group">
      <label>Target Speed <span id="v-speed">0.15</span></label>
      <input type="range" id="speed" min="0" max="60" value="15" step="1">
    </div>

    <div class="slider-group">
      <label>Stop Distance (mm) <span id="v-stop">400</span></label>
      <input type="range" id="stop_distance" min="100" max="2000" value="400" step="10">
    </div>

    <div class="test-controls">
      <div class="panel-title">Data Collection</div>
      <button id="btn-test" class="btn-test" onclick="toggleTest()">START RECORDING</button>
      <a href="/download_csv" class="btn-download button" target="_blank" style="box-sizing: border-box;">Download CSV</a>
    </div>
  </div>

  <!-- Telemetry Panel -->
  <div>
    <div class="status-grid">
      <div class="stat-box">
        <div class="stat-label">Obstacle Status</div>
        <div class="stat-value" id="disp-blocked">CLEAR</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">LiDAR Front</div>
        <div class="stat-value" id="disp-lidar">0 mm</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Encoder Speed</div>
        <div class="stat-value" id="disp-spd">0.00 m/s</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-container"><canvas id="chartSpeed"></canvas></div>
      <div class="chart-container"><canvas id="chartLidar"></canvas></div>
    </div>
  </div>
</div>

<script>
// UI Controls
const speedEl = document.getElementById("speed");
const speedDisp = document.getElementById("v-speed");
speedEl.addEventListener("input", () => {
  const v = (parseFloat(speedEl.value)/100).toFixed(2);
  speedDisp.textContent = v; sendParam("speed", parseFloat(v));
});

const stopEl = document.getElementById("stop_distance");
const stopDisp = document.getElementById("v-stop");
stopEl.addEventListener("input", () => {
  const v = parseFloat(stopEl.value);
  stopDisp.textContent = v; sendParam("stop_distance", v);
});

function sendParam(key, value) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({[key]: value})}); }
function setEnabled(v) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled: v})}); }

// Chart.js Setup
Chart.defaults.color = '#9ca3af';
Chart.defaults.font.family = "'Inter', sans-serif";

const commonOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
  scales: { x: { display: false }, y: { grid: { color: '#1f2937' } } },
  plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
  elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.1 } }
};

const chartSpeed = new Chart(document.getElementById('chartSpeed'), {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Encoder Speed (m/s)', borderColor: '#10b981', data: [] }] },
  options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: -0.5, max: 1.5 } } }
});

const chartLidar = new Chart(document.getElementById('chartLidar'), {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'LiDAR Closest Distance (mm)', borderColor: '#3b82f6', data: [] }] },
  options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 2000 } } }
});

// Telemetry Polling
let isTesting = false;
let dataPoints = 0;

function toggleTest() {
  isTesting = !isTesting;
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({is_testing: isTesting})});
  if (isTesting) {
    chartSpeed.data.labels = []; chartSpeed.data.datasets[0].data = [];
    chartLidar.data.labels = []; chartLidar.data.datasets[0].data = [];
    dataPoints = 0;
  }
}

async function poll() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    
    document.getElementById("disp-spd").textContent = d.enc_speed.toFixed(3) + " m/s";
    document.getElementById("disp-lidar").textContent = d.lidar_closest.toFixed(0) + " mm";
    
    const blockedEl = document.getElementById("disp-blocked");
    if(d.lidar_blocked) {
      blockedEl.textContent = "BLOCKED"; blockedEl.classList.add("blocked");
    } else {
      blockedEl.textContent = "CLEAR"; blockedEl.classList.remove("blocked");
    }
    
    isTesting = d.is_testing;
    const btn = document.getElementById("btn-test");
    const badge = document.getElementById("mode-badge");
    if(isTesting) { 
      btn.innerHTML = "STOP RECORDING"; btn.classList.add("active"); 
      badge.textContent = "RECORDING"; badge.style.color = "#f59e0b";
      
      dataPoints++;
      chartSpeed.data.labels.push(dataPoints);
      chartSpeed.data.datasets[0].data.push(d.enc_speed);
      
      chartLidar.data.labels.push(dataPoints);
      chartLidar.data.datasets[0].data.push(d.lidar_closest);
      
      chartSpeed.update(); chartLidar.update();
    } else { 
      btn.innerHTML = "START RECORDING"; btn.classList.remove("active"); 
      badge.textContent = "IDLE"; badge.style.color = "#9ca3af";
    }
  } catch(e) {}
  setTimeout(poll, 100);
}
poll();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/status")
def status():
    with state_lock:
        return jsonify({k: state[k] for k in
                        ("speed", "enabled", "stop_distance", "lidar_closest", "lidar_blocked",
                         "enc_speed", "enc_dist", "is_testing", "test_id")})


@app.route("/set", methods=["POST"])
def set_param():
    data = request.get_json(force=True)
    with state_lock:
        for k, v in data.items():
            if k == "is_testing":
                if v and not state.get("is_testing", False):
                    state["test_id"] = f"test_{int(time.time())}"
                    state["is_testing"] = True
                    with data_lock:
                        data_log.clear()
                elif not v and state.get("is_testing", False):
                    state["is_testing"] = False
            elif k in state:
                if k == "enabled" and state["enabled"] and not v:
                    state["reset_encoder_dist"] = True
                state[k] = v
    return jsonify({"ok": True})


@app.route("/download_csv")
def download_csv():
    with data_lock:
        if not data_log:
            return "No data recorded yet. Click 'START RECORDING' first.", 400
        keys = data_log[0].keys()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_log)
    return Response(output.getvalue(), mimetype="text/csv", 
                    headers={"Content-Disposition": "attachment;filename=exp2_telemetry.csv"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = JetRacer(init_lidar=True)
    car.arm(delay=3)

    tt = threading.Thread(target=telemetry_loop, daemon=True)
    tt.start()

    lt = threading.Thread(target=lidar_loop, args=(car,), daemon=True)
    lt.start()

    st = threading.Thread(target=sensor_loop, daemon=True)
    st.start()

    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
