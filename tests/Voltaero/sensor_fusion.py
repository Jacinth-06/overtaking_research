#!/usr/bin/env python3
"""
exp1_keyboard.py — Keyboard / button control
=============================================
Car moves with commands from the Flask dashboard.
No camera, no LiDAR. Includes real-time graphing and CSV download.

Run:   python exp1_keyboard.py
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
    "command": "stop",
    "steer_amount": 0.0,
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

# ── Data Logging ──────────────────────────────────────────────────────────────
data_log = []
data_lock = threading.Lock()

def telemetry_loop():
    print("[telemetry] Local data logging loop started (20Hz)")
    while True:
        with state_lock:
            s_copy = dict(state)
        with _imu_cache_lock:
            imu_copy = dict(_imu_cache)
        with _encoder_cache_lock:
            enc_copy = dict(_encoder_cache)

        if s_copy.get("is_testing"):
            dp = {
                "timestamp": round(time.time(), 3),
                "command": s_copy["command"],
                "speed": s_copy["speed"],
                "enc_speed": enc_copy["speed"],
                "enc_dist": enc_copy["distance"],
                "imu_ax": imu_copy["ax"],
                "imu_ay": imu_copy["ay"],
                "imu_az": imu_copy["az"],
                "imu_gx": imu_copy["gx"],
                "imu_gy": imu_copy["gy"],
                "imu_gz": imu_copy["gz"],
            }
            with data_lock:
                data_log.append(dp)
        
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

            # Parse IMU
            gx = int.from_bytes(frame[4:6],   'big', signed=True) / 32768 * 2000
            gy = int.from_bytes(frame[6:8],   'big', signed=True) / 32768 * 2000
            gz = int.from_bytes(frame[8:10],  'big', signed=True) / 32768 * 2000
            ax = int.from_bytes(frame[10:12], 'big', signed=True) / 32768 * 2 * 9.8
            ay = int.from_bytes(frame[12:14], 'big', signed=True) / 32768 * 2 * 9.8
            az = int.from_bytes(frame[14:16], 'big', signed=True) / 32768 * 2 * 9.8

            with _imu_cache_lock:
                _imu_cache["ax"] = round(ax, 2); _imu_cache["ay"] = round(ay, 2); _imu_cache["az"] = round(az, 2)
                _imu_cache["gx"] = round(gx, 1); _imu_cache["gy"] = round(gy, 1); _imu_cache["gz"] = round(gz, 1)

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
    print("[loop] Keyboard control loop started")
    while True:
        with state_lock:
            s_copy = dict(state)
        with _imu_cache_lock:
            imu_data = dict(_imu_cache)
        with _encoder_cache_lock:
            enc_speed = _encoder_cache["speed"]
            enc_dist = _encoder_cache["distance"]

        if s_copy["enabled"]:
            cmd = s_copy["command"]
            speed = s_copy["speed"]
            steer_amt = s_copy.get("steer_amount", 0.0)

            if cmd == "forward": car.steer(steer_amt); car.forward(speed)
            elif cmd == "backward": car.steer(steer_amt); car.reverse(speed)
            elif cmd == "left": car.steer(-1.0); car.forward(speed)
            elif cmd == "right": car.steer(1.0); car.forward(speed)
            else: car.stop()
        else:
            car.stop()

        with state_lock:
            state["enc_speed"] = enc_speed
            state["enc_dist"] = enc_dist
            state["imu_ax"] = imu_data["ax"]; state["imu_ay"] = imu_data["ay"]; state["imu_az"] = imu_data["az"]
            state["imu_gx"] = imu_data["gx"]; state["imu_gy"] = imu_data["gy"]; state["imu_gz"] = imu_data["gz"]

        time.sleep(0.05)


# ── Flask / dashboard ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 1 — Keyboard Control & Telemetry</title>
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
  
  .btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
  button { padding: 0.75rem 1rem; border: none; border-radius: 8px; font-family: var(--font); font-weight: 600; font-size: 0.9rem; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { background: var(--accent-hover); }
  .btn-danger { background: var(--surface-border); color: var(--danger); }
  .btn-danger:hover { background: var(--danger); color: white; }
  
  .dpad { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.25rem; margin: 1.5rem 0; }
  .dpad button { background: var(--surface-border); color: var(--text-muted); padding: 1rem 0; font-size: 1.2rem; }
  .dpad button:hover { background: var(--accent); color: white; }
  .dpad button:active { transform: scale(0.95); }
  .dpad .center { background: var(--danger); color: white; }
  .dpad .empty { visibility: hidden; }

  .test-controls { margin-top: 2rem; border-top: 1px solid var(--surface-border); padding-top: 1.5rem; }
  .btn-test { background: var(--success); color: white; width: 100%; margin-bottom: 0.5rem; }
  .btn-test.active { background: var(--warn); color: #000; }
  .btn-download { background: var(--surface-border); color: var(--text-main); width: 100%; text-decoration: none; display: inline-block; text-align: center; font-weight: 600; font-size: 0.9rem; padding: 0.75rem 1rem; border-radius: 8px; }
  .btn-download:hover { background: #374151; }
  
  .charts-grid { display: grid; grid-template-columns: 1fr; gap: 1.5rem; }
  .chart-container { position: relative; height: 260px; width: 100%; background: #182235; border-radius: 8px; padding: 1rem; border: 1px solid var(--surface-border); }

  .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
  .stat-box { background: #182235; padding: 1rem; border-radius: 8px; border: 1px solid var(--surface-border); }
  .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem; }
  .stat-value { font-size: 1.25rem; font-weight: 600; font-family: monospace; color: var(--accent); }

  @media (max-width: 900px) { .dashboard { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>Exp 1: Manual Keyboard Control</h1>
  <div class="badge">Telemetry Mode: <span id="mode-badge">IDLE</span></div>
</div>

<div class="dashboard">
  <!-- Control Panel -->
  <div class="panel">
    <div class="panel-title">Vehicle Control</div>
    <div class="btn-group" style="margin-bottom: 1.5rem;">
      <button class="btn-primary" onclick="setEnabled(true)">Enable Drive</button>
      <button class="btn-danger" onclick="setEnabled(false)">Stop / Disable</button>
    </div>

    <div class="slider-group">
      <label>Speed <span id="v-speed">0.15</span></label>
      <input type="range" id="speed" min="0" max="60" value="15" step="1">
    </div>

    <div class="slider-group">
      <label>Fine Steer Adjust <span id="v-steer">0.00</span></label>
      <input type="range" id="steer_amount" min="-100" max="100" value="0" step="1">
    </div>

    <div class="dpad">
      <div class="empty"></div>
      <button onmousedown="sendCmd('forward')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('forward')" ontouchend="sendCmd('stop')">&#9650;</button>
      <div class="empty"></div>
      <button onmousedown="sendCmd('left')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('left')" ontouchend="sendCmd('stop')">&#9664;</button>
      <button class="center" onmousedown="sendCmd('stop')">&#9632;</button>
      <button onmousedown="sendCmd('right')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('right')" ontouchend="sendCmd('stop')">&#9654;</button>
      <div class="empty"></div>
      <button onmousedown="sendCmd('backward')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('backward')" ontouchend="sendCmd('stop')">&#9660;</button>
      <div class="empty"></div>
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
        <div class="stat-label">Encoder Speed</div>
        <div class="stat-value" id="disp-spd">0.00 m/s</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Encoder Distance</div>
        <div class="stat-value" id="disp-dist">0.00 m</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-container"><canvas id="chartSpeed"></canvas></div>
      <div class="chart-container"><canvas id="chartAccel"></canvas></div>
      <div class="chart-container"><canvas id="chartYaw"></canvas></div>
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

const steerEl = document.getElementById("steer_amount");
const steerDisp = document.getElementById("v-steer");
steerEl.addEventListener("input", () => {
  const v = (parseFloat(steerEl.value)/100).toFixed(2);
  steerDisp.textContent = v; sendParam("steer_amount", parseFloat(v));
});

function sendParam(key, value) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({[key]: value})}); }
function setEnabled(v) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({enabled: v})}); }
function sendCmd(cmd) { fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({command: cmd})}); }

// Keyboard Support
document.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  switch(e.key.toLowerCase()) {
    case "w": case "arrowup": sendCmd("forward"); break;
    case "s": case "arrowdown": sendCmd("backward"); break;
    case "a": case "arrowleft": sendCmd("left"); break;
    case "d": case "arrowright": sendCmd("right"); break;
    case " ": sendCmd("stop"); break;
  }
});
document.addEventListener("keyup", (e) => {
  if(["w","a","s","d","arrowup","arrowdown","arrowleft","arrowright"].includes(e.key.toLowerCase())) sendCmd("stop");
});

// Chart.js Setup
Chart.defaults.color = '#9ca3af';
Chart.defaults.font.family = "'Inter', sans-serif";

const commonOptions = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 0 },
  scales: {
    x: { display: false }, 
    y: { grid: { color: '#1f2937' } }
  },
  plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
  elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.1 } }
};

const chartSpeed = new Chart(document.getElementById('chartSpeed'), {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Encoder Speed (m/s)', borderColor: '#10b981', data: [] }] },
  options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: -1, max: 1 } } }
});

const chartAccel = new Chart(document.getElementById('chartAccel'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Accel X', borderColor: '#3b82f6', data: [] },
    { label: 'Accel Y', borderColor: '#ef4444', data: [] }
  ]},
  options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: -10, max: 10 } } }
});

const chartYaw = new Chart(document.getElementById('chartYaw'), {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Yaw Rate Z (deg/s)', borderColor: '#f59e0b', data: [] }] },
  options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: -200, max: 200 } } }
});

// Telemetry Polling
let isTesting = false;
let dataPoints = 0;

function toggleTest() {
  isTesting = !isTesting;
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({is_testing: isTesting})});
  if (isTesting) {
    // Clear graphs
    chartSpeed.data.labels = []; chartSpeed.data.datasets[0].data = [];
    chartAccel.data.labels = []; chartAccel.data.datasets[0].data = []; chartAccel.data.datasets[1].data = [];
    chartYaw.data.labels = []; chartYaw.data.datasets[0].data = [];
    dataPoints = 0;
  }
}

async function poll() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    
    document.getElementById("disp-spd").textContent = d.enc_speed.toFixed(3) + " m/s";
    document.getElementById("disp-dist").textContent = d.enc_dist.toFixed(3) + " m";
    
    isTesting = d.is_testing;
    const btn = document.getElementById("btn-test");
    const badge = document.getElementById("mode-badge");
    if(isTesting) { 
      btn.innerHTML = "STOP RECORDING"; btn.classList.add("active"); 
      badge.textContent = "RECORDING"; badge.style.color = "#f59e0b";
      
      dataPoints++;
      chartSpeed.data.labels.push(dataPoints);
      chartSpeed.data.datasets[0].data.push(d.enc_speed);
      
      chartAccel.data.labels.push(dataPoints);
      chartAccel.data.datasets[0].data.push(d.imu_ax);
      chartAccel.data.datasets[1].data.push(d.imu_ay);
      
      chartYaw.data.labels.push(dataPoints);
      chartYaw.data.datasets[0].data.push(d.imu_gz);
      
      chartSpeed.update(); chartAccel.update(); chartYaw.update();
    } else { 
      btn.innerHTML = "START RECORDING"; btn.classList.remove("active"); 
      badge.textContent = "IDLE"; badge.style.color = "#9ca3af";
    }
  } catch(e) {}
  setTimeout(poll, 100); // 10Hz UI refresh
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
                        ("command", "speed", "enabled", "enc_speed", "enc_dist",
                         "imu_ax", "imu_ay", "imu_az", "imu_gx", "imu_gy", "imu_gz",
                         "is_testing", "test_id")})


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
                    headers={"Content-Disposition": "attachment;filename=exp1_telemetry.csv"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = JetRacer(init_lidar=False)
    car.arm(delay=3)

    tt = threading.Thread(target=telemetry_loop, daemon=True)
    tt.start()

    st = threading.Thread(target=sensor_loop, daemon=True)
    st.start()

    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
