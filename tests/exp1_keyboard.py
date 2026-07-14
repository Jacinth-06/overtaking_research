#!/usr/bin/env python3
"""
exp1_keyboard.py — Keyboard / button control
=============================================
Car moves with commands from the Flask dashboard (Forward, Backward, Left, Right, Stop).
No camera, no LiDAR, no vision pipeline.

Run:   python exp1_keyboard.py
Open:  http://<jetson-ip>:5000
"""

import threading
import time
import serial
import queue
import requests
from flask import Flask, Response, render_template_string, request, jsonify

from jetracer import JetRacer

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "speed": 0.15,
    "enabled": False,
    "command": "stop",        # "forward", "backward", "left", "right", "stop"
    "steer_amount": 0.0,      # -1.0 to 1.0, set by left/right buttons
    # Encoder data
    "enc_speed": 0.0, "enc_dist": 0.0,
    # IMU data
    "imu_ax": 0, "imu_ay": 0, "imu_az": 0,
    "imu_gx": 0, "imu_gy": 0, "imu_gz": 0,
    # Telemetry
    "is_testing": False,
    "test_id": "",
    "reset_encoder_dist": False,
}

state_lock = threading.Lock()

FIREBASE_URL = "https://jetracer-f1b1c-default-rtdb.asia-southeast1.firebasedatabase.app"
telemetry_queue = queue.Queue()

# ── Sensor caches ─────────────────────────────────────────────────────────────
_imu_cache = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}
_imu_cache_lock = threading.Lock()

_encoder_cache = {"speed": 0.0, "distance": 0.0}
_encoder_cache_lock = threading.Lock()


# ── Firebase Telemetry ────────────────────────────────────────────────────────
def firebase_loop():
    print("[firebase] Telemetry background thread started")
    batch = {}
    last_upload_time = time.time()

    while True:
        try:
            test_id, timestamp, data_point = telemetry_queue.get(timeout=0.5)
            if test_id not in batch:
                batch[test_id] = {}
            key = str(int(timestamp * 1000))
            batch[test_id][key] = data_point
        except queue.Empty:
            pass

        now = time.time()
        if now - last_upload_time >= 2.0:
            if batch:
                try:
                    for tid, b_data in batch.items():
                        url = f"{FIREBASE_URL}/Tune Q/{tid}.json"
                        requests.patch(url, json=b_data, timeout=5)
                    batch.clear()
                except Exception as e:
                    print(f"[firebase] upload error: {e}")
            last_upload_time = now


# ── Sensor loop (IMU + encoder) ──────────────────────────────────────────────
def sensor_loop():
    """
    Background thread: parses the Waveshare RP2040 protocol correctly.
    Full packet: AA 55 2D 01 [39 data bytes] [checksum]
    Total = 45 bytes including both headers.
    """
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
            if not b or b[0] != HEAD1:
                continue
            b = ser.read(1)
            if not b or b[0] != HEAD2:
                continue
            b = ser.read(1)
            if not b:
                continue
            frame_size = b[0]
            if frame_size < 5 or frame_size > 50:
                continue
            remaining = frame_size - 3
            rest = ser.read(remaining)
            if len(rest) != remaining:
                continue
            frame = bytes([HEAD1, HEAD2, frame_size]) + rest
            calc_sum = sum(frame[:-1]) & 0xFF
            recv_sum = frame[-1]
            if calc_sum != recv_sum:
                continue

            # Parse IMU
            gx = int.from_bytes(frame[4:6],   'big', signed=True) / 32768 * 2000
            gy = int.from_bytes(frame[6:8],   'big', signed=True) / 32768 * 2000
            gz = int.from_bytes(frame[8:10],  'big', signed=True) / 32768 * 2000
            ax = int.from_bytes(frame[10:12], 'big', signed=True) / 32768 * 2 * 9.8
            ay = int.from_bytes(frame[12:14], 'big', signed=True) / 32768 * 2 * 9.8
            az = int.from_bytes(frame[14:16], 'big', signed=True) / 32768 * 2 * 9.8

            with _imu_cache_lock:
                _imu_cache["ax"] = round(ax, 2)
                _imu_cache["ay"] = round(ay, 2)
                _imu_cache["az"] = round(az, 2)
                _imu_cache["gx"] = round(gx, 1)
                _imu_cache["gy"] = round(gy, 1)
                _imu_cache["gz"] = round(gz, 1)

            # Parse Encoder
            lvel = int.from_bytes(frame[34:36], 'big', signed=True)
            rvel = int.from_bytes(frame[36:38], 'big', signed=True)
            now = time.time()
            dt = now - last_time
            if dt > 0:
                avg_vel = (lvel + rvel) / 2.0
                speed_ms = avg_vel * SPEED_SCALE
                d_dist = speed_ms * dt
                total_distance += d_dist
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

            if cmd == "forward":
                car.steer(steer_amt)
                car.forward(speed)
            elif cmd == "backward":
                car.steer(steer_amt)
                car.reverse(speed)
            elif cmd == "left":
                car.steer(-1.0)
                car.forward(speed)
            elif cmd == "right":
                car.steer(1.0)
                car.forward(speed)
            else:  # "stop"
                car.stop()
        else:
            car.stop()

        # Update shared state
        with state_lock:
            state["enc_speed"] = enc_speed
            state["enc_dist"] = enc_dist
            state["imu_ax"] = imu_data["ax"]
            state["imu_ay"] = imu_data["ay"]
            state["imu_az"] = imu_data["az"]
            state["imu_gx"] = imu_data["gx"]
            state["imu_gy"] = imu_data["gy"]
            state["imu_gz"] = imu_data["gz"]

        # Telemetry
        if s_copy.get("is_testing") and s_copy.get("test_id"):
            current_time = time.time()
            data_point = {
                "timestamp": current_time,
                "command": s_copy["command"],
                "speed": s_copy["speed"],
                "enc_speed": enc_speed,
                "enc_dist": enc_dist,
                "imu_ax": imu_data["ax"], "imu_ay": imu_data["ay"], "imu_az": imu_data["az"],
                "imu_gx": imu_data["gx"], "imu_gy": imu_data["gy"], "imu_gz": imu_data["gz"],
            }
            telemetry_queue.put((s_copy["test_id"], current_time, data_point))

        time.sleep(0.05)  # 20 Hz


# ── Flask / dashboard ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 1 — Keyboard Control</title>
<style>
  :root {
    --bg: #0e1117; --surface: #161b27; --border: #2a3040;
    --accent: #00d4aa; --warn: #ffb020; --danger: #ff4d4d;
    --text: #e8ecf1; --muted: #6b7a99;
    --font: 'JetBrains Mono', 'Fira Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         display: flex; flex-direction: column; align-items: center;
         min-height: 100vh; padding: 1rem; }
  h1 { font-size: 1rem; letter-spacing: .15em; color: var(--accent);
       text-transform: uppercase; margin-bottom: 1rem; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 1.5rem; max-width: 500px; width: 100%; margin-bottom: 1rem; }
  .card h2 { font-size: .7rem; letter-spacing: .12em; color: var(--muted);
             text-transform: uppercase; margin-bottom: .75rem; }
  .status-bar { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: .75rem; }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
  .slider-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .55rem; }
  .slider-row label { font-size: .7rem; color: var(--muted); width: 85px; flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val { font-size: .75rem; width: 45px; text-align: right; color: var(--text); }
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; flex-wrap: wrap; }
  button { padding: .5rem 1.1rem; border: none; border-radius: 6px;
           cursor: pointer; font-family: var(--font); font-size: .8rem;
           font-weight: 600; letter-spacing: .04em; }
  .btn-go   { background: var(--accent); color: #061612; }
  .btn-stop { background: var(--danger); color: #fff; }
  .btn-dir  { background: #2a3a5a; color: var(--text); min-width: 70px; }
  .btn-dir.active { background: var(--accent); color: #061612; }
  .btn-dir:hover  { filter: brightness(1.2); }
  .divider { border: none; border-top: 1px solid var(--border); margin: .75rem 0; }
  .dpad { display: grid; grid-template-columns: 70px 70px 70px; grid-template-rows: 50px 50px 50px;
          gap: 4px; justify-content: center; margin: 1rem 0; }
  .dpad button { width: 100%; height: 100%; }
  .dpad .empty { visibility: hidden; }
</style>
</head>
<body>
<h1>&#9675; Exp 1 &mdash; Keyboard Control</h1>

<div class="card">
  <h2>Drive Controls</h2>
  <div class="slider-row">
    <label>Speed</label>
    <input type="range" id="speed" min="0" max="60" value="15" step="1">
    <span class="val" id="v-speed">0.15</span>
  </div>
  <div class="btn-row">
    <button class="btn-go" onclick="setEnabled(true)">&#9654; GO</button>
    <button class="btn-stop" onclick="setEnabled(false)">&#9632; STOP</button>
  </div>
  <hr class="divider">
  <h2>Direction Pad</h2>
  <div class="dpad">
    <div class="empty"></div>
    <button class="btn-dir" onmousedown="sendCmd('forward')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('forward')" ontouchend="sendCmd('stop')">&#9650; FWD</button>
    <div class="empty"></div>
    <button class="btn-dir" onmousedown="sendCmd('left')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('left')" ontouchend="sendCmd('stop')">&#9664; LEFT</button>
    <button class="btn-dir" onmousedown="sendCmd('stop')" style="background:var(--danger);color:#fff;">&#9632;</button>
    <button class="btn-dir" onmousedown="sendCmd('right')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('right')" ontouchend="sendCmd('stop')">&#9654; RIGHT</button>
    <div class="empty"></div>
    <button class="btn-dir" onmousedown="sendCmd('backward')" onmouseup="sendCmd('stop')" ontouchstart="sendCmd('backward')" ontouchend="sendCmd('stop')">&#9660; REV</button>
    <div class="empty"></div>
  </div>
  <hr class="divider">
  <h2>Steering Fine Control</h2>
  <div class="slider-row">
    <label>Steer</label>
    <input type="range" id="steer_amount" min="-100" max="100" value="0" step="1">
    <span class="val" id="v-steer_amount">0.00</span>
  </div>
</div>

<div class="card">
  <h2>Telemetry</h2>
  <div class="status-bar">
    <div class="stat"><span class="stat-val" id="v-cmd">stop</span>
                      <span class="stat-lbl">command</span></div>
    <div class="stat"><span class="stat-val" id="v-enc-spd">0.00</span>
                      <span class="stat-lbl">enc spd</span></div>
    <div class="stat"><span class="stat-val" id="v-enc-dist">0.00</span>
                      <span class="stat-lbl">enc dist</span></div>
    <div class="stat"><span class="stat-val" id="v-imu-a">0,0,0</span>
                      <span class="stat-lbl">imu a</span></div>
    <div class="stat"><span class="stat-val" id="v-imu-g">0,0,0</span>
                      <span class="stat-lbl">imu g</span></div>
  </div>
  <div class="btn-row">
    <button id="btn-test" onclick="toggleTest()" style="background: #6b7a99; color: #fff;">&#9654; START TEST</button>
    <span class="stat"><span class="stat-val" id="v-test">--</span>
                       <span class="stat-lbl">test id</span></span>
  </div>
</div>

<script>
const speedEl = document.getElementById("speed");
const speedDisp = document.getElementById("v-speed");
speedEl.addEventListener("input", () => {
  const v = parseFloat(speedEl.value);
  speedDisp.textContent = (v/100).toFixed(2);
  sendParam("speed", v/100);
});

const steerEl = document.getElementById("steer_amount");
const steerDisp = document.getElementById("v-steer_amount");
steerEl.addEventListener("input", () => {
  const v = parseFloat(steerEl.value);
  steerDisp.textContent = (v/100).toFixed(2);
  sendParam("steer_amount", v/100);
});

function sendParam(key, value) {
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({[key]: value})});
}
function setEnabled(v) {
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({enabled: v})});
}
function sendCmd(cmd) {
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({command: cmd})});
}

let isTesting = false;
function toggleTest() {
  isTesting = !isTesting;
  fetch("/set", {method:"POST", headers:{"Content-Type":"application/json"},
                 body: JSON.stringify({is_testing: isTesting})});
}

// Keyboard support
document.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  switch(e.key.toLowerCase()) {
    case "w": case "arrowup":    sendCmd("forward"); break;
    case "s": case "arrowdown":  sendCmd("backward"); break;
    case "a": case "arrowleft":  sendCmd("left"); break;
    case "d": case "arrowright": sendCmd("right"); break;
    case " ": sendCmd("stop"); break;
  }
});
document.addEventListener("keyup", (e) => {
  switch(e.key.toLowerCase()) {
    case "w": case "arrowup":
    case "s": case "arrowdown":
    case "a": case "arrowleft":
    case "d": case "arrowright":
      sendCmd("stop"); break;
  }
});

async function poll() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    document.getElementById("v-cmd").textContent = d.command;
    document.getElementById("v-enc-spd").textContent = d.enc_speed.toFixed(3) + " m/s";
    document.getElementById("v-enc-dist").textContent = d.enc_dist.toFixed(3) + " m";
    const imuAEl = document.getElementById("v-imu-a");
    if(imuAEl) imuAEl.textContent = d.imu_ax + "," + d.imu_ay + "," + d.imu_az;
    const imuGEl = document.getElementById("v-imu-g");
    if(imuGEl) imuGEl.textContent = d.imu_gx + "," + d.imu_gy + "," + d.imu_gz;
    const testEl = document.getElementById("v-test");
    if(testEl) testEl.textContent = d.is_testing ? d.test_id : "--";
    isTesting = d.is_testing;
    const btn = document.getElementById("btn-test");
    if(btn) {
      if(isTesting) { btn.innerHTML = "&#9632; END TEST"; btn.style.background = "#ffb020"; }
      else { btn.innerHTML = "&#9654; START TEST"; btn.style.background = "#6b7a99"; }
    }
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
                elif not v and state.get("is_testing", False):
                    state["is_testing"] = False
            elif k in state:
                if k == "enabled" and state["enabled"] and not v:
                    state["reset_encoder_dist"] = True
                state[k] = v
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = JetRacer(init_lidar=False)
    car.arm(delay=3)

    ft = threading.Thread(target=firebase_loop, daemon=True)
    ft.start()

    st = threading.Thread(target=sensor_loop, daemon=True)
    st.start()

    t = threading.Thread(target=control_loop, args=(car,), daemon=True)
    t.start()

    print("[flask] Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
