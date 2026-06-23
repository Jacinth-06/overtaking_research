#!/usr/bin/env python3
"""
sensor_test.py — IMU + Encoder diagnostic tool
================================================
Run on Jetson directly. No camera, no car, no Firebase.

Usage:
    python3 sensor_test.py           # runs all tests interactively
    python3 sensor_test.py --imu     # IMU only
    python3 sensor_test.py --enc     # Encoder only

Tests:
    1. IMU static bias  (car still, measure gz drift)
    2. IMU yaw drift    (car still, integrate yaw for 10s)
    3. Encoder speed    (drive manually, measure speed)
    4. Encoder distance (drive 1m, check reported distance)
"""

import serial
import time
import math
import argparse
import sys
import threading

# ── Config — match your version5.py exactly ──────────────────────────────────
SERIAL_PORT  = '/dev/ttyACM0'
BAUD_RATE    = 115200
SPEED_SCALE  = 0.00748   # your current calibration constant

HEAD1 = 0xAA
HEAD2 = 0x55

# ── Shared parsed data ────────────────────────────────────────────────────────
_latest = {
    "gz": 0.0, "gx": 0.0, "gy": 0.0,
    "ax": 0.0, "ay": 0.0, "az": 0.0,
    "lvel": 0, "rvel": 0,
    "speed_ms": 0.0,
    "checksum_fails": 0,
    "packets_ok": 0,
}
_lock = threading.Lock()
_stop_flag = threading.Event()


# ── Serial reader thread ──────────────────────────────────────────────────────
def reader_thread():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[serial] Opened {SERIAL_PORT} at {BAUD_RATE} baud\n")
    except Exception as e:
        print(f"[serial] FAILED to open {SERIAL_PORT}: {e}")
        print("  Check: ls /dev/ttyACM*   or try /dev/ttyUSB0")
        _stop_flag.set()
        return

    while not _stop_flag.is_set():
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
                with _lock:
                    _latest["checksum_fails"] += 1
                continue

            # Parse gyro
            gx = int.from_bytes(frame[4:6],   'big', signed=True) / 32768 * 2000
            gy = int.from_bytes(frame[6:8],   'big', signed=True) / 32768 * 2000
            gz = int.from_bytes(frame[8:10],  'big', signed=True) / 32768 * 2000
            # Parse accel
            ax = int.from_bytes(frame[10:12], 'big', signed=True) / 32768 * 2 * 9.8
            ay = int.from_bytes(frame[12:14], 'big', signed=True) / 32768 * 2 * 9.8
            az = int.from_bytes(frame[14:16], 'big', signed=True) / 32768 * 2 * 9.8
            # Parse encoder
            lvel = int.from_bytes(frame[34:36], 'big', signed=True)
            rvel = int.from_bytes(frame[36:38], 'big', signed=True)
            avg_vel  = (lvel + rvel) / 2.0
            speed_ms = avg_vel * SPEED_SCALE

            with _lock:
                _latest["gz"] = gz
                _latest["gx"] = gx
                _latest["gy"] = gy
                _latest["ax"] = ax
                _latest["ay"] = ay
                _latest["az"] = az
                _latest["lvel"] = lvel
                _latest["rvel"] = rvel
                _latest["speed_ms"] = speed_ms
                _latest["packets_ok"] += 1

        except Exception as e:
            print(f"[serial] parse error: {e}")
            time.sleep(0.05)

    ser.close()


def get():
    with _lock:
        return dict(_latest)


def wait_for_first_packet(timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if get()["packets_ok"] > 0:
            return True
        time.sleep(0.1)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — IMU Static Bias
# ═══════════════════════════════════════════════════════════════════════════════
def test_imu_bias(duration=5.0, n_samples=200):
    print("=" * 60)
    print("TEST 1 — IMU STATIC BIAS")
    print("=" * 60)
    print(f"Keep the car COMPLETELY STILL for {duration}s")
    print("Press Enter when ready...")
    input()

    gz_samples = []
    gx_samples = []
    gy_samples = []
    az_samples = []

    t0 = time.time()
    print(f"Collecting for {duration}s...", end="", flush=True)
    while time.time() - t0 < duration:
        d = get()
        gz_samples.append(d["gz"])
        gx_samples.append(d["gx"])
        gy_samples.append(d["gy"])
        az_samples.append(d["az"])
        time.sleep(duration / n_samples)

    print(" done.\n")

    gz_bias = sum(gz_samples) / len(gz_samples)
    gx_bias = sum(gx_samples) / len(gx_samples)
    gy_bias = sum(gy_samples) / len(gy_samples)
    gz_std  = (sum((x - gz_bias)**2 for x in gz_samples) / len(gz_samples)) ** 0.5
    az_mean = sum(az_samples) / len(az_samples)

    print(f"  gz bias  = {gz_bias:+.3f} deg/s   (want: ~0.0)")
    print(f"  gz noise = {gz_std:.3f} deg/s  (1-sigma)")
    print(f"  gx bias  = {gx_bias:+.3f} deg/s")
    print(f"  gy bias  = {gy_bias:+.3f} deg/s")
    print(f"  az mean  = {az_mean:.2f} m/s²    (want: ~9.8 if flat)")

    # Yaw error estimate over maneuver duration (~8s at 0.08 m/s over 0.7m)
    maneuver_time = 0.70 / 0.08
    yaw_error_deg = gz_bias * maneuver_time
    print(f"\n  At {maneuver_time:.1f}s maneuver, gz bias alone → {yaw_error_deg:+.1f}° yaw error")

    if abs(gz_bias) < 0.5:
        print("  ✓ gz bias is LOW — IMU is usable as-is")
    elif abs(gz_bias) < 2.0:
        print("  ⚠ gz bias is MODERATE — subtract bias in code:")
        print(f"    gz_corrected = gz - ({gz_bias:.3f})")
    else:
        print("  ✗ gz bias is HIGH — must subtract bias, or pos_y will be totally wrong")
        print(f"    gz_corrected = gz - ({gz_bias:.3f})")

    return gz_bias


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — IMU Yaw Drift (still car, integrate yaw for 10s)
# ═══════════════════════════════════════════════════════════════════════════════
def test_imu_drift(gz_bias=0.0, duration=10.0):
    print("\n" + "=" * 60)
    print("TEST 2 — IMU YAW DRIFT (integrated over time)")
    print("=" * 60)
    print(f"Keep the car STILL for {duration}s. Integrating yaw...")
    print("Press Enter when ready...")
    input()

    yaw = 0.0
    pos_y = 0.0
    # Simulate at enc_speed=0.08 m/s (your typical maneuver speed)
    SIMULATED_SPEED = 0.08

    last_t = time.time()
    t0 = last_t

    print(f"\n  {'time':>5}  {'gz(raw)':>9}  {'gz(corrected)':>14}  {'yaw°':>7}  {'fake pos_y':>10}")
    print("  " + "-" * 55)

    while time.time() - t0 < duration:
        time.sleep(0.05)
        now = time.time()
        dt = now - last_t
        last_t = now

        d = get()
        gz_raw = d["gz"]
        gz_corr = gz_raw - gz_bias

        yaw += math.radians(gz_corr) * dt
        vy   = SIMULATED_SPEED * math.sin(yaw)
        pos_y += vy * dt

        elapsed = now - t0
        if int(elapsed * 10) % 10 == 0:   # print every ~1s
            print(f"  {elapsed:5.1f}  {gz_raw:+9.3f}  {gz_corr:+14.3f}  "
                  f"{math.degrees(yaw):+7.2f}  {pos_y:+10.4f} m")

    print(f"\n  Final yaw  = {math.degrees(yaw):+.2f}°  (want: ~0.0°, car didn't move)")
    print(f"  Final pos_y = {pos_y:+.4f} m (fake lateral drift from yaw error)")

    if abs(math.degrees(yaw)) < 2.0:
        print("  ✓ Yaw drift is LOW — IMU integration is reliable")
    elif abs(math.degrees(yaw)) < 8.0:
        print("  ⚠ Yaw drift is MODERATE — bias subtraction will help")
    else:
        print("  ✗ Yaw drift is HIGH — IMU alone cannot track pos_y reliably")
        print("    Consider: fusing vision lane error into pos_y correction")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Encoder Live Speed Monitor
# ═══════════════════════════════════════════════════════════════════════════════
def test_encoder_speed(duration=15.0):
    print("\n" + "=" * 60)
    print("TEST 3 — ENCODER LIVE SPEED")
    print("=" * 60)
    print("Drive the car at a steady throttle while this runs.")
    print("Compare enc_speed to your known throttle setting.")
    print(f"Running for {duration}s. Press Enter to start...")
    input()

    t0 = time.time()
    samples = []

    print(f"\n  {'time':>5}  {'lvel':>6}  {'rvel':>6}  {'speed m/s':>10}  {'speed cm/s':>11}")
    print("  " + "-" * 45)

    last_print = 0.0
    while time.time() - t0 < duration:
        time.sleep(0.05)
        d = get()
        samples.append(d["speed_ms"])
        elapsed = time.time() - t0
        if elapsed - last_print >= 0.5:
            print(f"  {elapsed:5.1f}  {d['lvel']:+6d}  {d['rvel']:+6d}  "
                  f"{d['speed_ms']:10.4f}  {d['speed_ms']*100:10.1f}")
            last_print = elapsed

    if samples:
        mean_spd = sum(samples) / len(samples)
        max_spd  = max(samples)
        min_spd  = min(samples)
        nonzero  = [x for x in samples if abs(x) > 0.001]
        print(f"\n  Mean speed = {mean_spd:.4f} m/s  ({mean_spd*100:.1f} cm/s)")
        print(f"  Max        = {max_spd:.4f} m/s")
        print(f"  Min        = {min_spd:.4f} m/s")
        if nonzero:
            std = (sum((x - mean_spd)**2 for x in nonzero) / len(nonzero))**0.5
            print(f"  Noise (σ)  = {std:.4f} m/s  ({std/mean_spd*100:.1f}% of mean)" if mean_spd > 0 else "")
        print(f"\n  SPEED_SCALE currently = {SPEED_SCALE}")
        print(f"  If actual speed differs, new SPEED_SCALE = {SPEED_SCALE} * (actual / {mean_spd:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Encoder Distance Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
def test_encoder_distance():
    print("\n" + "=" * 60)
    print("TEST 4 — ENCODER DISTANCE ACCURACY")
    print("=" * 60)
    print("Mark a 1.0m distance on the floor with tape.")
    print("Place the car at the START mark.")
    print("Press Enter to zero the counter, then drive to the END mark and press Enter again.")
    input()

    # Zero distance
    total_dist = 0.0
    last_t = time.time()
    last_ok = get()["packets_ok"]

    print("  [driving] Drive to the 1.0m mark now, then press Enter...")

    # Accumulate distance in a background way while user drives
    dist_thread_dist = [0.0]
    dist_thread_stop = [False]

    def dist_acc():
        nonlocal last_t
        ld = 0.0
        lt = time.time()
        while not dist_thread_stop[0]:
            time.sleep(0.02)
            now = time.time()
            dt = now - lt
            lt = now
            spd = get()["speed_ms"]
            ld += spd * dt
        dist_thread_dist[0] = ld

    dt = threading.Thread(target=dist_acc, daemon=True)
    dt.start()

    input()
    dist_thread_stop[0] = True
    dt.join()

    reported = dist_thread_dist[0]
    print(f"\n  Reported distance = {reported:.4f} m")
    print(f"  Actual distance   = 1.0000 m  (your tape mark)")

    if reported > 0:
        correction = SPEED_SCALE * (1.0 / reported)
        error_pct  = (reported - 1.0) / 1.0 * 100
        print(f"  Error             = {error_pct:+.1f}%")
        print(f"  Corrected SPEED_SCALE = {correction:.6f}  (currently {SPEED_SCALE})")

        if abs(error_pct) < 5.0:
            print("  ✓ Encoder distance is accurate (< 5% error)")
        elif abs(error_pct) < 15.0:
            print("  ⚠ Moderate error — update SPEED_SCALE to improve MPC maneuver distance")
        else:
            print("  ✗ Large error — MPC maneuver will end too early/late, update SPEED_SCALE")
    else:
        print("  ✗ No movement detected — check encoder wiring / SPEED_SCALE > 0")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Packet health check
# ═══════════════════════════════════════════════════════════════════════════════
def test_packet_health(duration=3.0):
    print("\n" + "=" * 60)
    print("TEST 0 — PACKET HEALTH CHECK")
    print("=" * 60)
    t0 = time.time()
    start_ok = get()["packets_ok"]
    start_fail = get()["checksum_fails"]
    time.sleep(duration)
    d = get()
    ok   = d["packets_ok"] - start_ok
    fail = d["checksum_fails"] - start_fail
    total = ok + fail
    rate = ok / duration

    print(f"  Packets OK       = {ok}  ({rate:.1f} Hz)")
    print(f"  Checksum fails   = {fail}")
    print(f"  Packet loss rate = {fail/total*100:.1f}%" if total > 0 else "  No packets received!")

    if rate < 10:
        print("  ✗ Packet rate is LOW — serial port may be wrong or sensor not running")
    elif fail / total > 0.05 if total > 0 else False:
        print("  ⚠ High packet loss — check USB cable / baud rate")
    else:
        print("  ✓ Packet stream looks healthy")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Sensor diagnostic tool")
    parser.add_argument("--imu", action="store_true", help="IMU tests only")
    parser.add_argument("--enc", action="store_true", help="Encoder tests only")
    args = parser.parse_args()

    run_imu = args.imu or (not args.imu and not args.enc)
    run_enc = args.enc or (not args.imu and not args.enc)

    # Start serial reader
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   SENSOR DIAGNOSTIC TOOL — JetRacer RP2040      ║")
    print("╚══════════════════════════════════════════════════╝\n")
    print(f"  Serial: {SERIAL_PORT} @ {BAUD_RATE}")
    print(f"  SPEED_SCALE: {SPEED_SCALE}\n")

    print("Waiting for first packet...", end="", flush=True)
    if not wait_for_first_packet(timeout=5.0):
        print(" TIMEOUT")
        print("No packets received. Check serial port and sensor power.")
        _stop_flag.set()
        sys.exit(1)
    print(" OK\n")

    # Always run packet health first
    test_packet_health(duration=3.0)

    gz_bias = 0.0

    if run_imu:
        gz_bias = test_imu_bias(duration=5.0)
        test_imu_drift(gz_bias=gz_bias, duration=10.0)

    if run_enc:
        test_encoder_speed(duration=15.0)
        test_encoder_distance()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if run_imu:
        print(f"  gz bias to subtract in control_loop:  {gz_bias:.4f} deg/s")
        print(f"  Add to your code:")
        print(f"    yaw_rate_rad = math.radians(imu_gz - ({gz_bias:.4f}))")
    print("\nDone.")
    _stop_flag.set()


if __name__ == "__main__":
    main()