#!/usr/bin/env python3
"""
testimu_encoder.py — Sanity check for wheel-encoder distance and gyro heading.

Sequence:
  1. FORWARD — drive straight until the encoder reports 100 cm traveled.
  2. TURN    — steer hard right while moving, integrating gyro yaw until
               the car has rotated ~90° (a completed right turn).
  3. STOP    — stop immediately once the turn target is reached.

How to use this to validate your sensors:
  - Phase 1: measure the ACTUAL distance traveled with a tape measure and
    compare it to the "encoder distance" printed when phase 1 ends. If they
    disagree, SPEED_SCALE (the wheel-speed-to-m/s calibration) needs
    recalibrating.
  - Phase 2: mark the car's heading before the turn (e.g. a line of tape
    on the floor) and compare to the heading after it stops. If the real
    turn isn't ~90°, the gyro reading is off — could be axis (gz may not
    be your actual yaw axis if the IMU is mounted rotated), scale, or
    bias drift.

Sign conventions below (STEER_RIGHT, TURN_SIGN) are assumptions — flip
them if the car turns left instead, or if the reported yaw runs in the
opposite direction to the real turn.

Run:   python testimu_encoder.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import serial

from jetracer import JetRacer

# ── Config ────────────────────────────────────────────────────────────────
FORWARD_DISTANCE_M = 1.00      # 100 cm
TURN_TARGET_DEG    = 90.0      # how far the turn should rotate the heading
TURN_SIGN          = 1.0       # +1.0 if a right turn should increase yaw_deg; flip to -1.0 if gz's sign is reversed for your IMU mounting
STEER_RIGHT        = 1.0       # full steering lock to the right — flip to -1.0 if this turns left instead
DRIVE_SPEED        = 0.15      # same units/scale as car.forward() elsewhere in your code
LOOP_HZ            = 50

STOP_DISTANCE_MM   = 300.0     # safety: abort phase 1 if something is this close in front
SERIAL_PORT        = "/dev/ttyACM0"
SERIAL_BAUD        = 115200
SPEED_SCALE        = 0.00748   # calibrated: 1.12 m actual / 10.8 m reported (from version3.py)

# ── Shared sensor caches (filled by background thread) ─────────────────────
_imu_cache      = {"gz": 0.0}
_encoder_cache  = {"speed": 0.0, "distance": 0.0}
_cache_lock     = threading.Lock()
_stop_event     = threading.Event()


def sensor_loop():
    """Parses the same Waveshare RP2040 packet format used in version3.py:
    AA 55 [size] [...39 data bytes...] [checksum]"""
    print("[sensors] thread started")
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    except Exception as e:
        print(f"[sensors] failed to open serial: {e}")
        return

    HEAD1, HEAD2 = 0xAA, 0x55
    total_distance = 0.0
    last_time = time.time()

    while not _stop_event.is_set():
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
                continue  # garbage — resync
            rest = ser.read(frame_size - 3)
            if len(rest) != frame_size - 3:
                continue
            frame = bytes([HEAD1, HEAD2, frame_size]) + rest

            if (sum(frame[:-1]) & 0xFF) != frame[-1]:
                continue  # bad checksum

            gz = int.from_bytes(frame[8:10], "big", signed=True) / 32768 * 2000  # deg/s

            lvel = int.from_bytes(frame[34:36], "big", signed=True)
            rvel = int.from_bytes(frame[36:38], "big", signed=True)

            now = time.time()
            dt = now - last_time
            if dt > 0:
                speed_ms = ((lvel + rvel) / 2.0) * SPEED_SCALE
                total_distance += speed_ms * dt
                with _cache_lock:
                    _encoder_cache["speed"] = speed_ms
                    _encoder_cache["distance"] = total_distance
            last_time = now

            with _cache_lock:
                _imu_cache["gz"] = gz

        except Exception as e:
            print(f"[sensors] error: {e}")
            time.sleep(0.1)


def main():
    car = JetRacer(init_lidar=True)
    car.arm(delay=3)

    st = threading.Thread(target=sensor_loop, daemon=True)
    st.start()

    time.sleep(0.5)  # let the first few sensor packets arrive before we trust the cache

    try:
        # ── Phase 1: drive straight 100 cm ──────────────────────────────
        print(f"[test] Phase 1: forward {FORWARD_DISTANCE_M * 100:.0f} cm")
        with _cache_lock:
            start_dist = _encoder_cache["distance"]

        car.steer(0.0)
        car.forward(DRIVE_SPEED)

        traveled = 0.0
        while traveled < FORWARD_DISTANCE_M:
            with _cache_lock:
                traveled = _encoder_cache["distance"] - start_dist

            # Safety: abort if something is right in front of the car
            try:
                scan = car.lidar_scan(samples=100)
                front = [d for a, d in scan.items() if (a >= 320 or a <= 40) and d > 10]
                if front and min(front) < STOP_DISTANCE_MM:
                    print("[test] ABORT: obstacle detected during forward phase")
                    car.stop()
                    return
            except Exception:
                pass  # lidar not available / not critical for this test

            time.sleep(1.0 / LOOP_HZ)

        car.stop()
        print(f"[test] Phase 1 done. Encoder reports {traveled * 100:.1f} cm traveled.")
        time.sleep(0.5)

        # ── Phase 2: turn right ~90° ─────────────────────────────────────
        # ── Phase 2: turn right ~90° ─────────────────────────────────────
        print(f"[test] Phase 2: turning right {TURN_TARGET_DEG:.0f}°")
        
        # CRITICAL RESET STEP: Wipe out any drift accumulated during Phase 1
        yaw_deg = 0.0
        
        # CRITICAL RESET STEP: Flush out old timestamps so 'dt' starts clean
        last_time = time.time() 

        car.steer(STEER_RIGHT)
        car.forward(DRIVE_SPEED)

        while abs(yaw_deg) < TURN_TARGET_DEG:
            now = time.time()
            dt = now - last_time
            last_time = now

            with _cache_lock:
                gz = _imu_cache["gz"]

            # Software integration of the reset values
            yaw_deg += TURN_SIGN * gz * dt
            time.sleep(1.0 / LOOP_HZ)

        # ── Phase 3: stop immediately on completion ──────────────────────
        car.stop()
        print(f"[test] Phase 2 done. Gyro reports {yaw_deg:.1f}° turned.")
        print("[test] Stopped.")
        print()
        print("Now compare:")
        print(f"  - Phase 1 encoder distance ({traveled * 100:.1f} cm) vs. actual tape measurement")
        print(f"  - Phase 2 gyro heading change ({yaw_deg:.1f}°) vs. actual heading change")

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()


if __name__ == "__main__":
    main()