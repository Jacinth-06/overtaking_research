#!/usr/bin/env python3
"""
exp3_speed_up_green.py — Drive straight, boost speed when green is seen.

Sequence:
  1. CREEP  — drive forward at BASE_SPEED continuously, no steering (straight).
  2. DETECT — background thread finds the widest green blob each cycle.
  3. BOOST  — while a green blob's bounding-box width exceeds
              GREEN_BOX_WIDTH px, drive at FAST_SPEED instead.
  4. NORMAL — once no box exceeds the threshold, drop back to BASE_SPEED.

How to use this to validate:
  - Lift the wheels off the ground first.
  - Hold something green in front of the camera, move it closer and
    confirm the console switches from NORMAL to BOOST and the reported
    speed value changes. If it triggers too early/late, adjust
    GREEN_BOX_WIDTH.

Run:   python3 tests/exp3_speed_up_green.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import cv2

from jetracer import JetRacer
from detector import detect_green_objects

# ── Config ────────────────────────────────────────────────────────────────
GREEN_BOX_WIDTH = 250      # px — tune live, proxy for "close enough to boost"
BASE_SPEED      = 0.12     # normal forward speed
FAST_SPEED      = 0.22     # boosted forward speed while green is close
LOOP_HZ         = 10

# ── Shared detector cache (filled by background thread) ────────────────────
_detector_cache = {"boost": False}
_cache_lock     = threading.Lock()
_stop_event     = threading.Event()


def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink")


def detector_loop():
    """Runs green detection continuously, caches whether to boost speed."""
    print("[detector] thread started")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[detector] failed to open CSI camera")
        return

    while not _stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue

        results = detect_green_objects(frame)
        boost = any((x2 - x1) > GREEN_BOX_WIDTH for _, _, (x1, y1, x2, y2) in results)

        with _cache_lock:
            _detector_cache["boost"] = boost

        time.sleep(1.0 / LOOP_HZ)

    cap.release()


def main():
    car = JetRacer()
    car.arm(delay=3)

    dt = threading.Thread(target=detector_loop, daemon=True)
    dt.start()

    time.sleep(0.5)  # let the first detection cycle land before we trust the cache

    print("[test] Running. Ctrl+C to quit.")
    try:
        while True:
            with _cache_lock:
                boost = _detector_cache["boost"]

            speed = FAST_SPEED if boost else BASE_SPEED
            car.steer(0.0)
            car.forward(speed)
            print(f"[test] {'BOOST' if boost else 'NORMAL'} speed={speed:.2f}")

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()
        car.close()


if __name__ == "__main__":
    main()
