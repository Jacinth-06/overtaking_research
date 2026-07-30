#!/usr/bin/env python3
"""
exp4_stop_red.py — Drive forward, stop when red is seen, resume when clear.

Sequence:
  1. DRIVE  — car drives forward at BASE_SPEED continuously (straight).
  2. DETECT — background thread detects the largest red blob each cycle.
  3. STOP   — while a red blob's bounding-box width exceeds RED_BOX_WIDTH px,
              the car stops (speed = 0).
  4. RESUME — once no red blob exceeds the threshold, the car drives again at
              BASE_SPEED immediately.

How to use this to validate:
  - Lift the wheels off the ground first.
  - Hold something red in front of the camera and confirm the console switches
    from DRIVE to STOP and the wheels stop.
  - Remove the red object and confirm the car resumes (console → DRIVE).
  - Adjust RED_BOX_WIDTH if the trigger fires too early or too late.

Run:   python3 tests/Voltaero/exp4_stop_red.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import cv2

from jetracer import JetRacer
from detector import detect_objects   # detect_objects returns red blobs

# ── Config ────────────────────────────────────────────────────────────────
RED_BOX_WIDTH = 150      # px — minimum blob width that triggers a stop
BASE_SPEED    = 0.12     # normal forward speed
LOOP_HZ       = 10

# ── Shared detector cache (filled by background thread) ────────────────────
_detector_cache = {"red_seen": False}
_cache_lock     = threading.Lock()
_stop_event     = threading.Event()


def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink")


def detector_loop():
    """Runs red detection continuously, caches whether a large red blob is present."""
    print("[detector] thread started")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[detector] failed to open CSI camera")
        return

    while not _stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue

        results = detect_objects(frame)   # returns red blobs
        red_seen = any((x2 - x1) > RED_BOX_WIDTH for _, _, (x1, y1, x2, y2) in results)

        with _cache_lock:
            _detector_cache["red_seen"] = red_seen

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
                red_seen = _detector_cache["red_seen"]

            if red_seen:
                car.stop()
                print("[test] STOP — red detected")
            else:
                car.steer(0.0)
                car.forward(BASE_SPEED)
                print(f"[test] DRIVE speed={BASE_SPEED:.2f}")

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()
        car.close()


if __name__ == "__main__":
    main()
