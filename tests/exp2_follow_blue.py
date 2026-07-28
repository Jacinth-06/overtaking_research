#!/usr/bin/env python3
"""
exp2_follow_blue.py — Camera-triggered blue-object following demo.

Sequence:
  1. DETECT — background thread finds the largest blue blob each cycle,
              caches its centroid x-position (in [-1, 1], 0 = frame centre).
  2. STEER  — proportional controller steers toward the blob's centroid.
  3. DRIVE  — creeps forward at DRIVE_SPEED while a blob is visible.
  4. LOST   — stops if no blue blob has been seen for LOST_TIMEOUT seconds.

How to use this to validate:
  - Lift the wheels off the ground first.
  - Hold something blue in front of the camera, move it left/right and
    confirm the steering follows it (check console print of `error`).
  - Pull the blue object away / out of frame and confirm the car stops
    within LOST_TIMEOUT seconds.

Run:   python3 tests/exp2_follow_blue.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import cv2

from jetracer import JetRacer
from detector import detect_blue_objects

# ── Config ────────────────────────────────────────────────────────────────
DRIVE_SPEED   = 0.12     # forward speed while tracking
KP            = 0.9      # steering proportional gain — tune live
LOST_TIMEOUT  = 0.8      # seconds with no detection before stopping
LOOP_HZ       = 10

# ── Shared detector cache (filled by background thread) ────────────────────
_detector_cache = {"found": False, "error": 0.0}
_cache_lock     = threading.Lock()
_stop_event     = threading.Event()


def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink")


def detector_loop():
    """Runs blue detection continuously, caches centroid error of the largest blob."""
    print("[detector] thread started")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[detector] failed to open CSI camera")
        return

    while not _stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        results = detect_blue_objects(frame)

        if results:
            # Largest blob by bounding-box area
            _, _, (x1, y1, x2, y2) = max(
                results, key=lambda r: (r[2][2] - r[2][0]) * (r[2][3] - r[2][1])
            )
            cx = (x1 + x2) / 2.0
            error = (cx - w / 2.0) / (w / 2.0)   # normalised [-1, 1]
            with _cache_lock:
                _detector_cache["found"] = True
                _detector_cache["error"] = error
        else:
            with _cache_lock:
                _detector_cache["found"] = False

        time.sleep(1.0 / LOOP_HZ)

    cap.release()


def main():
    car = JetRacer()
    car.arm(delay=3)

    dt = threading.Thread(target=detector_loop, daemon=True)
    dt.start()

    time.sleep(0.5)  # let the first detection cycle land before we trust the cache

    print("[test] Running. Ctrl+C to quit.")
    last_seen = time.time()
    try:
        while True:
            with _cache_lock:
                found = _detector_cache["found"]
                error = _detector_cache["error"]

            now = time.time()
            if found:
                last_seen = now
                steer = max(-1.0, min(1.0, KP * error))
                car.steer(steer)
                car.forward(DRIVE_SPEED)
                print(f"[test] FOLLOWING error={error:+.2f} steer={steer:+.2f}")
            elif now - last_seen > LOST_TIMEOUT:
                car.stop()
                print("[test] LOST — stopped")
            # else: briefly out of frame, hold last command (no-op this tick)

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()
        car.close()


if __name__ == "__main__":
    main()
