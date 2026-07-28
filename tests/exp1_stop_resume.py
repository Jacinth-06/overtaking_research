#!/usr/bin/env python3
"""
exp1_stop_resume.py — Camera-triggered stop & resume demo.

Sequence:
  1. CREEP    — drive forward at DRIVE_SPEED continuously.
  2. DETECT   — background thread runs MobileNet-SSD on the CSI camera,
                caches the widest detected bounding box each cycle.
  3. STOP     — the moment any detected box exceeds CLOSE_BOX_WIDTH px,
                stop immediately.
  4. RESUME   — once no box exceeds the threshold, resume creeping.

How to use this to validate the detector:
  - Walk toward the lifted (off-ground) car holding any COCO/VOC-class
    object (person, chair, bottle...) and confirm STOP fires at a
    sensible real-world distance. If it fires too early/late, adjust
    CLOSE_BOX_WIDTH — this is a proxy for distance, not a calibrated
    measurement (see exp3_class_reaction.py notes on this).
  - Confirm RESUME fires promptly once you step out of frame.

Run:   python3 tests/exp1_stop_resume.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import cv2

from jetracer import JetRacer
from detector import detect_objects

# ── Config ────────────────────────────────────────────────────────────────
CLOSE_BOX_WIDTH = 300      # px — tune live; see docstring
DRIVE_SPEED     = 0.12     # same units/scale as car.forward() elsewhere in your code
LOOP_HZ         = 10       # detector loop rate — CPU-bound, keep modest on Nano

# ── Shared detector cache (filled by background thread) ────────────────────
_detector_cache = {"blocked": False}
_cache_lock     = threading.Lock()
_stop_event     = threading.Event()


def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink")


def detector_loop():
    """Runs detection continuously, caches whether anything is 'close'."""
    print("[detector] thread started")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[detector] failed to open CSI camera")
        return

    while not _stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue

        results = detect_objects(frame)
        blocked = any((x2 - x1) > CLOSE_BOX_WIDTH for _, _, (x1, y1, x2, y2) in results)

        with _cache_lock:
            _detector_cache["blocked"] = blocked

        time.sleep(1.0 / LOOP_HZ)

    cap.release()


def main():
    car = JetRacer()
    car.arm(delay=3)

    dt = threading.Thread(target=detector_loop, daemon=True)
    dt.start()

    time.sleep(0.5)  # let the first detection cycle land before we trust the cache

    print("[test] Running. Ctrl+C to quit.")
    was_blocked = False
    try:
        while True:
            with _cache_lock:
                blocked = _detector_cache["blocked"]

            if blocked and not was_blocked:
                print("[test] OBSTACLE — STOP")
                car.stop()
            elif not blocked:
                car.forward(DRIVE_SPEED)

            was_blocked = blocked
            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()
        car.close()


if __name__ == "__main__":
    main()