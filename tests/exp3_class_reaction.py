#!/usr/bin/env python3
"""
exp3_class_reaction.py — Class-conditional reaction demo.

Sequence:
  1. DETECT  — background thread runs MobileNet-SSD, caches all detected
               classes whose box width exceeds CLOSE_BOX_WIDTH.
  2. REACT   — decide behavior based on WHICH classes are close, not
               just whether something is close:
                 "person" close  -> FULL STOP
                 anything else   -> SLOW creep
                 nothing close   -> NORMAL creep

How to use this to validate the reaction logic:
  - Lift the car off the ground. Step into frame close to the camera —
    confirm FULL STOP. Step out, hold up a chair/bottle at the same
    distance — confirm SLOW creep instead of a stop. Clear the frame —
    confirm NORMAL creep resumes.
  - CLOSE_BOX_WIDTH is a rough distance proxy, not calibrated depth —
    see tests/testimu_encoder.py for the pattern of validating a sensor
    reading against a real-world tape measurement; the same idea
    applies here (walk toward the camera, note box_width vs. actual
    distance, adjust CLOSE_BOX_WIDTH accordingly).

Run:   python3 tests/exp3_class_reaction.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import cv2

from jetracer import JetRacer
from detector import detect_objects

# ── Config ────────────────────────────────────────────────────────────────
CLOSE_BOX_WIDTH = 300      # px — see docstring for calibration approach
NORMAL_SPEED    = 0.20     # same units/scale as car.forward() elsewhere in your code
SLOW_SPEED      = 0.10
LOOP_HZ         = 10       # detector loop rate — CPU-bound, keep modest on Nano

# ── Shared detector cache (filled by background thread) ────────────────────
_detector_cache = {"close_labels": []}
_cache_lock     = threading.Lock()
_stop_event     = threading.Event()


def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink")


def detector_loop():
    """Runs detection continuously, caches labels of all 'close' objects."""
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
        close_labels = [label for label, conf, (x1, y1, x2, y2) in results
                        if (x2 - x1) > CLOSE_BOX_WIDTH]

        with _cache_lock:
            _detector_cache["close_labels"] = close_labels

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
                close_labels = list(_detector_cache["close_labels"])

            if "person" in close_labels:
                print("[test] PERSON CLOSE — FULL STOP")
                car.stop()
            elif close_labels:
                print(f"[test] OBJECT CLOSE ({close_labels[0]}) — SLOWING")
                car.forward(SLOW_SPEED)
            else:
                car.forward(NORMAL_SPEED)

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()
        car.close()