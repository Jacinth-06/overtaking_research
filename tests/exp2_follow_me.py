#!/usr/bin/env python3
"""
exp2_follow_me.py — Visual-servoing demo: steer toward the largest
detected object, keeping it centered in frame.

Sequence:
  1. DETECT  — background thread runs MobileNet-SSD, caches the box
               of the LARGEST detected object (assumed closest/most
               prominent) and its horizontal offset from frame center.
  2. STEER   — proportional control: steer_val = offset * STEER_GAIN,
               clamped to [-1.0, 1.0].
  3. CREEP   — forward at DRIVE_SPEED only while something is tracked;
               stop if nothing is detected.

How to use this to validate the control loop:
  - Lift the car off the ground first. Walk left/right in front of the
    camera and confirm the front wheels turn to track you, not away.
    If it steers the wrong direction, STEER_GAIN's sign convention is
    inverted for your mounting — flip it to -abs(STEER_GAIN) or adjust
    car.steer()'s own STEER_LEFT/STEER_RIGHT convention instead.
  - If steering is twitchy, lower STEER_GAIN. If sluggish, raise it.

Run:   python3 tests/exp2_follow_me.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import cv2

from jetracer import JetRacer
from detector import detect_objects

# ── Config ────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 1280
DRIVE_SPEED  = 0.12     # same units/scale as car.forward() elsewhere in your code
STEER_GAIN   = 1.2      # how aggressively the car turns toward the target
LOOP_HZ      = 10       # detector loop rate — CPU-bound, keep modest on Nano

# ── Shared detector cache (filled by background thread) ────────────────────
_detector_cache = {"tracking": False, "label": None, "offset": 0.0}
_cache_lock     = threading.Lock()
_stop_event     = threading.Event()


def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink")


def detector_loop():
    """Runs detection continuously, caches the largest box's center offset."""
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

        if results:
            target = max(results, key=lambda r: (r[2][2] - r[2][0]))  # widest box
            label, conf, (x1, y1, x2, y2) = target
            center_x = (x1 + x2) / 2
            offset = (center_x - FRAME_WIDTH / 2) / (FRAME_WIDTH / 2)  # -1 (left) .. +1 (right)

            with _cache_lock:
                _detector_cache["tracking"] = True
                _detector_cache["label"] = label
                _detector_cache["offset"] = offset
        else:
            with _cache_lock:
                _detector_cache["tracking"] = False

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
                tracking = _detector_cache["tracking"]
                label = _detector_cache["label"]
                offset = _detector_cache["offset"]

            if tracking:
                steer_val = max(-1.0, min(1.0, offset * STEER_GAIN))
                car.steer(steer_val)
                car.forward(DRIVE_SPEED)
                print(f"[test] tracking {label} offset={offset:+.2f} steer={steer_val:+.2f}")
            else:
                car.stop()

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()
        car.close()


if __name__ == "__main__":
    main()