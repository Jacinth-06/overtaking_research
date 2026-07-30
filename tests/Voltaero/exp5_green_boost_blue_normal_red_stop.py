#!/usr/bin/env python3
"""
exp5_green_boost_blue_normal_red_stop.py — Multi-colour state-machine drive.

State machine (priority: red > blue > green > normal):
  ┌──────────┬──────────────────────────────────────────────────────────────┐
  │ State    │ Condition & Action                                           │
  ├──────────┼──────────────────────────────────────────────────────────────┤
  │ NORMAL   │ Default — drive straight at BASE_SPEED.                      │
  │ BOOST    │ Green blob (width > GREEN_BOX_WIDTH) detected →              │
  │          │ drive straight at FAST_SPEED.                                │
  │ SLOWDOWN │ Blue blob (width > BLUE_BOX_WIDTH) detected after BOOST →   │
  │          │ drop back to BASE_SPEED (normal).                            │
  │ STOP     │ Red blob (width > RED_BOX_WIDTH) detected at any time →     │
  │          │ stop the car completely.                                     │
  │ RESUME   │ Red blob disappears → return to whichever speed was active   │
  │          │ before the stop (NORMAL or POST-BLUE slowdown).              │
  └──────────┴──────────────────────────────────────────────────────────────┘

Detailed flow:
  1. Start in NORMAL state — BASE_SPEED.
  2. If green is visible (wide enough) → switch to BOOST (FAST_SPEED).
  3. While boosting, if blue is visible (wide enough) → switch to SLOWDOWN,
     which drives at BASE_SPEED.  Blue resets the boost; another green sighting
     can trigger BOOST again.
  4. At any state, if red is visible (wide enough) → STOP immediately.
     Once red is gone, resume whatever pre-stop speed was in use.

How to validate:
  - Lift the wheels off the ground first.
  - Show green → console should report BOOST and speed increase.
  - Show blue (while boosting) → console should report SLOWDOWN / NORMAL.
  - Show red at any time → console should report STOP and wheels stop.
  - Remove red → car resumes immediately.

Run:   python3 tests/Voltaero/exp5_green_boost_blue_normal_red_stop.py
Stop early any time with Ctrl+C — the finally block stops the car.
"""

import threading
import time

import cv2

from jetracer import JetRacer
from detector import detect_objects, detect_green_objects, detect_blue_objects

# ── Config ────────────────────────────────────────────────────────────────
GREEN_BOX_WIDTH = 250      # px — green blob width that triggers BOOST
BLUE_BOX_WIDTH  = 200      # px — blue blob width that cancels BOOST → NORMAL
RED_BOX_WIDTH   = 150      # px — red blob width that triggers STOP
BASE_SPEED      = 0.12     # normal forward speed
FAST_SPEED      = 0.22     # boosted forward speed when green seen
LOOP_HZ         = 10

# ── Shared detector cache (filled by background thread) ────────────────────
_detector_cache = {
    "green": False,
    "blue":  False,
    "red":   False,
}
_cache_lock = threading.Lock()
_stop_event = threading.Event()


def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink")


def detector_loop():
    """Detects green, blue, and red objects each frame and updates the shared cache."""
    print("[detector] thread started")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[detector] failed to open CSI camera")
        return

    while not _stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue

        green_results = detect_green_objects(frame)
        blue_results  = detect_blue_objects(frame)
        red_results   = detect_objects(frame)      # detect_objects returns red blobs

        green_seen = any((x2 - x1) > GREEN_BOX_WIDTH for _, _, (x1, y1, x2, y2) in green_results)
        blue_seen  = any((x2 - x1) > BLUE_BOX_WIDTH  for _, _, (x1, y1, x2, y2) in blue_results)
        red_seen   = any((x2 - x1) > RED_BOX_WIDTH   for _, _, (x1, y1, x2, y2) in red_results)

        with _cache_lock:
            _detector_cache["green"] = green_seen
            _detector_cache["blue"]  = blue_seen
            _detector_cache["red"]   = red_seen

        time.sleep(1.0 / LOOP_HZ)

    cap.release()


def main():
    car = JetRacer()
    car.arm(delay=3)

    dt = threading.Thread(target=detector_loop, daemon=True)
    dt.start()

    time.sleep(0.5)  # let the first detection cycle land before we trust the cache

    # Drive state: tracks whether we are in BOOST mode (green triggered) or NORMAL
    boosting = False   # True after green, reset to False when blue seen

    print("[test] Running. Ctrl+C to quit.")
    try:
        while True:
            with _cache_lock:
                green = _detector_cache["green"]
                blue  = _detector_cache["blue"]
                red   = _detector_cache["red"]

            # ── Colour priority: red > blue > green > normal ──────────────

            if red:
                # RED — highest priority: stop regardless of any other colour
                car.stop()
                print("[test] STOP — red detected")

            elif blue and boosting:
                # BLUE seen while boosting → slow back to normal speed
                boosting = False
                car.steer(0.0)
                car.forward(BASE_SPEED)
                print(f"[test] SLOWDOWN — blue detected, speed={BASE_SPEED:.2f}")

            elif green and not boosting:
                # GREEN seen in normal mode → boost speed
                boosting = True
                car.steer(0.0)
                car.forward(FAST_SPEED)
                print(f"[test] BOOST — green detected, speed={FAST_SPEED:.2f}")

            else:
                # No trigger colour present (or already in the right state) →
                # maintain current speed (FAST if boosting, BASE otherwise)
                speed = FAST_SPEED if boosting else BASE_SPEED
                state = "BOOSTING" if boosting else "NORMAL"
                car.steer(0.0)
                car.forward(speed)
                print(f"[test] {state} speed={speed:.2f}")

            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        car.stop()
        _stop_event.set()
        car.close()


if __name__ == "__main__":
    main()
