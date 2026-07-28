"""
exp1_stop_resume.py
Car creeps forward. Stops when an object gets close. Resumes when clear.
"""
import cv2, time
from jetracer import JetRacer
from detector import detect_objects

CLOSE_BOX_WIDTH = 300
CREEP_SPEED = 0.12

def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
             "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
             "videoconvert ! video/x-raw, format=BGR ! appsink")

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
car = JetRacer()
car.arm(delay=3)

print("Exp1: Stop & Resume demo. Ctrl+C to quit.")
was_blocked = False
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        results = detect_objects(frame)
        blocked = any((x2 - x1) > CLOSE_BOX_WIDTH for _, _, (x1, y1, x2, y2) in results)

        if blocked and not was_blocked:
            print("OBSTACLE — STOP")
            car.stop()
        elif not blocked:
            car.forward(CREEP_SPEED)
        was_blocked = blocked
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    car.stop(); car.close(); cap.release()
