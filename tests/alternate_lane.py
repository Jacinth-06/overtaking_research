from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import signal
import sys
import time
from base_ctrl import BaseController

# -----------------------------
# Configuration & Calibration
# -----------------------------
FRAME_WIDTH, FRAME_HEIGHT = 580, 440
BASE_SPEED = 0.12          # Slightly lower base speed helps with sharp turns
STEER_SENSITIVITY = 0.0025 # INCREASED from 0.0012



# Perspective Transform
SRC_POINTS = np.float32([[50, 480], [590, 480], [240, 300], [400, 300]])
DST_POINTS = np.float32([[150, 480], [490, 480], [150, 0], [490, 0]])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Tape Fingerprint Parameters
EXPECTED_LANE_WIDTH = 340 
TAPE_WIDTH_PIXELS = 14     
TAPE_MARGIN = 6
CONFIDENCE_THRESHOLD = 500 
SEARCH_MARGIN = 50         

# PID Variables
Kp, Ki, Kd = 0.7, 0.0, 0.15 # INCREASED Kp and Kd
last_error = 0
integral = 0

# -----------------------------
# Hardware Initialization
# -----------------------------
def is_raspberry_pi5():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return "Raspberry Pi 5" in f.read()
    except: return False

serial_device = "/dev/ttyAMA0" if is_raspberry_pi5() else "/dev/serial0"
base = BaseController(serial_device, 115200)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
))
picam2.start()

app = Flask(__name__)

# -----------------------------
# Core Logic
# -----------------------------

def find_tape_center(histogram, start_x, end_x):
    best_center = None
    max_score = 0
    for x in range(start_x, end_x - TAPE_WIDTH_PIXELS - TAPE_MARGIN):
        score = histogram[x] + histogram[x + TAPE_WIDTH_PIXELS]
        if score > max_score and score > CONFIDENCE_THRESHOLD:
            max_score = score
            best_center = x + (TAPE_WIDTH_PIXELS // 2)
    return best_center

def process_and_drive(frame):
    global last_error, integral
    h, w = frame.shape[:2]
    img_center = w // 2

    # 1. Reverting to Canny Edge Detection (as requested)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    warped = cv2.warpPerspective(edges, M, (w, h))

    # 2. Histogram with Strict Search Zones
    # We look at the bottom 40% of the image
    hist = np.sum(warped[int(h*0.6):, :], axis=0)
    
    # CRITICAL: Narrow zones prevent picking up noise in the middle
    left_x = find_tape_center(hist, 0, int(w * 0.35))
    right_x = find_tape_center(hist, int(w * 0.65), w - 1)

    # 3. Lane Reconstruction
    l_valid, r_valid = left_x is not None, right_x is not None
    
    if l_valid and r_valid:
        lane_center = (left_x + right_x) // 2
        l_col, r_col = (0, 255, 0), (0, 255, 0)
    elif l_valid:
        lane_center = left_x + (EXPECTED_LANE_WIDTH // 2)
        right_x = left_x + EXPECTED_LANE_WIDTH
        l_col, r_col = (0, 255, 255), (0, 0, 150)
    elif r_valid:
        lane_center = right_x - (EXPECTED_LANE_WIDTH // 2)
        left_x = right_x - EXPECTED_LANE_WIDTH
        l_col, r_col = (0, 0, 150), (0, 255, 255)
    else:
        # If totally lost, stop
        base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # 4. Aggressive Control Math
    error = lane_center - img_center
    integral += error
    derivative = error - last_error
    
    # Boosted steering adjustment
    # Increasing STEER_SENSITIVITY makes the speed difference between wheels larger
    steering_adj = error * STEER_SENSITIVITY 
    
    # Pivot Turn Logic: Allow wheels to go down to -0.1 for sharp "on-the-spot" turns
    # and up to 0.5 for a powerful push
    left_wheel = np.clip(BASE_SPEED + steering_adj, -0.1, 0.5)
    right_wheel = np.clip(BASE_SPEED - steering_adj, -0.1, 0.5)
    
    base.send_command({"T": 1, "L": round(left_wheel, 3), "R": round(right_wheel, 3)})
    last_error = error

    # 5. Visualization
    viz = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.line(viz, (img_center, 0), (img_center, h), (255, 255, 0), 2)
    cv2.circle(viz, (lane_center, h - 80), 15, (255, 120, 0), -1) 
    cv2.circle(viz, (left_x, h - 40), 20, l_col, -1)
    cv2.circle(viz, (right_x, h - 40), 20, r_col, -1)
    
    cv2.putText(viz, f"Error: {error}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return viz

def generate_frames():
    while True:
        frame = picam2.capture_array()
        processed = process_and_drive(frame)
        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/start")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")




# 1. Add a control flag at the top of your script
is_running = False


@app.route("/turn_left")
def turn_left():
    # spin left in place
    base.send_command({"T": 1, "L": -0.1, "R": 0.3})
    return "Turning left"

@app.route("/start_robot")
def start_robot():
    global is_running
    is_running = True
    return "Robot movement started!"

@app.route("/stop_robot")
def stop_robot():
    global is_running
    is_running = False
    # Send stop command immediately
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    return "Robot stopped!"

def generate_frames():
    while True:
        frame = picam2.capture_array()
        
        # Only process movement if the flag is True
        if is_running:
            processed = process_and_drive(frame)
        else:
            processed = frame # Just show the raw camera feed
            
        ret, jpeg = cv2.imencode(".jpg", processed)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

def cleanup(sig, frame):
    base.send_command({"T": 1, "L": 0.0, "R": 0.0})
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)