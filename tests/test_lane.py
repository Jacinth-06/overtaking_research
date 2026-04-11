from flask import Flask, Response
import cv2
import numpy as np
import signal
import sys

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Perspective Transform Points (adjust if needed)
SRC_POINTS = np.float32([
    [100, 480],
    [540, 480],
    [250, 300],
    [390, 300]
])

DST_POINTS = np.float32([
    [150, 480],
    [490, 480],
    [150, 0],
    [490, 0]
])

M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# -----------------------------
# Jetson CSI GStreamer Pipeline
# -----------------------------
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open CSI camera")
    sys.exit(1)

# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Bird’s Eye Processing
# -----------------------------
def process_frame(frame):
    h, w = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Perspective warp (Bird's Eye View)
    warped = cv2.warpPerspective(edges, M, (w, h))

    # Convert to color for visualization
    warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # Draw center reference line
    center_x = w // 2
    cv2.line(warped_color, (center_x, 0), (center_x, h), (0, 255, 255), 2)

    return warped_color

# -----------------------------
# Streaming Generator
# -----------------------------
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        processed = process_frame(frame)

        ret2, buffer = cv2.imencode(".jpg", processed)
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

# -----------------------------
# Route
# -----------------------------
@app.route("/")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
# Cleanup
# -----------------------------
def cleanup(sig, frame):
    print("Shutting down...")
    cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)

