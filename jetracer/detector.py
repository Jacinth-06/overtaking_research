import cv2
import numpy as np

MIN_AREA = 500

LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])


def detect_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = mask1 | mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        conf = min(1.0, area / (frame.shape[0] * frame.shape[1]) * 20)
        results.append(("red_object", float(conf), (x, y, x + w, y + h)))

    return results


# ── Blue detector ────────────────────────────────────────────────────────
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])


def detect_blue_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        conf = min(1.0, area / (frame.shape[0] * frame.shape[1]) * 20)
        results.append(("blue_object", float(conf), (x, y, x + w, y + h)))

    return results


# ── Green detector ───────────────────────────────────────────────────────
LOWER_GREEN = np.array([40, 100, 50])
UPPER_GREEN = np.array([85, 255, 255])


def detect_green_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        conf = min(1.0, area / (frame.shape[0] * frame.shape[1]) * 20)
        results.append(("green_object", float(conf), (x, y, x + w, y + h)))

    return results
