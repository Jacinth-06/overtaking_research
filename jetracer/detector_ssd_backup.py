import cv2
import numpy as np

import os
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT = os.path.join(_MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL    = os.path.join(_MODEL_DIR, "MobileNetSSD_deploy.caffemodel")
CONF_THRESH = 0.5

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


def detect_objects(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > CONF_THRESH:
            idx = int(detections[0, 0, i, 1])
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
            results.append((CLASSES[idx], float(conf), tuple(box)))
    return results

