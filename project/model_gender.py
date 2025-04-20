# model_gender.py

import cv2
from ultralytics import YOLO

# Load the trained model once (global scope)
model = YOLO("best_gender.pt")

def process_frame(frame):
    results = model(frame)
    alert = None
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if model.names[class_id].lower() == "male":
            alert = "Man detected!"
    return results[0].plot(), alert

