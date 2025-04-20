from ultralytics import YOLO
import cv2

model = YOLO("best_violence3.pt")

def process_frame(frame):
    results = model(frame)
    alert = None
    for box in results[0].boxes:
        if box.conf[0] > 0.7:
            alert = "Violence detected!"
    return results[0].plot(), alert

