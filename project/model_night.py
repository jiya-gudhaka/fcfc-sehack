from ultralytics import YOLO
import cv2

model = YOLO("best_night.pt")

def process_frame(frame):
    results = model(frame)
    alert = None
    for box in results[0].boxes:
        if model.names[int(box.cls[0])].lower() == "people":
            alert = "Person detected at night!"
    return results[0].plot(), alert

