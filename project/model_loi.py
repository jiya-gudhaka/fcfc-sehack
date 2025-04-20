from ultralytics import YOLO
import cv2

model = YOLO("best_loi.pt")

def process_frame(frame):
    results = model(frame)
    alert = None
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if model.names[class_id].lower() == "person":
            alert = "Loitering detected!"
    return results[0].plot(), alert

