from ultralytics import YOLO
import cv2

model = YOLO("best_crowd.pt")

def process_frame(frame):
    results = model(frame)
    person_count = sum(1 for box in results[0].boxes
                       if model.names[int(box.cls[0])].lower() == "person")
    alert = "Crowd detected!" if person_count >= 2 else None
    return results[0].plot(), alert

