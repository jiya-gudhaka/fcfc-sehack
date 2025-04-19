from ultralytics import YOLO
import cv2

model = YOLO("best_violence3.pt")

def process_frame(frame):
    results = model(frame)
    return results[0].plot()
