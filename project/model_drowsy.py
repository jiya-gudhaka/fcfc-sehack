from ultralytics import YOLO
import cv2

model = YOLO("best_drow.pt")

def process_frame(frame):
    results = model(frame)
    alert = None

    # Loop through the detected boxes and check for the "drowsy" label
    for box in results[0].boxes:
        label = box.cls[0]  # Assuming class label is returned as an index (0: awake, 1: drowsy)
        confidence = box.conf[0]
        
        if label == 5 and confidence > 0.7:  # Check for "drowsy" label with high confidence
            alert = "Drowsiness detected!"
    
    # Returning both the annotated frame and the alert
    return results[0].plot(), alert


