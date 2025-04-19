import cv2
import base64
import datetime
import time
import streamlit as st
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best_loi.pt")  # Replace with your model path

# Open webcam
cap = cv2.VideoCapture(0)

# Streamlit UI
st.title("👀 Real-Time Person Detection")
st.markdown("Running YOLOv8 on webcam feed. Detecting `person` class only.")
image_placeholder = st.empty()
alert_placeholder = st.empty()

FRAME_RATE = 3  # 3 frames per second
DELAY = 1.0 / FRAME_RATE  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame from webcam.")
        break

    # Run YOLO detection
    results = model(frame)[0]

    person_detected = False
    alert_data = {}

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label.lower() == "person" and confidence > 0.6:
            person_detected = True

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode frame for alert snapshot (optional, base64)
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            alert_data = {
                "label": label,
                "confidence": round(confidence, 2),
                "time": str(datetime.datetime.now()),
                "image": img_base64,
                "bbox": [x1, y1, x2, y2]
            }
            break  # Only send one alert per frame

    # Show the frame with boxes
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_placeholder.image(frame_rgb, channels="RGB")

    # Display alert if a person was detected
    if person_detected:
        alert_placeholder.markdown(
            f"🚨 **Alert:** `{alert_data['label']}` detected with **{alert_data['confidence']}** confidence at `{alert_data['time']}`"
        )
    else:
        alert_placeholder.empty()

    time.sleep(DELAY)
