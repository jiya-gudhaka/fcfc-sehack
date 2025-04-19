import cv2
import base64
import datetime
import time
import streamlit as st
from ultralytics import YOLO

# Load your night-vision YOLO model
model = YOLO("best_night.pt")

# Open webcam (adjust index if needed)
cap = cv2.VideoCapture(0)

st.title("🌙 Night Vision Person Detection")
st.markdown("Using YOLOv8 + enhanced contrast for night-time webcam detection.\n\n"
            "**Capturing 1 frame every 3 seconds.**")
image_placeholder = st.empty()
alert_placeholder = st.empty()

# One frame every 3 seconds
FRAME_RATE = 1 / 3
DELAY = 3.0  # seconds

def enhance_night_vision(frame):
    """Convert frame to grayscale and enhance contrast for better night visibility."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)  # Histogram equalization to enhance contrast
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Failed to capture video frame.")
        break

    # Enhance for night vision
    enhanced_frame = enhance_night_vision(frame)

    # Run YOLO detection
    results = model(enhanced_frame)[0]

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
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(enhanced_frame, f"{label} {round(confidence, 2)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Capture alert info
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

    # Display the video feed
    frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    image_placeholder.image(frame_rgb, channels="RGB")

    # Show alert info
    if person_detected:
        alert_placeholder.markdown(
            f"🚨 **Person detected!**\n- Label: `{alert_data['label']}`\n- Confidence: **{alert_data['confidence']}**\n- Time: `{alert_data['time']}`"
        )
    else:
        alert_placeholder.empty()

    # Wait before processing next frame
    time.sleep(DELAY)
