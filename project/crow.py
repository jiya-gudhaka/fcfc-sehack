import streamlit as st
import cv2
import base64
import datetime
from ultralytics import YOLO
import numpy as np
import time

# Load YOLO model once
model = YOLO("best_crowd.pt")  # Path to your crowd detection model

# Initialize webcam only once
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

# Streamlit UI setup
st.set_page_config(page_title="Crowd Detection", layout="wide")
st.title("👥 Real-Time Crowd Detection")
st.markdown("Analyzing webcam feed for **crowd detection**...")

placeholder = st.empty()

def detect_crowd(frame):
    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label.lower() == "crowd" and confidence > 0.6:
            # Draw bounding box on the image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Encode the frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return {
                "label": label,
                "confidence": round(confidence, 2),
                "time": str(datetime.datetime.now()),
                "image": img_base64,
                "bbox": [x1, y1, x2, y2]  # Bounding box coordinates
            }
    return None

# Detection loop (run once every 3 seconds)
for _ in range(1000):  # Adjust the number of iterations as needed
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("❌ Failed to read from webcam.")
        break

    alert = detect_crowd(frame)

    with placeholder.container():
        if alert:
            st.error("🚨 Crowd Detected!")
            st.markdown(f"**Label:** {alert['label']}")
            st.markdown(f"**Confidence:** {alert['confidence']}")
            st.markdown(f"**Time:** {alert['time']}")
            st.markdown(f"**Bounding Box:** {alert['bbox']}")
            
            # Decode and display image
            img_bytes = base64.b64decode(alert["image"])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            st.success("✅ No crowd detected.")

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    time.sleep(3)  # Wait 3 seconds before processing the next frame

# Optional: release the webcam when Streamlit app stops
# st.session_state.cap.release()
