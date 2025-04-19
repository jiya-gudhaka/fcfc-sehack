import streamlit as st
import cv2
import base64
import datetime
from ultralytics import YOLO
import numpy as np
import time

# Load YOLO model once
model = YOLO("best_drow.pt")

# Initialize webcam only once
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

# Streamlit UI setup
st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("😴 Real-Time Drowsiness Detection")
st.markdown("Analyzing webcam feed for signs of **drowsiness**...")

placeholder = st.empty()

def detect_drowsiness(frame):
    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = float(box.conf[0])

        if label.lower() == "drowsy" and confidence > 0.6:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return {
                "label": label,
                "confidence": round(confidence, 2),
                "time": str(datetime.datetime.now()),
                "image": img_base64
            }
    return None

# Detection loop (limited for performance; can be adjusted)
for _ in range(1000):  # Use a large number or while loop if preferred
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("❌ Failed to read from webcam.")
        break

    alert = detect_drowsiness(frame)

    with placeholder.container():
        if alert:
            st.error("🚨 Drowsiness Detected!")
            st.markdown(f"**Label:** {alert['label']}")
            st.markdown(f"**Confidence:** {alert['confidence']}")
            st.markdown(f"**Time:** {alert['time']}")
            img_bytes = base64.b64decode(alert["image"])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            st.success("✅ No drowsiness detected.")

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    time.sleep(3)  # Adjust delay as needed (3 seconds between checks)

# Optional cleanup
# st.session_state.cap.release()
