import streamlit as st
import cv2
import numpy as np

# Import your model processing modules
import model_drowsy
import model_gender
import model_crowd
import model_loi
import model_night
import model_violence3

# Map display names to modules
model_map = {
    "Drowsiness Detection": model_drowsy,
    "Gender Detection": model_gender,
    "Crowd": model_crowd,
    "Loitering": model_loi,
    "Night Anamoly": model_night,
    "Violence Detection": model_violence3,
}

# Streamlit setup
st.set_page_config(page_title="Real-time Detection Dashboard", layout="wide")
st.title("Inspector Clouseau")

# Sidebar: model selection
selected_models = st.sidebar.multiselect(
    "Select models to activate:",
    options=list(model_map.keys()),
    default=["Drowsiness Detection"]
)

# Streamlit camera widget (or default webcam)
FRAME_WINDOW = st.image([], channels="RGB")

# VideoCapture from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Could not open webcam.")
else:
    st.success("✅ Webcam is active.")

# Loop for live feed
run = st.checkbox("Run Detection", value=True)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Could not read frame.")
        break

    # Optional: Flip frame to mirror webcam
    frame = cv2.flip(frame, 1)

    alert = None  # Reset alert flag

    # Apply selected models
    for model_name in selected_models:
        frame, alert = model_map[model_name].process_frame(frame)
        
        if alert:
            st.error(alert)  # Show the alert
    
    result = model_map[model_name].process_frame(frame)
    print(result)  # See what it returns

    # Convert to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

# Release resources
cap.release()
