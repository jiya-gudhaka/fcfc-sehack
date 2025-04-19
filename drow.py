from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")  # Replace with your actual model path

# Start webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on frame
    results = model(frame)

    # Plot detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Drowsiness Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
