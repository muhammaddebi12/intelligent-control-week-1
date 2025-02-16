import torch
import cv2
import numpy as np
from ultralytics import YOLO


def load_model():
    """Load YOLOv5 model"""
    model = YOLO("yolov5s.pt")  # Use the small YOLOv5 model (s for small)
    return model


def detect_objects(model, frame):
    """Detect objects in the video frame"""
    results = model(frame)
    return results


def draw_boxes(frame, results):
    """Draw bounding boxes around detected objects"""
    h, w, _ = frame.shape

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            label = f"{result.names[class_id]}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def main():
    model = load_model()
    cap = cv2.VideoCapture(0)  # Use default webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(model, frame)
        frame = draw_boxes(frame, results)

        cv2.imshow("Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
