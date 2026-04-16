from ultralytics import YOLO
import numpy as np
import os

# Load model once (global)
# Use absolute path to ensure it works regardless of where uvicorn is run from
model_path = os.path.join(os.path.dirname(__file__), "..", "weight", "best.pt")
model = YOLO(model_path)

def predict_image(image):
    results = model(image)

    detections = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            detections.append({
                "class": model.names[cls_id],
                "confidence": conf,
                "bbox": coords
            })

    return detections