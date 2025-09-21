# models/yolo_model.py
from ultralytics import YOLO
from config.settings import MODEL_PATH, CONFIDENCE_THRESHOLD, DEVICE


class YoloDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.model.to(DEVICE)
        print(f"YOLO Model loaded on {DEVICE}")

    def detect(self, image):
        results = self.model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        return results[0] if results else None
