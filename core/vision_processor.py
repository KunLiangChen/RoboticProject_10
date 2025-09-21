# core/vision_processor.py

import cv2
import numpy as np


class VisionProcessor:
    def __init__(self, detector):
        self.detector = detector

    def process_frame(self, img):
        annotated_frame = img.copy()
        result = self.detector.detect(img)

        if result and result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes.xyxy[0].cpu().numpy()
            cls = int(result.boxes.cls[0].cpu().numpy())
            conf = float(result.boxes.conf[0].cpu().numpy())

            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            return {
                "detected": True,
                "bbox": (x1, y1, x2, y2),  # 边界框坐标
                "center": (center_x, center_y),  # 中心坐标
                "class": cls,  # 标签类型
                "confidence": conf,  # 置信度
                "annotated_frame": result.plot()  # 带标记的图像
            }

        return {"detected": False, "annotated_frame": annotated_frame}