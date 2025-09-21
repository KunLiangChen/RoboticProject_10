# core/navigator.py

import time
from config.settings import TARGET_MARKER_NAME, MARKER_DISTANCE_THRESHOLD


class MarkerNavigator:
    def __init__(self, vision_module, chassis):
        self.vision = vision_module
        self.chassis = chassis
        self.markers = []

    def on_marker_detected(self, marker_info):
        self.markers.clear()
        for info in marker_info:
            x, y, w, h, data = info
            self.markers.append({"x": x, "y": y, "w": w, "h": h, "text": data})

    def find_target_marker(self, name=TARGET_MARKER_NAME):
        for marker in self.markers:
            if marker["text"] == name:
                return marker
        return None

    def navigate_to_marker(self, camera, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            img = camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                continue

            target = self.find_target_marker()

            if target:
                width = target["w"]
                offset_x = target["x"] - 0.5

                if width > MARKER_DISTANCE_THRESHOLD:
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    return True
                else:
                    forward_speed = (MARKER_DISTANCE_THRESHOLD - width) * 1.0
                    forward_speed = max(0.1, min(0.5, forward_speed))
                    turn_speed = offset_x * 60
                    turn_speed = max(-60, min(60, turn_speed))
                    self.chassis.drive_speed(x=forward_speed, y=0, z=turn_speed, timeout=0.1)
            else:
                self.chassis.drive_speed(x=0, y=0, z=30, timeout=0.1)

            time.sleep(0.05)
        return False