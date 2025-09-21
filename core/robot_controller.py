# core/robot_controller.py

import time
from config.settings import ALIGNMENT_TOLERANCE, ROTATE_SPEED_P, MOVE_SPEED, DISTANCE_THRESHOLD


class RobotController:
    def __init__(self, chassis, sensor):
        self.chassis = chassis
        self.sensor = sensor

    def align_to_target(self, center_x, frame_width):
        frame_center_x = frame_width / 2
        error_x = center_x - frame_center_x

        if abs(error_x) <= ALIGNMENT_TOLERANCE:
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            return True
        else:
            z_speed = error_x * ROTATE_SPEED_P
            z_speed = max(-30, min(30, z_speed))
            self.chassis.drive_speed(x=0, y=0, z=z_speed, timeout=0.1)
            return False

    def move_forward_until_distance(self, target_distance=DISTANCE_THRESHOLD, timeout=10):
        start_time = time.time()
        while True:
            dist = self.get_distance()
            if dist is not None and dist <= target_distance:
                self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                return True
            elif time.time() - start_time > timeout:
                self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                return False
            time.sleep(0.1)

    def get_distance(self):
        # 这里可以通过回调或其他方式获取 current_distance
        pass  # 留给外部注入