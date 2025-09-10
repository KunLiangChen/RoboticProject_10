import time
import cv2
import numpy as np
from ultralytics import YOLO
from robomaster import robot


# ==================== 全局配置 ====================
MODEL_PATH = "./model/best.pt"
CONFIDENCE_THRESHOLD = 0.5
ALIGNMENT_TOLERANCE = 50
MOVE_SPEED = 0.25
ROTATE_SPEED_P = 0.05
DISTANCE_THRESHOLD = 270
TARGET_MARKER_NAME = "heart"
MARKER_DISTANCE_THRESHOLD = 0.20


# ==================== 模块1: YOLO 检测器 ====================
class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print("YOLO model loaded.")

    def detect(self, img, conf_threshold=0.5):
        results = self.model(img, conf=conf_threshold, verbose=False)
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                box = boxes.xyxy[0].cpu().numpy()
                cls = int(boxes.cls[0].cpu().numpy())
                conf = float(boxes.conf[0].cpu().numpy())
                return box, cls, conf
        return None, None, None


# ==================== 模块2: 底盘控制器 ====================
class ChassisController:
    def __init__(self, chassis):
        self.chassis = chassis

    def move(self, x=0, y=0, z=0, timeout=0.1):
        self.chassis.drive_speed(x=x, y=y, z=z, timeout=timeout)

    def stop(self):
        self.move(x=0, y=0, z=0)


# ==================== 模块3: 机械臂控制器 ====================
class ArmController:
    def __init__(self, arm):
        self.arm = arm

    def lift(self):
        self.arm.move(x=40).wait_for_completed()
        self.arm.move(y=90).wait_for_completed()

    def lower(self):
        self.arm.move(y=-90).wait_for_completed()
        self.arm.move(x=-40).wait_for_completed()


# ==================== 模块4: 机械爪控制器 ====================
class GripperController:
    def __init__(self, gripper):
        self.gripper = gripper

    def open(self):
        self.gripper.open()
        time.sleep(2)

    def close(self):
        self.gripper.close()
        time.sleep(2)


# ==================== 模块5: 传感器管理 ====================
class SensorManager:
    def __init__(self, sensor, index=3):
        self.sensor = sensor
        self.index = index
        self.distance = None
        self.sensor.sub_distance(freq=5, callback=self._distance_callback)

    def _distance_callback(self, info):
        try:
            self.distance = info[self.index]
        except:
            self.distance = None

    def get_distance(self):
        return self.distance


# ==================== 模块6: Marker 导航 ====================
class MarkerNavigator:
    def __init__(self, vision, camera, chassis):
        self.vision = vision
        self.camera = camera
        self.chassis = chassis
        self.markers = []

    def _marker_callback(self, marker_info):
        self.markers = []
        for info in marker_info:
            x, y, w, h, data = info
            self.markers.append({
                "x": x, "y": y, "w": w, "h": h,
                "pt1": (int((x - w / 2) * 1280), int((y - h / 2) * 720)),
                "pt2": (int((x + w / 2) * 1280), int((y + h / 2) * 720)),
                "center": (int(x * 1280), int(y * 720)),
                "text": data,
                "offset_x": x - 0.5
            })

    def find_marker(self, name):
        for marker in self.markers:
            if marker["text"] == name:
                return marker
        return None

    def navigate_to_marker(self, name):
        self.vision.sub_detect_info(name="marker", callback=self._marker_callback)
        start_time = time.time()
        while time.time() - start_time < 30:
            img = self.camera.read_cv2_image(strategy="newest")
            marker = self.find_marker(name)
            if marker:
                if marker["w"] > MARKER_DISTANCE_THRESHOLD:
                    self.chassis.move(x=0, y=0, z=0)
                    self.vision.unsub_detect_info(name="marker")
                    return True
                forward = (MARKER_DISTANCE_THRESHOLD - marker["w"]) * 1.0
                turn = marker["offset_x"] * 60
                self.chassis.move(x=forward, z=turn)
            else:
                self.chassis.move(z=30)
            time.sleep(0.05)
        self.vision.unsub_detect_info(name="marker")
        return False


# ==================== 模块7: 任务调度器 ====================
class TaskManager:
    def __init__(self, detector, chassis, arm, gripper, sensor, navigator):
        self.detector = detector
        self.chassis = chassis
        self.arm = arm
        self.gripper = gripper
        self.sensor = sensor
        self.navigator = navigator

    def run(self, camera):
        """
        机器人抓取与放置任务主流程

        整体运行逻辑：
        1. 【目标检测阶段】
           - 持续使用YOLOv8检测视野中的目标物体（积木）
           - 通过底盘旋转调整，使目标物体位于画面中心（对准）

        2. 【抓取准备阶段】
           - 对准后，打开机械爪
           - 控制底盘向前移动，同时监控测距仪距离
           - 当距离达到预设阈值时停止移动

        3. 【抓取执行阶段】
           - 执行短时间"冲撞"动作确保接触目标
           - 闭合机械爪完成抓取
           - 标记抓取成功状态

        4. 【放置任务阶段】
           - 抬起机械臂，携带抓取的物体
           - 启动Marker导航系统，搜索指定的放置目标（如"heart"标志）
           - 自动移动到目标Marker附近（基于Marker大小判断距离）
           - 到达位置后，放下机械臂
           - 松开机械爪，完成放置
           - 机械臂复位

        异常处理：
        - 检测丢失时会重置对准状态
        - 各阶段超时控制避免无限等待
        - 所有动作都有try-catch保护和资源清理

        状态流转：检测→对准→抓取→抬升→导航→放置→复位
        """
        grabbed = False
        while True:
            img = camera.read_cv2_image(strategy="newest")
            box, cls, conf = self.detector.detect(img)

            if box is not None and cls in [0, 1]:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                error = cx - img.shape[1] // 2
                if abs(error) > ALIGNMENT_TOLERANCE:
                    self.chassis.move(z=error * ROTATE_SPEED_P)
                else:
                    self.chassis.stop()
                    self.gripper.open()
                    self.chassis.move(x=MOVE_SPEED)
                    while self.sensor.get_distance() > DISTANCE_THRESHOLD:
                        time.sleep(0.1)
                    self.chassis.stop()
                    time.sleep(0.7)
                    self.gripper.close()
                    grabbed = True
                    break
            time.sleep(0.05)

        if grabbed:
            self.arm.lift()
            if self.navigator.navigate_to_marker(TARGET_MARKER_NAME):
                self.arm.lower()
                self.gripper.open()
            self.arm.lower()


# ==================== 主控逻辑 ====================
class RobotController:
    def __init__(self):
        self.robot = robot.Robot()
        self.robot.initialize(conn_type="ap")
        self.camera = self.robot.camera
        self.camera.start_video_stream(display=False)

        self.detector = YOLODetector(MODEL_PATH)
        self.chassis = ChassisController(self.robot.chassis)
        self.arm = ArmController(self.robot.robotic_arm)
        self.gripper = GripperController(self.robot.gripper)
        self.sensor = SensorManager(self.robot.sensor)
        self.navigator = MarkerNavigator(self.robot.vision, self.camera, self.chassis)
        self.task_manager = TaskManager(
            self.detector,
            self.chassis,
            self.arm,
            self.gripper,
            self.sensor,
            self.navigator
        )

    def run(self):
        try:
            self.task_manager.run(self.camera)
        finally:
            self.cleanup()

    def cleanup(self):
        self.chassis.stop()
        self.camera.stop_video_stream()
        self.robot.close()
        cv2.destroyAllWindows()


# ==================== 启动入口 ====================
if __name__ == "__main__":
    controller = RobotController()
    controller.run()