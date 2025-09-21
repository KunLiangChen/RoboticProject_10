# main.py

from robomaster import robot
from config.settings import *
from models.yolo_model import YoloDetector
from core.vision_processor import VisionProcessor
from core.robot_controller import RobotController
from core.gripper_arm import GripperArmController
from core.navigator import MarkerNavigator
import cv2


def main():
    # 初始化机器人所有组件
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gripper = ep_robot.gripper
    ep_arm = ep_robot.robotic_arm
    ep_vision = ep_robot.vision
    ep_sensor = ep_robot.sensor
    # 启动视频流
    ep_camera.start_video_stream(display=False)

    detector = YoloDetector()
    vision = VisionProcessor(detector)
    controller = RobotController(ep_chassis, ep_sensor)
    gripper_arm = GripperArmController(ep_gripper, ep_arm)
    navigator = MarkerNavigator(ep_vision, ep_chassis)

    ep_vision.sub_detect_info(name="marker", callback=navigator.on_marker_detected)

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest")
            if img is None:
                continue

            result = vision.process_frame(img)
            annotated_frame = result["annotated_frame"]

            if result["detected"]:
                aligned = controller.align_to_target(result["center"][0], img.shape[1])
                if aligned:
                    gripper_arm.open_gripper()
                    controller.move_forward_until_distance()
                    gripper_arm.close_gripper()
                    gripper_arm.lift_arm()
                    success = navigator.navigate_to_marker(ep_camera)
                    if success:
                        gripper_arm.lower_arm()
                        gripper_arm.open_gripper()
                    break

            cv2.imshow("Vision", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
