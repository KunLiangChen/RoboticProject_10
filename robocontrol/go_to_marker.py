# -*-coding:utf-8-*-
import cv2
import time
from robomaster import robot
from robomaster import vision

# 自定义 Marker 类
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * 1280), int((self._y - self._h / 2) * 720)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * 1280), int((self._y + self._h / 2) * 720)

    @property
    def center(self):
        return int(self._x * 1280), int(self._y * 720)

    @property
    def text(self):
        return self._info

    @property
    def width(self):
        return self._w

    @property
    def offset_x(self):  # 水平偏移 [-1, 1]
        return self._x - 0.5


markers = []

# 回调函数：检测 Marker
def on_detect_marker(marker_info):
    markers.clear()
    for info in marker_info:
        x, y, w, h, data = info
        markers.append(MarkerInfo(x, y, w, h, data))

# 寻找特定 Marker（例如 "heart"）
def find_target_marker(target_name="heart"):
    for marker in markers:
        if marker.text == target_name:
            return marker
    return None

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis

    ep_camera.start_video_stream(display=False)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    target_marker_name = "heart"  # 指定要识别的 Marker 名称
    arrived = False

    try:
        while not arrived:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            target = find_target_marker(target_marker_name)

            if target:
                print(f"检测到目标 Marker: {target.text}, 宽度: {target.width:.2f}, 偏移: {target.offset_x:.2f}")

                # 绘制框和文字
                cv2.rectangle(img, target.pt1, target.pt2, (0, 255, 0), 2)
                cv2.putText(img, target.text, target.center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 判断是否到达合适距离（假设 w > 0.3 表示距离合适）
                if target.width > 0.20:
                    print("已到达合适距离，停止移动。")
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    arrived = True
                else:
                    # 控制前进速度
                    forward_speed = (0.3 - target.width) * 1.0  # 简单比例控制
                    forward_speed = max(0.1, min(0.5, forward_speed))  # 限制速度范围

                    # 控制转向（z轴旋转）
                    turn_speed = target.offset_x * 60  # 调整转向灵敏度
                    turn_speed = max(-60, min(60, turn_speed))

                    print(f"前进速度: {forward_speed:.2f}, 转向速度: {turn_speed:.2f}")
                    ep_chassis.drive_speed(x=forward_speed, y=0, z=turn_speed, timeout=0.1)
            else:
                # 没有检测到目标，原地旋转寻找
                print("未检测到目标 Marker，原地旋转寻找...")
                ep_chassis.drive_speed(x=0, y=0, z=30, timeout=0.1)

            cv2.imshow("Markers", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    finally:
        ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
        ep_robot.close()
        cv2.destroyAllWindows()