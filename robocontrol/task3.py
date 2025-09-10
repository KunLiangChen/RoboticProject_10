import time
import cv2
import numpy as np
from ultralytics import YOLO  # 用于加载和运行YOLOv8 模型
import robomaster
from robomaster import robot
from robomaster import vision

# --- 配置 ---
# YOLOv8 模型路径
MODEL_PATH = "./model/best.pt"  # <--- 请修改为你的模型文件路径
CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
ALIGNMENT_TOLERANCE = 50  # 对准容差 (像素)
MOVE_SPEED = 0.25  # 前进速度 (m/s)
ROTATE_SPEED_P = 0.05  # 旋转速度比例系数
DISTANCE_THRESHOLD = 270  # 抓取距离阈值 (cm)
DISTANCE_SENSOR_INDEX = 3  # 测距仪连接的传感器板索引 (根据实际情况调整)

# Marker相关配置
TARGET_MARKER_NAME = "heart"  # 指定要识别的 Marker 名称
MARKER_DISTANCE_THRESHOLD = 0.20  # Marker距离阈值

# --- 全局变量 ---
current_distance = None  # 存储最新的测距仪数据
markers = []  # 存储检测到的marker信息


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


def distance_callback(adapter_info):
    """处理测距仪数据的回调函数"""
    global current_distance
    try:
        sensor_data = adapter_info[DISTANCE_SENSOR_INDEX]
        current_distance = sensor_data
        print(f"Distance: {current_distance} cm")
    except (IndexError, TypeError) as e:
        print(f"Error reading distance data: {e}")
        print(f"Raw adapter info received: {adapter_info}")
        current_distance = None


def on_detect_marker(marker_info):
    """处理marker检测的回调函数"""
    global markers
    markers.clear()
    for info in marker_info:
        x, y, w, h, data = info
        markers.append(MarkerInfo(x, y, w, h, data))


def find_target_marker(target_name=TARGET_MARKER_NAME):
    """寻找特定名称的marker"""
    for marker in markers:
        if marker.text == target_name:
            return marker
    return None


def move_to_marker(ep_chassis, ep_camera, ep_vision):
    """移动到目标marker附近"""
    print(f"开始寻找目标 Marker: {TARGET_MARKER_NAME}")

    # 订阅marker检测
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    arrived = False
    user_quit = False
    start_time = time.time()
    timeout = 30  # 30秒超时

    try:
        while not arrived and not user_quit:
            # 检查超时
            if time.time() - start_time > timeout:
                print("寻找marker超时")
                break

            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is None:
                print("Failed to get image frame for marker detection.")
                time.sleep(0.1)
                continue

            target = find_target_marker(TARGET_MARKER_NAME)

            if target:
                print(f"检测到目标 Marker: {target.text}, 宽度: {target.width:.2f}, 偏移: {target.offset_x:.2f}")

                # 绘制框和文字
                cv2.rectangle(img, target.pt1, target.pt2, (0, 255, 0), 2)
                cv2.putText(img, target.text, target.center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 判断是否到达合适距离
                if target.width > MARKER_DISTANCE_THRESHOLD:
                    print("已到达合适距离，停止移动。")
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    arrived = True
                else:
                    # 控制前进速度
                    forward_speed = (MARKER_DISTANCE_THRESHOLD - target.width) * 1.0
                    forward_speed = max(0.1, min(0.5, forward_speed))

                    # 控制转向（z轴旋转）
                    turn_speed = target.offset_x * 60
                    turn_speed = max(-60, min(60, turn_speed))

                    print(f"前进速度: {forward_speed:.2f}, 转向速度: {turn_speed:.2f}")
                    ep_chassis.drive_speed(x=forward_speed, y=0, z=turn_speed, timeout=0.1)
            else:
                # 没有检测到目标，原地旋转寻找
                print("未检测到目标 Marker，原地旋转寻找...")
                ep_chassis.drive_speed(x=0, y=0, z=30, timeout=0.1)

            # 显示图像（可选）
            try:
                cv2.imshow("Markers", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    user_quit = True
                    print("User requested quit via 'q' key during marker navigation.")
            except:
                pass  # 如果无法显示图像，继续执行

            time.sleep(0.05)

    except Exception as e:
        print(f"Error during marker navigation: {e}")
    finally:
        # 取消订阅marker检测
        try:
            ep_vision.unsub_detect_info(name="marker")
            print("已取消订阅marker检测")
        except Exception as e:
            print(f"取消订阅marker检测时出错: {e}")

    return arrived and not user_quit


def move_arm_up(ep_arm):
    """抬起机械臂"""
    print("开始抬起机械臂...")
    try:
        ep_arm.move(x=40).wait_for_completed()
        print("X轴移动完成")
        ep_arm.move(y=90).wait_for_completed()
        print("Y轴移动完成")
        print("机械臂抬起完成")
        return True
    except Exception as e:
        print(f"抬起机械臂时出错: {e}")
        return False


def move_arm_down(ep_arm):
    """放下机械臂"""
    print("开始放下机械臂...")
    try:
        ep_arm.move(y=-90).wait_for_completed()
        print("Y轴移动完成")
        ep_arm.move(x=-40).wait_for_completed()
        print("X轴移动完成")
        print("机械臂放下完成")
        return True
    except Exception as e:
        print(f"放下机械臂时出错: {e}")
        return False


def open_gripper(ep_gripper):
    """打开机械爪"""
    print("开始松开机械爪...")
    try:
        ep_gripper.open()
        time.sleep(2)
        print("机械爪已松开")
        return True
    except Exception as e:
        print(f"松开机械爪时出错: {e}")
        return False


def close_gripper(ep_gripper):
    """关闭机械爪"""
    print("开始闭合机械爪...")
    try:
        ep_gripper.close()
        time.sleep(2)
        print("机械爪已闭合")
        return True
    except Exception as e:
        print(f"闭合机械爪时出错: {e}")
        return False


def main():
    global current_distance

    # 1. 初始化YOLOv8模型
    print("Loading YOLOv8 model...")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. 初始化机器人
    print("Initializing robot...")
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="ap")
        print("Robot initialized.")
    except Exception as e:
        print(f"Failed to initialize robot: {e}")
        return

    # 3. 获取机器人组件
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gripper = ep_robot.gripper
    ep_sensor = ep_robot.sensor
    ep_arm = ep_robot.robotic_arm

    # 4. 启动摄像头流和测距仪订阅
    print("Starting camera stream and distance sensor...")
    ep_camera.start_video_stream(display=False)
    ep_sensor.sub_distance(freq=5, callback=distance_callback)  # 降低频率避免过多打印

    target_aligned = False
    target_grabbed = False
    task_completed = False
    user_quit = False

    try:
        print("Starting main loop... Press 'q' in the OpenCV window to quit.")
        while not task_completed and not user_quit:

            # 获取最新图像帧
            img = ep_camera.read_cv2_image(strategy="newest")
            if img is None:
                print("Failed to get image frame.")
                time.sleep(0.1)
                continue

            # 1. YOLOv8 推理
            results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # 获取带标注的图像帧
            annotated_frame = results[0].plot()

            # 获取检测结果
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    box = boxes.xyxy[0].cpu().numpy()
                    cls = int(boxes.cls[0].cpu().numpy())
                    conf = float(boxes.conf[0].cpu().numpy())

                    # 检查是否检测到积木
                    if (cls == 0 or cls == 1) and conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        frame_center_x = img.shape[1] / 2
                        frame_center_y = img.shape[0] / 2

                        print(f"Detected block at ({center_x:.1f}, {center_y:.1f})")

                        # 2. 旋转车身使积木正对中轴线
                        if not target_aligned and not target_grabbed:
                            error_x = center_x - frame_center_x
                            print(f"Alignment error X: {error_x:.1f}")

                            if abs(error_x) <= ALIGNMENT_TOLERANCE:
                                print("Target aligned.")
                                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                                target_aligned = True
                                time.sleep(0.5)
                                cv2.putText(annotated_frame, "ALIGNED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                            else:
                                z_speed = error_x * ROTATE_SPEED_P
                                z_speed = np.clip(z_speed, -30, 30)
                                print(f"Rotating with speed Z: {z_speed:.2f}")
                                ep_chassis.drive_speed(x=0, y=0, z=z_speed, timeout=0.1)
                                cv2.putText(annotated_frame, f"ALIGNING: Z={z_speed:.2f}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # 3. 对准后执行抓取动作
                    if target_aligned and not target_grabbed:
                        print("Target aligned. Initiating grab sequence...")
                        cv2.putText(annotated_frame, "GRABBING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        # 1. 打开机械爪
                        print("Opening gripper...")
                        if not open_gripper(ep_gripper):
                            print("打开机械爪失败")
                            break
                        time.sleep(1)

                        # 2. 向前移动并检查距离
                        print("Moving forward towards target...")
                        ep_chassis.drive_speed(x=MOVE_SPEED, y=0, z=0, timeout=0)

                        # 持续检查距离直到达到阈值或超时
                        move_start_time = time.time()
                        while True:
                            if current_distance is not None:
                                dist_text = f"Distance: {current_distance:.1f} cm"
                                print(dist_text)
                                cv2.rectangle(annotated_frame, (0, img.shape[0] - 30), (300, img.shape[0]), (0, 0, 0),
                                              -1)
                                cv2.putText(annotated_frame, dist_text, (10, img.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            if current_distance is not None and current_distance <= DISTANCE_THRESHOLD:
                                print(
                                    f"Target distance threshold ({DISTANCE_THRESHOLD} cm) reached: {current_distance} cm")
                                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                                print("Robot stopped at threshold.")
                                break
                            elif time.time() - move_start_time > 10:
                                print("Move timeout. Stopping.")
                                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                                break
                            time.sleep(0.1)

                        # 3. 执行"向前冲"动作
                        if current_distance is not None and current_distance <= DISTANCE_THRESHOLD:
                            CHARGE_DURATION = 0.7
                            CHARGE_SPEED = MOVE_SPEED
                            print(f"Executing forward charge for {CHARGE_DURATION} seconds...")
                            ep_chassis.drive_speed(x=CHARGE_SPEED, y=0, z=0, timeout=5)
                            time.sleep(CHARGE_DURATION)
                            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                            print("Charge completed. Stopping robot.")
                            time.sleep(0.5)

                        # 4. 闭合机械爪进行抓取
                        print("Closing gripper to grab object...")
                        if not close_gripper(ep_gripper):
                            print("闭合机械爪失败")
                            break

                        # 5. 标记抓取完成
                        target_grabbed = True
                        print("Object grab sequence finished.")
                        cv2.putText(annotated_frame, "GRABBED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # === 新增逻辑：抓取后执行后续动作 ===
                        if target_grabbed:
                            print("=" * 50)
                            print("开始执行放置任务...")

                            # 1. 抬起机械臂
                            if move_arm_up(ep_arm):
                                print("机械臂抬起成功")

                                # 2. 寻找marker并移动到附近
                                print("开始寻找目标marker并移动...")
                                success = move_to_marker(ep_chassis, ep_camera, ep_robot.vision)

                                if success:
                                    print("已到达目标marker附近")
                                    ep_chassis.drive_speed(x=0,y=0,z=0,timeout=0.1)
                                    time.sleep(2)
                                    ep_sensor.unsub_distance()
                                    time.sleep(1)
                                    # 3. 放下机械臂
                                    if move_arm_down(ep_arm):
                                        print("机械臂放下成功")

                                        # 4. 松开机械爪
                                        if open_gripper(ep_gripper):
                                            print("机械爪已松开，任务完成")
                                            task_completed = True
                                        else:
                                            print("松开机械爪失败")
                                    else:
                                        print("放下机械臂失败")
                                    ep_sensor.sub_distance(freq=5, callback=distance_callback)
                                else:
                                    print("未能成功到达目标marker")
                            else:
                                print("抬起机械臂失败")
                            print("=" * 50)

                else:
                    print("No block detected in current frame.")
                    cv2.putText(annotated_frame, "NO DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if target_aligned and not target_grabbed:
                        print("Lost target after alignment. Resetting alignment.")
                        target_aligned = False
                        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)

            else:
                print("No detection results.")
                cv2.putText(annotated_frame, "NO RESULTS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if target_aligned and not target_grabbed:
                    print("Lost target after alignment. Resetting alignment.")
                    target_aligned = False
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)

            # 显示带标注的图像
            try:
                cv2.imshow("YOLOv8 Detection - Robomaster", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    user_quit = True
                    print("User requested quit via 'q' key.")
            except:
                pass

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("开始清理资源...")

        # 停止所有运动
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            print("底盘已停止")
        except:
            pass

        # 停止视频流和传感器订阅
        try:
            ep_camera.stop_video_stream()
            print("视频流已停止")
        except:
            pass

        try:
            ep_sensor.unsub_distance()
            print("距离传感器订阅已取消")
        except:
            pass

        # 关闭机器人连接
        try:
            ep_robot.close()
            print("机器人连接已关闭")
        except:
            pass

        # 关闭OpenCV窗口
        try:
            cv2.destroyAllWindows()
            print("OpenCV窗口已关闭")
        except:
            pass

        print("Cleanup complete.")


if __name__ == '__main__':
    main()
