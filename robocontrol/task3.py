import time
import cv2
import numpy as np
from ultralytics import YOLO  # 用于加载和运行YOLOv8模型
import robomaster
from robomaster import robot

# --- 配置 ---
# YOLOv8 模型路径
MODEL_PATH = "YOUR_MODEL_PATH.pt"  # <--- 请修改为你的模型文件路径
CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
ALIGNMENT_TOLERANCE = 20  # 对准容差 (像素)
MOVE_SPEED = 0.1  # 前进速度 (m/s)
ROTATE_SPEED_P = 0.005  # 旋转速度比例系数
DISTANCE_THRESHOLD = 25  # 抓取距离阈值 (cm)
DISTANCE_SENSOR_INDEX = 3  # 测距仪连接的传感器板索引 (根据实际情况调整)

# --- 全局变量 ---
current_distance = None  # 存储最新的测距仪数据


def distance_callback(adapter_info):
    """处理测距仪数据的回调函数"""
    global current_distance
    # 假设测距仪连接在指定索引的传感器板上
    # 注意：adapter_info[INDEX] 可能包含多个值，如 [ir_distance, tof_distance1, tof_distance2, ...]
    # 请根据实际传感器返回的数据结构调整
    try:
        # 示例：假设返回的是 [ir_distance, tof_distance1, tof_distance2, ...]
        # 你需要确认哪个索引对应你的测距仪数据
        # 这里假设是第一个TOF传感器 (索引可能需要调整)
        # 常见结构可能是 [ir, tof1, tof2, tof3, tof4, line_sensor_data]
        # 如果是第四个TOF传感器，可能是 adapter_info[DISTANCE_SENSOR_INDEX][3]
        # *** 请根据实际打印的 adapter_info 内容调整 ***
        # 临时打印以查看结构
        # print(f"Raw adapter info: {adapter_info}")

        # 尝试获取指定索引的数据，可能需要进一步索引
        sensor_data = adapter_info[DISTANCE_SENSOR_INDEX]
        # 如果 sensor_data 本身是列表，可能需要再取一个索引，比如 [0] 或 [3]
        # *** 需要根据你的传感器实际输出调整这一行 ***
        current_distance = sensor_data[0]  # <--- 请根据实际情况调整索引
        print(f"Distance: {current_distance} cm")
    except (IndexError, TypeError) as e:
        print(f"Error reading distance data: {e}")
        # 打印原始数据以便调试
        print(f"Raw adapter info received: {adapter_info}")
        current_distance = None


def main():
    global current_distance

    # 1. 初始化YOLOv8模型
    print("Loading YOLOv8 model...")
    try:
        model = YOLO(MODEL_PATH)
        # 预热模型 (可选，但推荐)
        # dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # model(dummy_img)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. 初始化机器人
    print("Initializing robot...")
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta")  # 或 "ap" 根据你的连接方式
        print("Robot initialized.")
    except Exception as e:
        print(f"Failed to initialize robot: {e}")
        return

    # 3. 获取机器人组件
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gripper = ep_robot.gripper
    ep_sensor = ep_robot.sensor

    # 4. 启动摄像头流和测距仪订阅
    print("Starting camera stream and distance sensor...")
    ep_camera.start_video_stream(display=False)
    ep_sensor.sub_distance(freq=20, callback=distance_callback)  # 提高频率以更快响应

    target_aligned = False
    target_grabbed = False
    # 标志位，用于控制是否退出显示循环
    user_quit = False

    try:
        print("Starting main loop... Press 'q' in the OpenCV window to quit.")
        while not target_grabbed and not user_quit:

            # 获取最新图像帧
            img = ep_camera.read_cv2_image(strategy="newest")
            if img is None:
                print("Failed to get image frame.")
                time.sleep(0.1)
                continue

            # 1. YOLOv8 推理
            results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)  # verbose=False 减少输出

            # 获取带标注的图像帧
            annotated_frame = results[0].plot()  # 绘制检测框、标签等

            # 获取检测结果 (用于逻辑判断)
            if results and len(results) > 0:
                result = results[0]  # 假设每次只处理一张图片
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    # 假设只检测到一个目标，取第一个（置信度最高的）
                    # 如果可能有多个，需要选择最中心或最近的
                    box = boxes.xyxy[0].cpu().numpy()  # 获取第一个框的坐标 [x1, y1, x2, y2]
                    cls = int(boxes.cls[0].cpu().numpy())  # 获取类别索引
                    conf = float(boxes.conf[0].cpu().numpy())  # 获取置信度

                    # 检查是否检测到积木
                    if cls == 0 and conf >= CONFIDENCE_THRESHOLD:  # names: ['block'] -> index 0
                        x1, y1, x2, y2 = box
                        # 计算检测框中心点
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        frame_center_x = img.shape[1] / 2
                        frame_center_y = img.shape[0] / 2

                        print(
                            f"Detected block at ({center_x:.1f}, {center_y:.1f}). Frame center ({frame_center_x:.1f}, {frame_center_y:.1f})")

                        # 2. 旋转车身使积木正对中轴线 (仅在未对准且未抓取时)
                        if not target_aligned and not target_grabbed:
                            error_x = center_x - frame_center_x
                            print(f"Alignment error X: {error_x:.1f}")

                            # 如果中心点在画面中心容差范围内，则认为已对准
                            if abs(error_x) <= ALIGNMENT_TOLERANCE:
                                print("Target aligned.")
                                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)  # 停止旋转
                                target_aligned = True
                                time.sleep(0.5)  # 稳定一下
                                # 在图像上添加对准信息
                                cv2.putText(annotated_frame, "ALIGNED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                            else:
                                # 根据误差比例控制旋转速度 (简化版P控制)
                                # 误差为正，目标在右，需要向左转 (z为正)
                                z_speed = -error_x * ROTATE_SPEED_P
                                # 限制最大旋转速度
                                z_speed = np.clip(z_speed, -30, 30)
                                print(f"Rotating with speed Z: {z_speed:.2f}")
                                ep_chassis.drive_speed(x=0, y=0, z=z_speed, timeout=0.1)
                                # 在图像上添加旋转信息
                                cv2.putText(annotated_frame, f"ALIGNING: Z={z_speed:.2f}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # 3. 对准后执行抓取动作
                    if target_aligned and not target_grabbed:
                        print("Target aligned. Initiating grab sequence...")
                        # 在图像上添加抓取信息
                        cv2.putText(annotated_frame, "GRABBING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        # 打开机械爪
                        print("Opening gripper...")
                        ep_gripper.open()
                        time.sleep(2)  # 等待机械爪完全打开

                        # 向前移动并检查距离
                        print("Moving forward...")
                        ep_chassis.drive_speed(x=MOVE_SPEED, y=0, z=0, timeout=0.1)  # 开始前进

                        # 持续检查距离直到达到阈值或循环超时
                        move_start_time = time.time()
                        while True:
                            # 更新显示的距离
                            if current_distance is not None:
                                dist_text = f"Distance: {current_distance:.1f} cm"
                                print(dist_text)
                                # 在图像上更新距离信息
                                cv2.rectangle(annotated_frame, (0, img.shape[0] - 30), (300, img.shape[0]), (0, 0, 0),
                                              -1)  # 清除旧距离
                                cv2.putText(annotated_frame, dist_text, (10, img.shape[0] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            if current_distance is not None and current_distance <= DISTANCE_THRESHOLD:
                                print(f"Target distance reached: {current_distance} cm. Stopping.")
                                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)  # 停止移动
                                break
                            elif time.time() - move_start_time > 10:  # 超时保护，防止无限移动
                                print("Move timeout. Stopping.")
                                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                                break
                            time.sleep(0.1)  # 短暂等待下一次距离检查

                        # 停止移动后，闭合机械爪
                        print("Closing gripper to grab object...")
                        ep_gripper.close()
                        time.sleep(2)  # 等待机械爪闭合

                        target_grabbed = True
                        print("Object grabbed successfully.")
                        cv2.putText(annotated_frame, "GRABBED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    print("No block detected in current frame.")
                    cv2.putText(annotated_frame, "NO DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # 如果之前已对准但当前帧未检测到，可能需要重新搜索或处理异常
                    # 这里简单处理：重置对准状态以便重新寻找
                    if target_aligned and not target_grabbed:
                        print("Lost target after alignment. Resetting alignment.")
                        target_aligned = False
                        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)  # 确保停止

            else:
                print("No detection results.")
                cv2.putText(annotated_frame, "NO RESULTS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if target_aligned and not target_grabbed:
                    print("Lost target after alignment. Resetting alignment.")
                    target_aligned = False
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)

            # --- 显示带标注的图像 ---
            cv2.imshow("YOLOv8 Detection - Robomaster", annotated_frame)

            # 等待按键，1ms延迟。如果按下 'q' 则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                user_quit = True
                print("User requested quit via 'q' key.")

            time.sleep(0.05)  # 控制主循环频率

    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4. 清理资源
        print("Stopping video stream and unsubscribing...")
        ep_camera.stop_video_stream()
        try:
            ep_sensor.unsub_distance()
        except:
            pass  # 防止 unsubscribe 时出错

        print("Stopping chassis...")
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            time.sleep(1)
        except:
            pass

        print("Closing robot connection...")
        try:
            ep_robot.close()
        except:
            pass
        print("Robot connection closed.")

        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        print("OpenCV windows closed. Cleanup complete.")


if __name__ == '__main__':
    main()