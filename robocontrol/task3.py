import time
import cv2
import numpy as np
from ultralytics import YOLO  # 用于加载和运行YOLOv8 模型
import robomaster
from robomaster import robot
from robomaster import vision
import torch

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

# 搜索相关配置
SEARCH_ROTATION_SPEED = 15  # 搜索旋转速度
SEARCH_TIMEOUT = 30  # 搜索超时时间(秒)
ALIGNMENT_STABILITY_FRAMES = 5  # 对齐稳定帧数（增加稳定性）
MISSING_DETECTION_THRESHOLD = 10  # 丢失检测阈值
GRAB_CONFIRMATION_TIME = 1  # 抓取确认时间
BACKWARD_DISTANCE = 0.3  # 后退距离(米)

# 多目标处理配置
TARGET_SELECTION_STRATEGY = "closest"  # 目标选择策略: "closest", "largest", "center"
STICKY_TARGET_FRAMES = 10  # 粘性目标帧数，保持跟踪同一目标

# 性能优化配置
PROCESS_EVERY_N_FRAMES = 3  # 每N帧处理一次推理
YOLO_IMG_SIZE = 640  # YOLO输入图像尺寸

# --- 全局变量 ---
current_distance = None  # 存储最新的测距仪数据
markers = []  # 存储检测到的marker信息
detection_lost_count = 0  # 检测丢失计数
alignment_stable_count = 0  # 对齐稳定计数
is_object_grabbed = False  # 是否已抓取物体
frame_count = 0  # 帧计数器

# 多目标跟踪变量
current_target_id = None  # 当前跟踪的目标ID
target_sticky_count = 0  # 目标粘性计数
last_target_info = None  # 上一帧目标信息


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


# 自定义积木检测类
class BlockInfo:
    def __init__(self, x1, y1, x2, y2, cls, conf, frame_width, frame_height):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls = cls
        self.conf = conf
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height
        self.frame_center_x = frame_width / 2
        self.frame_center_y = frame_height / 2
        self.distance_to_center = np.sqrt((self.center_x - self.frame_center_x) ** 2 +
                                          (self.center_y - self.frame_center_y) ** 2)

    def __str__(self):
        return f"Block(center:({self.center_x:.1f},{self.center_y:.1f}), area:{self.area:.1f}, dist:{self.distance_to_center:.1f})"


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


def reset_state():
    """重置状态变量"""
    global detection_lost_count, alignment_stable_count, is_object_grabbed, frame_count, markers
    global current_target_id, target_sticky_count, last_target_info
    detection_lost_count = 0
    alignment_stable_count = 0
    is_object_grabbed = False
    frame_count = 0
    markers.clear()  # 清空marker列表
    current_target_id = None
    target_sticky_count = 0
    last_target_info = None
    print("状态已重置")


def should_process_frame():
    """判断是否应该处理当前帧"""
    global frame_count
    frame_count += 1
    return frame_count % PROCESS_EVERY_N_FRAMES == 0


def select_best_block(blocks, frame_width, frame_height):
    """选择最佳的积木目标"""
    global current_target_id, target_sticky_count, last_target_info

    if not blocks:
        return None

    # 如果已经有粘性目标且未超时，优先保持跟踪
    if last_target_info and target_sticky_count < STICKY_TARGET_FRAMES:
        # 寻找与上一帧最接近的目标
        min_distance = float('inf')
        best_block = None

        for block in blocks:
            distance = np.sqrt((block.center_x - last_target_info.center_x) ** 2 +
                               (block.center_y - last_target_info.center_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_block = block

        # 如果找到了足够接近的目标，继续跟踪
        if min_distance < 100:  # 像素距离阈值
            target_sticky_count += 1
            last_target_info = best_block
            print(f"保持跟踪目标，粘性计数: {target_sticky_count}")
            return best_block

    # 根据策略选择新目标
    if TARGET_SELECTION_STRATEGY == "closest":
        # 选择距离画面中心最近的目标
        best_block = min(blocks, key=lambda b: b.distance_to_center)
    elif TARGET_SELECTION_STRATEGY == "largest":
        # 选择面积最大的目标
        best_block = max(blocks, key=lambda b: b.area)
    elif TARGET_SELECTION_STRATEGY == "center":
        # 选择最靠近画面中心的目标（考虑画面尺寸）
        center_x, center_y = frame_width / 2, frame_height / 2
        best_block = min(blocks, key=lambda b: abs(b.center_x - center_x) + abs(b.center_y - center_y))
    else:
        best_block = blocks[0]

    # 更新跟踪状态
    current_target_id = id(best_block)
    target_sticky_count = 1
    last_target_info = best_block
    print(f"选择新目标: {best_block}")
    return best_block


def search_for_block(ep_chassis, ep_camera, model, device):
    """搜索积木模块"""
    print("开始搜索积木...")
    search_start_time = time.time()
    rotation_direction = 1  # 1为顺时针，-1为逆时针
    consecutive_miss_count = 0  # 连续未检测到计数
    global current_target_id, target_sticky_count, last_target_info

    # 重置目标跟踪状态
    current_target_id = None
    target_sticky_count = 0
    last_target_info = None

    while time.time() - search_start_time < SEARCH_TIMEOUT:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if img is None:
            print("Failed to get image frame for search.")
            time.sleep(0.1)
            continue

        # 控制推理频率
        if should_process_frame():
            # YOLO检测
            try:
                results = model(img, conf=CONFIDENCE_THRESHOLD, device=device, imgsz=YOLO_IMG_SIZE, verbose=False)
            except Exception as e:
                print(f"YOLO检测错误: {e}")
                time.sleep(0.1)
                continue

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    # 提取所有检测到的积木
                    blocks = []
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())

                        # 检查是否检测到积木
                        if (cls == 0 or cls == 1) and conf >= CONFIDENCE_THRESHOLD:
                            block = BlockInfo(box[0], box[1], box[2], box[3], cls, conf, img.shape[1], img.shape[0])
                            blocks.append(block)

                    if blocks:
                        # 选择最佳目标
                        best_block = select_best_block(blocks, img.shape[1], img.shape[0])
                        if best_block:
                            print(f"检测到积木: {best_block}，停止搜索")
                            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                            consecutive_miss_count = 0  # 重置丢失计数
                            return True, [best_block.x1, best_block.y1, best_block.x2, best_block.y2]
                        else:
                            consecutive_miss_count += 1
                    else:
                        consecutive_miss_count += 1
                else:
                    consecutive_miss_count += 1
            else:
                consecutive_miss_count += 1

            # 如果连续多次未检测到，继续搜索
            if consecutive_miss_count > MISSING_DETECTION_THRESHOLD:
                consecutive_miss_count = 0  # 重置计数

        # 继续旋转搜索
        ep_chassis.drive_speed(x=0, y=0, z=SEARCH_ROTATION_SPEED * rotation_direction, timeout=0.1)
        time.sleep(0.1)

    # 超时停止
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
    print("搜索超时，未找到积木")
    return False, None


def align_to_block(ep_chassis, center_x, frame_center_x):
    """对齐积木"""
    global alignment_stable_count

    error_x = center_x - frame_center_x
    print(f"对齐误差 X: {error_x:.1f}")

    if abs(error_x) <= ALIGNMENT_TOLERANCE:
        alignment_stable_count += 1
        if alignment_stable_count >= ALIGNMENT_STABILITY_FRAMES:
            print("积木对齐完成")
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            alignment_stable_count = 0  # 重置计数
            return True
        else:
            print(f"对齐稳定计数: {alignment_stable_count}/{ALIGNMENT_STABILITY_FRAMES}")
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
    else:
        alignment_stable_count = 0  # 重置计数
        z_speed = error_x * ROTATE_SPEED_P
        z_speed = np.clip(z_speed, -30, 30)
        print(f"旋转速度 Z: {z_speed:.2f}")
        ep_chassis.drive_speed(x=0, y=0, z=z_speed, timeout=0.1)

    return False


def move_to_marker(ep_chassis, ep_camera, ep_vision, model, device):
    """移动到目标marker附近"""
    print(f"开始寻找目标 Marker: {TARGET_MARKER_NAME}")

    # 订阅marker检测
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    # 清空之前的marker数据
    global markers
    markers.clear()

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
                ep_chassis.drive_speed(x=0, y=0, z=15, timeout=0.1)

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
            # 清空marker数据
            markers.clear()
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


def backward_robot(ep_chassis, distance=BACKWARD_DISTANCE):
    """后退指定距离"""
    print(f"开始后退 {distance} 米...")
    try:
        ep_chassis.move(x=-distance, y=0, z=0, xy_speed=0.3).wait_for_completed()
        print("后退完成")
        return True
    except Exception as e:
        print(f"后退时出错: {e}")
        return False


def execute_single_task_cycle(ep_robot, model, device):
    """执行单次任务循环"""
    global current_distance, detection_lost_count, alignment_stable_count, is_object_grabbed, frame_count
    global current_target_id, target_sticky_count, last_target_info

    # 获取机器人组件
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gripper = ep_robot.gripper
    ep_sensor = ep_robot.sensor
    ep_arm = ep_robot.robotic_arm

    # 重置状态
    reset_state()

    # 2. 搜索积木
    print("=== 阶段2: 搜索积木 ===")
    block_found, box = search_for_block(ep_chassis, ep_camera, model, device)
    if not block_found:
        print("搜索积木失败")
        return False

    # 3. 对齐积木
    print("=== 阶段3: 对齐积木 ===")
    alignment_timeout = time.time() + 60  # 60秒对齐超时（增加时间）
    alignment_completed = False

    while not alignment_completed and time.time() < alignment_timeout:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if img is None:
            print("获取图像失败")
            time.sleep(0.1)
            continue

        # 控制推理频率
        if should_process_frame():
            try:
                results = model(img, conf=CONFIDENCE_THRESHOLD, device=device, imgsz=YOLO_IMG_SIZE, verbose=False)
            except Exception as e:
                print(f"YOLO检测错误: {e}")
                time.sleep(0.1)
                continue

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    # 提取所有检测到的积木
                    blocks = []
                    for i in range(len(boxes)):
                        box_coords = boxes.xyxy[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())

                        # 检查是否检测到积木
                        if (cls == 0 or cls == 1) and conf >= CONFIDENCE_THRESHOLD:
                            block = BlockInfo(box_coords[0], box_coords[1], box_coords[2], box_coords[3],
                                              cls, conf, img.shape[1], img.shape[0])
                            blocks.append(block)

                    if blocks:
                        # 选择最佳目标（保持跟踪）
                        best_block = select_best_block(blocks, img.shape[1], img.shape[0])
                        if best_block:
                            alignment_completed = align_to_block(ep_chassis, best_block.center_x,
                                                                 best_block.frame_center_x)
                            detection_lost_count = 0  # 重置丢失计数
                        else:
                            detection_lost_count += 1
                    else:
                        detection_lost_count += 1
                else:
                    detection_lost_count += 1
            else:
                detection_lost_count += 1

            # 如果连续丢失检测超过阈值，重新搜索
            if detection_lost_count > MISSING_DETECTION_THRESHOLD:
                print("积木丢失，重新搜索")
                block_found, box = search_for_block(ep_chassis, ep_camera, model, device)
                if not block_found:
                    print("重新搜索失败")
                    return False
                detection_lost_count = 0  # 重置计数

        time.sleep(0.05)

    if not alignment_completed:
        print("对齐超时")
        return False

    # 4. 前进并抓取
    print("=== 阶段4: 前进并抓取 ===")

    # 打开机械爪
    if not open_gripper(ep_gripper):
        print("打开机械爪失败")
        return False

    # 向前移动
    print("向前移动...")
    ep_chassis.drive_speed(x=MOVE_SPEED, y=0, z=0, timeout=0)

    move_start_time = time.time()
    while True:
        if current_distance is not None and current_distance <= DISTANCE_THRESHOLD:
            print(f"到达抓取距离: {current_distance} cm")
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            break
        elif time.time() - move_start_time > 10:
            print("移动超时")
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            break
        time.sleep(0.1)

    # 执行"向前冲"动作
    CHARGE_DURATION = 0.7
    CHARGE_SPEED = MOVE_SPEED
    print(f"执行前冲动作 {CHARGE_DURATION} 秒...")
    ep_chassis.drive_speed(x=CHARGE_SPEED, y=0, z=0, timeout=5)
    time.sleep(CHARGE_DURATION)
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
    time.sleep(0.5)

    # 闭合机械爪进行抓取
    print("闭合机械爪抓取物体...")
    if not close_gripper(ep_gripper):
        print("闭合机械爪失败")
        return False

    # 确认抓取
    time.sleep(GRAB_CONFIRMATION_TIME)
    is_object_grabbed = True
    print("物体抓取完成")

    # 5. 寻找标识并放下
    print("=== 阶段5: 寻找标识并放下 ===")

    # 抬起机械臂
    if move_arm_up(ep_arm):
        print("机械臂抬起成功")

        # 寻找marker并移动到附近
        print("寻找目标marker...")
        success = move_to_marker(ep_chassis, ep_camera, ep_robot.vision, model, device)

        if success:
            print("已到达目标marker附近")
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            time.sleep(2)
            ep_sensor.unsub_distance()
            time.sleep(1)

            # 放下机械臂
            if move_arm_down(ep_arm):
                print("机械臂放下成功")

                # 松开机械爪
                if open_gripper(ep_gripper):
                    print("物体已放下")
                else:
                    print("松开机械爪失败")
            else:
                print("放下机械臂失败")

            ep_sensor.sub_distance(freq=5, callback=distance_callback)
        else:
            print("未能成功到达目标marker")
            return False
    else:
        print("抬起机械臂失败")
        return False

    # 6. 后退一段距离
    print("=== 阶段6: 后退 ===")
    if not backward_robot(ep_chassis, BACKWARD_DISTANCE):
        print("后退失败")
        return False

    print("单次任务循环完成")
    return True


def check_gpu_status():
    """检查GPU状态"""
    print("检查GPU状态...")
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU可用: {gpu_name}")
        print(f"GPU总内存: {gpu_memory:.1f} GB")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    else:
        device = 'cpu'
        print("GPU不可用，将使用CPU进行推理")
    return device


def main():
    global current_distance

    # 1. 检查GPU状态
    device = check_gpu_status()

    # 2. 初始化YOLOv8模型
    print("Loading YOLOv8 model...")
    try:
        model = YOLO(MODEL_PATH)
        model.to(device)  # 明确指定设备
        print(f"Model loaded successfully on {device}.")
        print(f"模型输入尺寸: {YOLO_IMG_SIZE}")
        print(f"推理频率: 每{PROCESS_EVERY_N_FRAMES}帧处理一次")
        print(f"目标选择策略: {TARGET_SELECTION_STRATEGY}")
        print(f"对齐稳定帧数: {ALIGNMENT_STABILITY_FRAMES}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. 初始化机器人
    print("Initializing robot...")
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="ap")
        print("Robot initialized.")
    except Exception as e:
        print(f"Failed to initialize robot: {e}")
        return

    # 4. 获取机器人组件
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gripper = ep_robot.gripper
    ep_sensor = ep_robot.sensor
    ep_arm = ep_robot.robotic_arm

    # 5. 启动摄像头流和测距仪订阅
    print("Starting camera stream and distance sensor...")
    ep_camera.start_video_stream(display=False)
    ep_sensor.sub_distance(freq=5, callback=distance_callback)  # 降低频率避免过多打印

    task_cycle = 0
    user_quit = False

    try:
        print("Starting main loop... Press 'q' in the OpenCV window to quit.")

        while not user_quit:
            task_cycle += 1
            print(f"\n{'=' * 60}")
            print(f"开始执行第 {task_cycle} 轮任务循环")
            print(f"{'=' * 60}")

            # 执行单次任务循环
            cycle_success = execute_single_task_cycle(ep_robot, model, device)

            if cycle_success:
                print(f"第 {task_cycle} 轮任务循环成功完成")
            else:
                print(f"第 {task_cycle} 轮任务循环失败")

            print(f"{'=' * 60}\n")

            # 检查用户退出
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    user_quit = True
                    print("User requested quit via 'q' key.")
            except:
                pass

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