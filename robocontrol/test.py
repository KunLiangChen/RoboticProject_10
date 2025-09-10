# # -*-coding:utf-8-*-
# # Copyright (c) 2020 DJI.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License in the file LICENSE.txt or at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
#
# import time
# import robomaster
# from robomaster import robot
#
#
# def sub_data_handler(sub_info):
#     # 完全闭合 closed, 完全张开opened, 处在中间位置normal.
#     status = sub_info
#     print("gripper status:{0}.".format(status))
#
#
# if __name__ == '__main__':
#
#     ep_robot = robot.Robot()
#     ep_robot.initialize(conn_type="ap")
#
#     ep_gripper = ep_robot.gripper
#     # 订阅机械爪状态
#     ep_camera = ep_robot.camera
#     ep_camera.start_video_stream()
#     time.sleep(10)
#     ep_camera.stop_video_stream()
#     ep_gripper.sub_status(freq=5, callback=sub_data_handler)
#     ep_gripper.open()
#     time.sleep(3)
#     ep_gripper.close()
#     time.sleep(3)
#     ep_gripper.unsub_status()
#     ep_robot.close()
# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
DISTANCE_SENSOR_INDEX = 3  # 测距仪连接的传感器板索引 (根据实际情况调整)
from robomaster import robot
import time
# import ultralytics
def sub_info_handler(adapter_info):
    # adapter_info 是一个包含 6 个传感器板信息的列表
    # 假设你的红外测距模块连接在第一个传感器板上
    distance_data = adapter_info[3]
    print("Received sensor data:", distance_data)

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
        current_distance = sensor_data  # <--- 请根据实际情况调整索引
        print(f"Distance: {current_distance} cm")
    except (IndexError, TypeError) as e:
        print(f"Error reading distance data: {e}")
        # 打印原始数据以便调试
        print(f"Raw adapter info received: {adapter_info}")
        current_distance = None


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_gripper = ep_robot.gripper
    ep_gripper.close(power=50)
    time.sleep(0.5)
    ep_gripper.pause()

    # 获取传感器转接模块对象
    sensor_adaptor = ep_robot.sensor

    # 订阅传感器转接板信息，频率为 10 Hz
    sensor_adaptor.sub_distance(freq=10, callback=distance_callback)

    try:
        # 保持程序运行，等待数据推送
        while True:
            pass
    except KeyboardInterrupt:
        # 停止订阅并关闭机器人连接
        sensor_adaptor.unsub_adapter()
        ep_robot.close()
        print("Subscription cancelled and robot connection closed.")