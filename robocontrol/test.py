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

from robomaster import robot
import ultralytics
def sub_info_handler(adapter_info):
    # adapter_info 是一个包含 6 个传感器板信息的列表
    # 假设你的红外测距模块连接在第一个传感器板上
    distance_data = adapter_info[3]
    print("Received sensor data:", distance_data)

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # 获取传感器转接模块对象
    sensor_adaptor = ep_robot.sensor

    # 订阅传感器转接板信息，频率为 10 Hz
    sensor_adaptor.sub_distance(freq=10, callback=sub_info_handler)

    try:
        # 保持程序运行，等待数据推送
        while True:
            pass
    except KeyboardInterrupt:
        # 停止订阅并关闭机器人连接
        sensor_adaptor.unsub_adapter()
        ep_robot.close()
        print("Subscription cancelled and robot connection closed.")