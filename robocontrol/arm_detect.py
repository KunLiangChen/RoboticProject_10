import time

from robomaster import robot
def sub_data_handler(sub_info):
    pos_x, pos_y = sub_info
    print("Robotic Arm: pos x:{0}, pos y:{1}".format(pos_x, pos_y))


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_arm = ep_robot.robotic_arm
    ep_grip = ep_robot.gripper
    ep_arm.sub_position(freq=5, callback=sub_data_handler)
    ep_arm.move(x=40).wait_for_completed()
    ep_arm.move(y=90).wait_for_completed()
    ep_arm.move(y=-90).wait_for_completed()
    ep_arm.move(x=-50).wait_for_completed()
    ep_grip.open()
    time.sleep(2)
    ep_arm.unsub_position()

    ep_robot.close()
    # ep_robot.close()
    #Totally lift up x:134 y:90
    #Totally down x:172 y:-58