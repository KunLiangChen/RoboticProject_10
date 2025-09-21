# core/gripper_arm.py

import time


class GripperArmController:
    def __init__(self, gripper, arm):
        self.gripper = gripper
        self.arm = arm

    def open_gripper(self):
        try:
            self.gripper.open()
            time.sleep(2)
            return True
        except Exception as e:
            print(f"[ERROR] Open gripper failed: {e}")
            return False

    def close_gripper(self):
        try:
            self.gripper.close()
            time.sleep(2)
            return True
        except Exception as e:
            print(f"[ERROR] Close gripper failed: {e}")
            return False

    def lift_arm(self):
        try:
            self.arm.move(x=40).wait_for_completed()
            self.arm.move(y=90).wait_for_completed()
            return True
        except Exception as e:
            print(f"[ERROR] Lift arm failed: {e}")
            return False

    def lower_arm(self):
        try:
            self.arm.move(y=-90).wait_for_completed()
            self.arm.move(x=-40).wait_for_completed()
            return True
        except Exception as e:
            print(f"[ERROR] Lower arm failed: {e}")
            return False