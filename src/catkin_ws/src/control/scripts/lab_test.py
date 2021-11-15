#!/usr/bin/env python3

import sys
import rospy
from drone_interface.msg import PositionSetpointRelative
import std_msgs.msg

class Mission():

    def __init__(self):
        rospy.init_node("lab_test_control", anonymous=False)

        self.setpoint_publisher = rospy.Publisher(
            "drone/cmd/set_position_relative", PositionSetpointRelative, queue_size=10
        )

        self.takeoff_publisher = rospy.Publisher(
            "drone/cmd/takeoff", std_msgs.msg.Empty, queue_size=10
        )

        self.land_publisher = rospy.Publisher(
            "drone/cmd/land", std_msgs.msg.Empty, queue_size=10
        )

    def _takeoff(self):
        rospy.loginfo("Taking off")
        self.takeoff_publisher.publish(std_msgs.msg.Empty())

    def _land(self):
        rospy.loginfo("Landing")
        self.land_publisher.publish(std_msgs.msg.Empty())

    def _create_setpoint_message(self, dx, dy, dz, dpsi,
        max_horizontal_speed=0.5, max_vertical_speed=0.5, max_yaw_rotation_speed=45
    ):
        rospy.loginfo(f"Moving to x={dx}, y={dy}, z={dz}, psi={dpsi*180/3.1415}")
        msg = PositionSetpointRelative()
        msg.header.stamp = rospy.Time.now()
        msg.dx = dx
        msg.dy = dy
        msg.dz = dz
        msg.dpsi = dpsi
        msg.max_horizontal_speed = max_horizontal_speed
        msg.max_vertical_speed = max_vertical_speed
        msg.max_yaw_rotation_speed = max_yaw_rotation_speed

        return msg

    def _abort(self):
        rospy.loginfo("Aborting, landing")
        self.land_publisher.publish(std_msgs.msg.Empty())
        self._get_keyboard_input("Exit program? (yes) ", "yes")
        sys.exit(0)

    def _get_keyboard_input(self, display_msg, expected_ans):
        ans = ""
        while ans != expected_ans:
            ans = input(f"{display_msg}").lower()
            if ans == "abort" or ans == "a":
                self._abort()

class TakeoffLand(Mission):
    def __init__(self):
        super().__init__()

    def start(self):
        self._get_keyboard_input("Ready to take off? (yes) ", "yes")
        self._takeoff()

        self._get_keyboard_input("Land? (yes) ", "yes")
        self._land()

class ForwardBackward(Mission):
    def __init__(self):
        super().__init__()

    def start(self):

        self._get_keyboard_input("Ready to take off? (yes) ", "yes")
        self._takeoff()

        self._get_keyboard_input("Go up 0.5m? (yes/abort) ", "yes")
        msg = self._create_setpoint_message(0, 0, -0.5, 0)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Go forward 1m? (yes/abort) ", "yes")
        msg = self._create_setpoint_message(1, 0, 0, 0)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Go backward 2m? (yes/abort) ", "yes")
        msg = self._create_setpoint_message(-2, 0, 0, 0)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Go forward 1m? (yes/abort) ", "yes")
        msg = self._create_setpoint_message(1, 0, 0, 0)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Land? (yes) ", "yes")
        self._land()

class SquareMission(Mission):

    def __init__(self):
        super().__init__()

    def start(self):
        rospy.loginfo("Starting mission")

        self._get_keyboard_input("Ready to take off? (yes) ", "yes")
        self._takeoff()

        self._get_keyboard_input("Go to first checkpoint (yes/abort) ", "yes")

        # Move in a square
        msg = self._create_setpoint_message(1, 1, 0, -3.1415/2)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Go to second checkpoint (yes/abort) ", "yes")
        msg = self._create_setpoint_message(2, 0, 0, -3.1415/2)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Go to third checkpoint (yes/abort) ", "yes")
        msg = self._create_setpoint_message(2, 0, 0, -3.1415/2)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Go to forth checkpoint (yes/abort) ", "yes")
        msg = self._create_setpoint_message(2, 0, 0, -3.1415/2)
        self.setpoint_publisher.publish(msg)

        self._get_keyboard_input("Go to fifth checkpoint (yes/abort) ", "yes")
        msg = self._create_setpoint_message(1, -1, 0, 0)
        self.setpoint_publisher.publish(msg)

        # Land
        self._get_keyboard_input("Land? (yes) ", "yes")
        self._land()

def main():
    ans = input("Choose mission\n" \
        "(1 = takeoff - landing)\n" \
        "(2 = takeoff - 0.5m up - 1m forward - 2m backward - 1m forwards - landing)\n" \
        "(3 = takeoff - fly square (2x2m) - landing\n")

    if ans == "1":
        mission = TakeoffLand()
    elif ans == "2":
        mission = ForwardBackward()
    elif ans == "3":
        mission = SquareMission()
    else:
        print(f"Invalid choice {ans}")
        sys.exit(0)

    mission.start()

if __name__ == "__main__":
    main()