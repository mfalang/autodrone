#!/usr/bin/env python3

import sys
import time
from tracemalloc import start
import rospy

import std_msgs.msg
from drone_interface.msg import AttitudeSetpoint

class AttitudeMission():

    def __init__(self):
        rospy.init_node("attitude_control_test", anonymous=False)

        self.attitude_setpoint_publisher = rospy.Publisher(
            "drone/cmd/set_attitude", AttitudeSetpoint, queue_size=1
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

    def _create_attitude_setpoint_message(self, roll, pitch, yaw_rate, vz):
        rospy.loginfo(f"Setting attitude to R:{roll}, P:{pitch}, dY={yaw_rate}, vz={vz}")
        msg = AttitudeSetpoint()
        msg.header.stamp = rospy.Time.now()
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw_rate = yaw_rate
        msg.climb_rate = vz

        return msg

    def _set_attitude(self, roll, pitch, yaw_rate, vz):
        msg = self._create_attitude_setpoint_message(roll, pitch, yaw_rate, vz)
        self.attitude_setpoint_publisher.publish(msg)

    def _hover(self):
        self._set_attitude(0, 0, 0, 0)

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

class PitchAndRollTest(AttitudeMission):

    def __init__(self):
        super().__init__()

    def start(self):
        self._get_keyboard_input("Ready to take off? (yes) ", "yes")
        self._takeoff()

        pitch_angle = 5 # deg
        control_loop_hz = 20
        command_duration = 1 # sec

        num_commands = command_duration * control_loop_hz

        rate = rospy.Rate(control_loop_hz)

        self._get_keyboard_input("Pitch forward? (yes) ", "yes")
        start_time = time.time()
        for i in range(num_commands):
            self._set_attitude(0, pitch_angle, 0, 0)
            rate.sleep()
        self._hover()

        print(f"Ran command for {time.time() - start_time:4f} seconds")

        self._get_keyboard_input("Pitch backward? (yes) ", "yes")

        start_time = time.time()
        for i in range(num_commands):
            self._set_attitude(0, -pitch_angle, 0, 0)
            rate.sleep()
        self._hover()
        print(f"Ran command for {time.time() - start_time:4f} seconds")

        roll_angle = 5 # deg

        self._get_keyboard_input("Roll right? (yes) ", "yes")

        start_time = time.time()
        for i in range(num_commands):
            self._set_attitude(roll_angle, 0, 0, 0)
            rate.sleep()
        self._hover()
        print(f"Ran command for {time.time() - start_time:4f} seconds")

        self._get_keyboard_input("Roll left? (yes) ", "yes")

        start_time = time.time()
        for i in range(num_commands):
            self._set_attitude(-roll_angle, 0, 0, 0)
            rate.sleep()
        self._hover()
        print(f"Ran command for {time.time() - start_time:4f} seconds")

        self._get_keyboard_input("Roll right and pitch forward? (yes) ", "yes")

        start_time = time.time()
        for i in range(num_commands):
            self._set_attitude(roll_angle, pitch_angle, 0, 0)
            rate.sleep()
        self._hover()
        print(f"Ran command for {time.time() - start_time:4f} seconds")

        self._get_keyboard_input("Roll left and pitch backward? (yes) ", "yes")

        start_time = time.time()
        for i in range(num_commands):
            self._set_attitude(-roll_angle, -pitch_angle, 0, 0)
            rate.sleep()
        self._hover()
        print(f"Ran command for {time.time() - start_time:4f} seconds")

        self._get_keyboard_input("Mission finished, land? (yes) ", "yes")
        self._land()

def main():
    ans = input("Choose mission\n" \
        "(1 = takeoff - pitch forward for 1 sec - pitch backward for 1 sec - roll right for 1 sec - roll left 1 sec - roll and pitch right/forward for 1 sec - roll and pitch left/backward for 1 sec - landing)\n" \
        )

    if ans == "1":
        mission = PitchAndRollTest()
    else:
        print(f"Invalid choice {ans}")
        sys.exit(0)

    mission.start()

if __name__ == "__main__":
    main()


