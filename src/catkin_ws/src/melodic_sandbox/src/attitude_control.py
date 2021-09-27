#!/usr/bin/env python3

import rospy
import olympe
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
import olympe.messages.ardrone3.PilotingSettings as piloting_settings
import olympe.messages.ardrone3.SpeedSettings as speed_settings

DRONE_IP = "10.202.0.1"

class TestAttitudeControl():

    def __init__(self, ip):
        self.drone = olympe.Drone(ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()
        
    def start(self):
        rospy.loginfo("Taking off")
        self._print_state()
        assert self.drone(TakeOff()).wait().success()
        rospy.loginfo("Takeoff complete")
        
        # TODO: Try to use the PCMD with a varying input to see if the drone 
        # is able to fly more smoothly towards a target. Just some simple 1/x
        # graph or something as the reference

        self._wait(10)

        # self.drone(piloting_settings.MaxTilt(15)).wait().success()
        # self.drone(speed_settings.MaxPitchRollRotationSpeed(10)).wait().success()

        self._print_state()
        # Test roll
        self._set_attitude(20, 0, 0, 0, 3)
        self._print_state()
        self._wait(10)
        self._print_state()
        self._set_attitude(-20, 0, 0, 0, 3)
        self._print_state()
        self._wait(10)
        self._print_state()

        # Test pitch
        self._set_attitude(0, 20, 0, 0, 3)
        self._wait(10)
        self._set_attitude(0, -20, 0, 0, 3)
        self._wait(10)

        # Test yaw rate
        self._set_attitude(0, 0, 20, 0, 3)
        self._wait(10)
        self._set_attitude(0, 0, -20, 0, 3)
        self._wait(10)

        rospy.loginfo("Landing")

        assert self.drone(Landing()).wait().success()
        self.drone.disconnect()

    def _wait(self, seconds):
        rospy.loginfo("Waiting")
        for i in range(1000 * seconds):
            time.sleep(0.001)
        rospy.loginfo("Resuming")

    def _print_state(self):
        rospy.loginfo(f"State: {self.drone.get_state(olympe.messages.ardrone3.PilotingState.FlyingStateChanged)}")

    def _hover(self):
        self.drone(PCMD(1, 0, 0, 0, 0, 0))

    def _generate_reference(max_angle):
        pass

    def _set_attitude(self, roll, pitch, yaw_rate, throttle, duration):
        rospy.loginfo(f"Setting attitude roll: {roll} pitch: {pitch} yaw_rate: {yaw_rate} throttle: {throttle} duration: {duration}")
        refresh_rate = 20 # Hz
        start = time.time()
        # self.drone.piloting_pcmd(roll, pitch, yaw_rate, throttle, duration)
        for i in range(refresh_rate*duration):
            self.drone(PCMD(1, roll, pitch, yaw_rate, throttle, 0))
            time.sleep(0.05)
        end = time.time()
        rospy.loginfo(f"Held attitude for {end-start} seconds")

def main():
    rospy.init_node("attitude_control_test", anonymous=False)
    mission = TestAttitudeControl(DRONE_IP)
    mission.start()

if __name__ == "__main__":
    main()