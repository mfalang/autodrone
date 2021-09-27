#!/usr/bin/env python3

import rospy
import olympe
import time
import olympe.messages.ardrone3.Piloting as piloting
import olympe.messages.ardrone3.PilotingState as piloting_state

DRONE_IP = "10.202.0.1"

# TODO: Check out moveBy extended as this has speed settings which is probably
# something we want here

class TestPostionControl():

    def __init__(self, ip):
        self.drone = olympe.Drone(ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()
        
    def start(self):
        rospy.loginfo("Taking off")
        assert self.drone(
            piloting.TakeOff()
            >> piloting_state.FlyingStateChanged(state="hovering", _timeout=5)
        ).wait().success()

        # Test moving forward and backward
        dX = 3
        rospy.loginfo(f"Moving {dX}m forwards")
        assert self.drone(
            piloting.moveBy(dX, 0, 0, 0)
            >> piloting_state.FlyingStateChanged(state="hovering", _timeout=5)
        ).wait().success()
        rospy.loginfo("Reached target")

        rospy.loginfo(f"Moving {dX}m backwards")
        assert self.drone(
            piloting.moveBy(-dX, 0, 0, 0)
            >> piloting_state.FlyingStateChanged(state="hovering", _timeout=5)
        ).wait().success()
        rospy.loginfo("Reached target")
        
        rospy.loginfo("Landing")

        assert self.drone(piloting.Landing()).wait().success()
        self.drone.disconnect()

def main():
    rospy.init_node("position_control_test", anonymous=False)
    mission = TestPostionControl(DRONE_IP)
    mission.start()

if __name__ == "__main__":
    main()