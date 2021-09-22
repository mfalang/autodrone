#!/usr/bin/env python3

import rospy
import olympe
import time
from olympe.messages.ardrone3.Piloting import TakeOff, Landing

DRONE_IP = "10.202.0.1"

class SimpleMission():

    def run_mission(self):
        rospy.loginfo("Starting mission")
        
        drone = olympe.Drone(DRONE_IP)
        drone.connect()
        assert drone(TakeOff()).wait().success()
        time.sleep(5)
        assert drone(Landing()).wait().success()
        drone.disconnect()

        rospy.loginfo("Mission finished")
        rospy.shutdown()

def main():
    mission = SimpleMission()
    mission.run_mission()

if __name__ == "__main__":
    main()