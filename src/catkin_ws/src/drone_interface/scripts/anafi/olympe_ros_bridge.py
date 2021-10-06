#!/usr/bin/env python3

# Script for interfacing to the Anafi-drone through ROS topics.

import rospy
import olympe
import olympe.messages as olympe_msgs
import olympe.enums.ardrone3 as ardrone3_enums
import threading

from anafi_data_publisher import AnafiDataPublisher
from command_listener import CommandListener

class OlympeRosBridge():

    def __init__(self, drone_ip):
        rospy.init_node("anafi_interface", anonymous=False)

        self.drone = olympe.Drone(drone_ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()

        self.command_listener = CommandListener(self.drone)
        self.telemetry_publisher = AnafiDataPublisher(self.drone)

    def start(self):

        rospy.sleep(1)
        self.telemetry_publisher.init()
        self.command_listener.init(camera_angle=-90)
        rospy.sleep(1)

        threading.Thread(target=self.telemetry_publisher.publish_telemetry, args=(), daemon=True).start()
        # threading.Thread(target=self.telemetry_publisher.publish_image, args=(), daemon=True).start()

        rospy.spin()

def main():
    drone_ip = "10.202.0.1"
    anafi_interface = OlympeRosBridge(drone_ip)
    anafi_interface.start()


if __name__ == "__main__":
    main()
