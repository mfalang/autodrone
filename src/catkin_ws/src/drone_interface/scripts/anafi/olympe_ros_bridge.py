#!/usr/bin/env python3

# Script for interfacing to the Anafi-drone through ROS topics.

import sys
import rospy
import olympe
import threading

from anafi_data_publisher import TelemetryPublisher, CameraPublisher
from command_listener import CommandListener

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

class OlympeRosBridge():
    """
    Class which bridges Olympe and ROS, making it possible to receive the data
    from the Anafi drone via ROS topics as well as sending commands to the
    drone via ROS topics.
    """
    def __init__(self, drone_ip):
        rospy.init_node("anafi_interface", anonymous=False)

        self.drone = olympe.Drone(drone_ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()

        self.command_listener = CommandListener(self.drone)
        self.telemetry_publisher = TelemetryPublisher(self.drone)
        self.camera_streamer = CameraPublisher(self.drone)

    def start(self):
        """
        Start the interface.
        """

        rospy.sleep(1)
        self.command_listener.init(camera_angle=-90)
        rospy.sleep(1)

        threading.Thread(target=self.telemetry_publisher.publish, args=(), daemon=True).start()
        threading.Thread(target=self.camera_streamer.publish, args=(), daemon=True).start()

        rospy.spin()

def main():
    args = sys.argv[1:]

    if args[0] == "physical":
        drone_ip = "192.168.42.1"
    elif args[0] == "simulation":
        drone_ip = "10.202.0.1"
    else:
        rospy.logerr("Incorrect argument, must be <physical/simulator>")
        sys.exit()

    anafi_interface = OlympeRosBridge(drone_ip)
    anafi_interface.start()


if __name__ == "__main__":
    main()
