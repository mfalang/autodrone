#!/usr/bin/env python3

# Script for interfacing to the Anafi-drone through ROS topics.

import os
import sys
import yaml
import rospy
import olympe
import threading

from anafi_data_publisher import GpsPublisher, TelemetryPublisher, CameraPublisher
from command_listener import CommandListener

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

class OlympeRosBridge():
    """
    Class which bridges Olympe and ROS, making it possible to receive the data
    from the Anafi drone via ROS topics as well as sending commands to the
    drone via ROS topics.
    """
    def __init__(self):
        rospy.init_node("anafi_interface", anonymous=False)

        config_file = rospy.get_param("~config_file")
        script_dir = os.path.dirname(os.path.realpath(__file__))

        try:
            with open(f"{script_dir}/../../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        drone_ip = rospy.get_param("~drone_ip")

        self.drone = olympe.Drone(drone_ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()

        self.command_listener = CommandListener(self.drone)
        # self.telemetry_publisher = TelemetryPublisher(self.drone)
        self.telemetry_publisher = TelemetryPublisher(
            self.drone, self.config["drone"]["topics"]["telemetry"]
        )
        self.gps_publisher = GpsPublisher(
            self.drone, self.config["drone"]["topics"]["gnss"]
        )
        self.camera_streamer = CameraPublisher(
            self.drone, self.config["drone"]["topics"]["camera"]
        )

    def start(self):
        """
        Start the interface.
        """

        rospy.sleep(1)
        self.command_listener.init(camera_angle=-90)
        rospy.sleep(1)

        threading.Thread(target=self.telemetry_publisher.publish, args=(), daemon=True).start()
        threading.Thread(target=self.gps_publisher.publish, args=(), daemon=True).start()
        threading.Thread(target=self.camera_streamer.publish, args=(), daemon=True).start()

        rospy.spin()

def main():
    anafi_interface = OlympeRosBridge()
    anafi_interface.start()


if __name__ == "__main__":
    main()
