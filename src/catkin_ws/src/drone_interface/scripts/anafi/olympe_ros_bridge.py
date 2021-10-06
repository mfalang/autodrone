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

        # TODO: Find a way to check for new commands to run while at the same time
        # publishing telemetry continuously. This could e.g. be a listener that
        # just subscribes to the input command topics and then adds them to a
        # queue or something, and then this queue is continuously checked in order
        # and any elements in it executed. These elements would then be added in
        # a callback function for the subscribed topic

        rospy.sleep(1)
        self.telemetry_publisher.init()
        self.command_listener.init(camera_angle=-90)
        rospy.sleep(1)

        threading.Thread(target=self.telemetry_publisher.publish_telemetry, args=(), daemon=True).start()
        threading.Thread(target=self.telemetry_publisher.publish_image, args=(), daemon=True).start()

        rospy.spin()
        # while not rospy.is_shutdown():
        #     pass
            #start = time.time()
            # self.telemetry_publisher.publish_telemetry()
            #rospy.loginfo(f"Publishing took {time.time() - start} seconds")
            # rospy.sleep(0.2)

        # self.motion_controller.takeoff()
        # while not self.motion_controller.takeoff_complete():
        #     # self.state_monitor.get_speed()
        #     rospy.sleep(0.2)
        # rospy.loginfo("Takeoff complete")
        # rospy.sleep(3)
        # self.motion_controller.move(0, 0, -10, 0)
        # rospy.sleep(0.1)
        # # rospy.sleep(2)
        # # self.motion_controller.move(3, 3, 0, 3.14/2)
        # # rospy.sleep(2)
        # # self.motion_controller.move(3, 3, 0, 3.14/2)
        # while not self.motion_controller.move_complete():
        #     self.state_monitor.get_speed()
        #     rospy.sleep(0.2)
        # rospy.loginfo("Reached target")
        # self.motion_controller.land()
        # while not self.motion_controller.land_complete():
        #     # self.state_monitor.get_speed()
        #     rospy.sleep(0.2)
        # rospy.loginfo("Landing complete")


def main():
    drone_ip = "10.202.0.1"
    anafi_interface = OlympeRosBridge(drone_ip)
    anafi_interface.start()


if __name__ == "__main__":
    main()
