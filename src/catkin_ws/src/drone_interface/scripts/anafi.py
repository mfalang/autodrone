#!/usr/bin/env python3

# Script for interfacing to the Anafi-drone through ROS topics.


import rospy
import geometry_msgs.msg
import olympe
import olympe.messages.ardrone3 as ardrone3_msgs
import olympe.enums.ardrone3 as ardrone3_enums
from scipy.spatial.transform import Rotation
import numpy as np

class MotionController():

    TAKINGOFF_STATE = ardrone3_enums.PilotingState.FlyingStateChanged_State.takingoff
    HOVERING_STATE = ardrone3_enums.PilotingState.FlyingStateChanged_State.hovering
    LANDED_STATE = ardrone3_enums.PilotingState.FlyingStateChanged_State.landed
    
    def __init__(self, drone):
        self.drone = drone

    def _get_flying_state(self):
        return self.drone.get_state(
            ardrone3_msgs.PilotingState.FlyingStateChanged
        )["state"]

    def takeoff(self):
        rospy.loginfo("Taking off")
        self.drone(ardrone3_msgs.Piloting.TakeOff())

    def takeoff_complete(self):
        return self._get_flying_state() == self.HOVERING_STATE

    def land(self):
        rospy.loginfo("Landing")
        self.drone(ardrone3_msgs.Piloting.Landing())

    def land_complete(self):
        return self._get_flying_state() == self.LANDED_STATE

    # TODO: Update these to use movebyextended
    def move(self, dx, dy, dz, dpsi):
        rospy.loginfo(f"Moving dx: {dx} dy: {dy} dz: {dz} dpsi: {dpsi}")
        self.drone(ardrone3_msgs.Piloting.moveBy(dx, dy, dz, dpsi))

    def move_complete(self):
        return self._get_flying_state() == self.HOVERING_STATE

# TODO: Consider merging this class with the TelemetryPublisher class
# TODO: Include the rest of the stuff from the drone, like gimbal information,
# state information, images from the camera, etc.
class StateMonitor():

    def __init__(self, drone):
        self.drone = drone

    def get_attitude_euler(self):
        attitude = self.drone.get_state(ardrone3_msgs.PilotingState.AttitudeChanged)
        return [attitude["roll"], attitude["pitch"], attitude["yaw"]]

    def get_attitude_quat(self):
        attitude = self.drone.get_state(ardrone3_msgs.PilotingState.AttitudeChanged)
        
        return Rotation.from_euler(
            "XYZ", 
            [attitude["roll"], attitude["pitch"], attitude["yaw"]], 
            degrees=False
        ).as_quat()

    def get_velocity(self):
        """
        Return velocity in the body frame (x forward, y right, z down)
        """
        [roll, pitch, yaw] = self.get_attitude_euler()

        R_ned_to_body = Rotation.from_euler(
            "xyz", 
            [roll, pitch, yaw], 
            degrees=False
        ).as_matrix().T

        speed = self.drone.get_state(ardrone3_msgs.PilotingState.SpeedChanged)

        velocity_ned = np.array([speed["speedX"], speed["speedY"], speed["speedZ"]])

        velocity_body = R_ned_to_body @ velocity_ned 

        return velocity_body

class TelemetryPublisher():

    def __init__(self, drone):
        self.attitude_publisher = rospy.Publisher(
            "anafi/attitude", geometry_msgs.msg.Quaternion, queue_size=10
        )

        self.velocity_publisher = rospy.Publisher(
            "anafi/velocity_body", geometry_msgs.msg.Point, queue_size=10
        )

        self.drone = drone

        self.state_monitor = StateMonitor(self.drone)

    def publish(self):
        self._publish_attitude()
        self._publish_velocity()

    def _publish_attitude(self):
        attitude = geometry_msgs.msg.Quaternion()
        [attitude.x, attitude.y, attitude.z, attitude.w] = self.state_monitor.get_attitude_quat()
        self.attitude_publisher.publish(attitude)

    def _publish_velocity(self):
        velocity = geometry_msgs.msg.Point()
        [velocity.x, velocity.y, velocity.z] = self.state_monitor.get_velocity()
        self.velocity_publisher.publish(velocity)


class OlympeRosBridge():

    def __init__(self, drone_ip):
        rospy.init_node("anafi_interface", anonymous=False)

        

        self.drone = olympe.Drone(drone_ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()

        self.motion_controller = MotionController(self.drone)
        self.telemetry_publisher = TelemetryPublisher(self.drone)

    def start(self):

        # TODO: Find a way to check for new commands to run while at the same time
        # publishing telemetry continuously. This could e.g. be a listener that
        # just subscribes to the input command topics and then adds them to a 
        # queue or something, and then this queue is continuously checked in order
        # and any elements in it executed. These elements would then be added in
        # a callback function for the subscribed topic

        while not rospy.is_shutdown():
            self.telemetry_publisher.publish()
            rospy.sleep(0.2)

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