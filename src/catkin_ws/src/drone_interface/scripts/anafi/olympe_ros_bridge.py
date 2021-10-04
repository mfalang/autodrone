#!/usr/bin/env python3

# Script for interfacing to the Anafi-drone through ROS topics.

import rospy
import olympe
import olympe.messages as olympe_msgs
import olympe.enums.ardrone3 as ardrone3_enums
import threading

import publisher

# TODO: Must have an init drone command. There must also be a topic that
# published when the init is done
class Controller():

    TAKINGOFF_STATE = ardrone3_enums.PilotingState.FlyingStateChanged_State.takingoff
    HOVERING_STATE = ardrone3_enums.PilotingState.FlyingStateChanged_State.hovering
    LANDED_STATE = ardrone3_enums.PilotingState.FlyingStateChanged_State.landed

    def __init__(self, drone):
        self.drone = drone

    def init(self, camera_angle):
        # Init gimbal
        max_speed = 90 # Max speeds: Pitch 180, Roll/Yaw 0.

        self.drone(olympe_msgs.gimbal.set_max_speed(
            gimbal_id=0,
            yaw=0,
            pitch=max_speed,
            roll=0,
        ))

        assert self.drone(olympe_msgs.gimbal.max_speed(
            gimbal_id=0,
            current_yaw=0,
            current_pitch=max_speed,
            current_roll=0,
        )).wait().success(), "Failed to set max gimbal speed"

        self.drone(olympe_msgs.gimbal.set_target(
            gimbal_id=0,
            control_mode="position",
            pitch_frame_of_reference="relative",
            pitch=camera_angle,
            roll_frame_of_reference="relative",
            roll=0,
            yaw_frame_of_reference="relative",
            yaw=0
        ))

        assert self.drone(olympe_msgs.gimbal.attitude(
            gimbal_id=0,
            pitch_relative=camera_angle,
        )).wait(5).success(), "Failed to pitch camera"

        rospy.loginfo(f"Initialized gimbal at {camera_angle}")

    def _get_flying_state(self):
        return self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.FlyingStateChanged
        )["state"]

    def takeoff(self):
        rospy.loginfo("Taking off")
        self.drone(olympe_msgs.ardrone3.Piloting.TakeOff())

    def takeoff_complete(self):
        return self._get_flying_state() == self.HOVERING_STATE

    def land(self):
        rospy.loginfo("Landing")
        self.drone(olympe_msgs.ardrone3.Piloting.Landing())

    def land_complete(self):
        return self._get_flying_state() == self.LANDED_STATE

    # TODO: Update these to use movebyextended
    def move(self, dx, dy, dz, dpsi):
        rospy.loginfo(f"Moving dx: {dx} dy: {dy} dz: {dz} dpsi: {dpsi}")
        self.drone(olympe_msgs.ardrone3.Piloting.moveBy(dx, dy, dz, dpsi))

    def move_complete(self):
        return self._get_flying_state() == self.HOVERING_STATE



class OlympeRosBridge():

    def __init__(self, drone_ip):
        rospy.init_node("anafi_interface", anonymous=False)

        self.drone = olympe.Drone(drone_ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()

        self.controller = Controller(self.drone)
        self.telemetry_publisher = publisher.Publisher(self.drone)

    def start(self):

        # TODO: Find a way to check for new commands to run while at the same time
        # publishing telemetry continuously. This could e.g. be a listener that
        # just subscribes to the input command topics and then adds them to a
        # queue or something, and then this queue is continuously checked in order
        # and any elements in it executed. These elements would then be added in
        # a callback function for the subscribed topic

        rospy.sleep(1)
        self.telemetry_publisher.init()
        self.controller.init(camera_angle=-90)
        rospy.sleep(1)

        threading.Thread(target=self.telemetry_publisher.collect_telemetry, args=(), daemon=True).start()
        threading.Thread(target=self.telemetry_publisher.collect_image, args=(), daemon=True).start()

        while not rospy.is_shutdown():
            #start = time.time()
            self.telemetry_publisher.publish()
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
