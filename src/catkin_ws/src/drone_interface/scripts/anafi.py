#!/usr/bin/env python3

# Script for interfacing to the Anafi-drone through ROS topics.

import rospy
import geometry_msgs.msg
import sensor_msgs.msg
import diagnostic_msgs.msg
import olympe
import olympe.messages as olympe_msgs
import olympe.enums.ardrone3 as ardrone3_enums
from scipy.spatial.transform import Rotation
import numpy as np
import os
import pathlib
import time
import cv2 as cv
import cv_bridge
import queue

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

# TODO: Consider merging this class with the TelemetryPublisher class

class StateMonitor():

    def __init__(self, drone):
        self.drone = drone

    def get_attitude_euler(self):
        attitude = self.drone.get_state(olympe_msgs.ardrone3.PilotingState.AttitudeChanged)
        return [attitude["roll"], attitude["pitch"], attitude["yaw"]]

    def get_attitude_quat(self):
        attitude = self.drone.get_state(olympe_msgs.ardrone3.PilotingState.AttitudeChanged)

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

        speed = self.drone.get_state(olympe_msgs.ardrone3.PilotingState.SpeedChanged)

        velocity_ned = np.array([speed["speedX"], speed["speedY"], speed["speedZ"]])

        velocity_body = R_ned_to_body @ velocity_ned

        return velocity_body

    def get_gps_data(self):
        """
        GPS data. 
        Return format: [gps_fix (1/0), latitude, longitude, altitude,
        latitude_accuracy, longitude_accuracy, altitude_accuracy]
        """
        gps_fix = self.drone.get_state(
            olympe_msgs.ardrone3.GPSSettingsState.GPSFixStateChanged
        )["fixed"]

        gps_pos = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.GpsLocationChanged
        )

        return [
            gps_fix, gps_pos["latitude"], gps_pos["longitude"], gps_pos["altitude"],
            gps_pos["latitude_accuracy"], gps_pos["longitude_accuracy"], gps_pos["altitude_accuracy"]
        ]

    def get_flying_state(self):
        flying_state = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.FlyingStateChanged
        )["state"].name

        return flying_state

    def get_gimbal_attitude(self):
        gimbal_attitude = self.drone.get_state(
            olympe_msgs.gimbal.attitude
        )

        return [
            gimbal_attitude[0]["roll_relative"], 
            gimbal_attitude[0]["pitch_relative"],
            gimbal_attitude[0]["yaw_relative"]
        ]

    def get_image(self, image_dir):
        self.drone(olympe_msgs.camera.take_photo(cam_id=0))
        photo_saved = self.drone(
            olympe_msgs.camera.photo_progress(result="photo_saved")
        ).wait()

        media_id = photo_saved.received_events().last().args["media_id"]
        media_download = self.drone(olympe.media.download_media(
            media_id,
            integrity_check=False
        ))
        resources = media_download.as_completed()
        
        for resource in resources:
            resource_id = resource.received_events().last()._resource_id
            if not resource.success():
                rospy.logerr("Failed to download image")
        
        image = cv.imread(f"{image_dir}/{resource_id}")
        
        return image

class RosMsg():

    def __init__(self, msg, type):
        self.msg = msg
        self.type = type
        
class TelemetryPublisher():

    def __init__(self, drone):
        
        self.publish_queue = queue.Queue()

        self.attitude_publisher = rospy.Publisher(
            "anafi/attitude", geometry_msgs.msg.QuaternionStamped, queue_size=10
        )

        self.velocity_publisher = rospy.Publisher(
            "anafi/velocity_body", geometry_msgs.msg.PointStamped, queue_size=10
        )

        self.gps_data_publisher = rospy.Publisher(
            "anafi/gps_data", sensor_msgs.msg.NavSatFix, queue_size=10
        )

        self.flying_state_publisher = rospy.Publisher(
            "anafi/flying_state", diagnostic_msgs.msg.DiagnosticArray, queue_size=10
        )

        self.gimbal_attitude_publisher = rospy.Publisher(
            "anafi/gimbal_attitude", geometry_msgs.msg.PointStamped, queue_size=10
        )

        self.image_publisher = rospy.Publisher(
            "anafi/image_rect_color", sensor_msgs.msg.Image, queue_size=10
        )

        self.drone = drone

        self.state_monitor = StateMonitor(self.drone)

    def init(self):
        # Make directory for images
        script_dir = os.path.dirname(os.path.realpath(__file__))
        today = time.localtime()
        self.image_dir = f"{script_dir}/../../../images" \
            f"/{today.tm_year}-{today.tm_mon}-{today.tm_mday}" \
            f"/{today.tm_hour}-{today.tm_min}-{today.tm_sec}"
        pathlib.Path(self.image_dir).mkdir(parents=True, exist_ok=True)

        self.drone.media.download_dir = self.image_dir

        # Init camera
        self.drone(olympe_msgs.camera.set_camera_mode(cam_id=0, value="photo")).wait()
        self.drone(
            olympe_msgs.camera.set_photo_mode(
                cam_id=0,
                mode="single",
                format="rectilinear",
                file_format="jpeg",
                burst="burst_14_over_1s", # ignored in singel mode
                bracketing="preset_1ev", # ignored in single mode
                capture_interval=1 # ignored in single mode
            )
        ).wait().success()
        rospy.loginfo("Initialized camera")

    def publish(self):
        while not self.publish_queue.empty():
            msg = self.publish_queue.get()
            if msg.type == "attitude":
                self.attitude_publisher.publish(msg.msg)
            elif msg.type == "velocity":
                self.velocity_publisher.publish(msg.msg)
            elif msg.type == "gps_data":
                self.gps_data_publisher.publish(msg.msg)
            elif msg.type == "flying_state":
                self.flying_state_publisher.publish(msg.msg)
            elif msg.type == "gimbal_attitude":
                self.gimbal_attitude_publisher.publish(msg.msg)
            elif msg.type == "image":
                self.image_publisher.publish(msg.msg)
            else:
                rospy.logerr("Invalid message type")

    def collect_data(self):
        # start = time.time()
        self._publish_attitude()
        self._publish_velocity()
        self._publish_gps()
        self._publish_flying_state()
        self._publish_gimbal_attitude()
        # self._publish_image()
        # rospy.loginfo(f"Publishing took {time.time() - start} seconds")

    def collect_image(self):
        self._publish_image()


    def _publish_attitude(self):
        attitude = geometry_msgs.msg.QuaternionStamped()
        attitude.header.stamp = rospy.Time.now()
        [
            attitude.quaternion.x, 
            attitude.quaternion.y, 
            attitude.quaternion.z, 
            attitude.quaternion.w
        ] = self.state_monitor.get_attitude_quat()

        # self.attitude_publisher.publish(attitude)
        self.publish_queue.put(RosMsg(attitude, "attitude"))

    def _publish_velocity(self):
        velocity = geometry_msgs.msg.PointStamped()
        velocity.header.stamp = rospy.Time.now()
        [
            velocity.point.x, 
            velocity.point.y, 
            velocity.point.z
        ] = self.state_monitor.get_velocity()

        # self.velocity_publisher.publish(velocity)
        self.publish_queue.put(RosMsg(velocity, "velocity"))

    def _publish_gps(self):
        gps_data_msg = sensor_msgs.msg.NavSatFix()
        gps_data_msg.header.stamp = rospy.Time.now()
        gps_data = self.state_monitor.get_gps_data()

        gps_data_msg.status.status = gps_data[0]
        gps_data_msg.status.service = sensor_msgs.msg.NavSatStatus.SERVICE_GPS
        gps_data_msg.latitude = gps_data[1]
        gps_data_msg.longitude = gps_data[2]
        gps_data_msg.altitude = gps_data[3]

        gps_data_msg.position_covariance = [
            gps_data[4], 0, 0, 
            0, gps_data[5], 0, 
            0, 0, gps_data[6]
        ]

        gps_data_msg.position_covariance_type = sensor_msgs.msg.NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        # self.gps_data_publisher.publish(gps_data_msg)
        self.publish_queue.put(RosMsg(gps_data_msg, "gps_data"))

    def _publish_flying_state(self):
        flying_state_msg = diagnostic_msgs.msg.DiagnosticArray()
        flying_state_msg.header.stamp = rospy.Time.now()
        flying_state = self.state_monitor.get_flying_state()
        flying_status = diagnostic_msgs.msg.DiagnosticStatus()
        flying_status.name = "Anafi"
        flying_status.level = diagnostic_msgs.msg.DiagnosticStatus.OK
        flying_status.message = flying_state
        flying_state_msg.status = [flying_status]

        # self.flying_state_publisher.publish(flying_state_msg)
        self.publish_queue.put(RosMsg(flying_state_msg, "flying_state"))

    def _publish_gimbal_attitude(self):
        gimbal_attitude_msgs = geometry_msgs.msg.PointStamped()
        gimbal_attitude_msgs.header.stamp = rospy.Time.now()
        [
            gimbal_attitude_msgs.point.x,
            gimbal_attitude_msgs.point.y,
            gimbal_attitude_msgs.point.z
        ] = self.state_monitor.get_gimbal_attitude()

        # self.gimbal_attitude_publisher.publish(gimbal_attitude_msgs)
        self.publish_queue.put(RosMsg(gimbal_attitude_msgs, "gimbal_attitude"))

    def _publish_image(self):
        image_msg = cv_bridge.CvBridge().cv2_to_imgmsg(
            self.state_monitor.get_image(self.image_dir)
        )
        image_msg.header.stamp = rospy.Time.now()
        # self.image_publisher.publish(image_msg)
        self.publish_queue.put(RosMsg(image_msg, "image"))

class OlympeRosBridge():

    def __init__(self, drone_ip):
        rospy.init_node("anafi_interface", anonymous=False)

        self.drone = olympe.Drone(drone_ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()

        self.controller = Controller(self.drone)
        self.telemetry_publisher = TelemetryPublisher(self.drone)

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

        while not rospy.is_shutdown():
            start = time.time()
            self.telemetry_publisher.collect_data()
            self.telemetry_publisher.collect_image()
            self.telemetry_publisher.publish()
            rospy.loginfo(f"Publishing took {time.time() - start} seconds")
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
