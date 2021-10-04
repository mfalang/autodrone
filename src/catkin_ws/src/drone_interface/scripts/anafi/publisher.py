
import os
import time
import queue
import pathlib

import rospy
import sensor_msgs.msg
import geometry_msgs.msg
import diagnostic_msgs.msg

import olympe
import olympe.messages as olympe_msgs

import cv2 as cv
import cv_bridge
import numpy as np
from scipy.spatial.transform import Rotation


class Publisher():

    def __init__(self, drone):

        self.drone = drone
        self.image_dir = ""

        # Queue for storing messages ready for publishing
        self.publish_queue = queue.Queue()

        # Topics to publish to
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
        rospy.loginfo("Initialized Anafi publisher")

    def publish(self):
        """
        Publish all messages currently residing in the publishing queue.
        """
        while not self.publish_queue.empty():
            element = self.publish_queue.get()
            if element["type"] == "attitude":
                self.attitude_publisher.publish(element["msg"])
            elif element["type"] == "velocity":
                self.velocity_publisher.publish(element["msg"])
            elif element["type"] == "gps_data":
                self.gps_data_publisher.publish(element["msg"])
            elif element["type"] == "flying_state":
                self.flying_state_publisher.publish(element["msg"])
            elif element["type"] == "gimbal_attitude":
                self.gimbal_attitude_publisher.publish(element["msg"])
            elif element["type"] == "image":
                self.image_publisher.publish(element["msg"])
            else:
                rospy.logerr("Invalid message type")

    def collect_telemetry(self):
        """
        Collect telemetry data from drone. This should be run in its own thread.
        """
        while not rospy.is_shutdown():
            self._collect_attitude()
            self._collect_velocity()
            self._collect_gps_data()
            self._collect_flying_state()
            self._collect_gimbal_attitude()

    def collect_image(self):
        """
        Collect an image from the camera on the drone. This should be run in its
        own thread.
        """
        while not rospy.is_shutdown():
            self._collect_image()

    def _add_msg_to_queue(self, msg, msg_type):
        self.publish_queue.put({"msg": msg, "type": msg_type})

    def _collect_attitude(self):
        attitude_msg = geometry_msgs.msg.QuaternionStamped()
        attitude_msg.header.stamp = rospy.Time.now()

        att_euler = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.AttitudeChanged
        )

        [
            attitude_msg.quaternion.x,
            attitude_msg.quaternion.y,
            attitude_msg.quaternion.z,
            attitude_msg.quaternion.w
        ] = Rotation.from_euler(
            "XYZ",
            [att_euler["roll"], att_euler["pitch"], att_euler["yaw"]],
            degrees=False
        ).as_quat()

        self._add_msg_to_queue(attitude_msg, "attitude")

    def _collect_velocity(self):
        velocity_msg = geometry_msgs.msg.PointStamped()
        velocity_msg.header.stamp = rospy.Time.now()

        att_euler = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.AttitudeChanged
        )

        R_ned_to_body = Rotation.from_euler(
            "xyz",
            [att_euler["roll"], att_euler["pitch"], att_euler["yaw"]],
            degrees=False
        ).as_matrix().T

        speed = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.SpeedChanged
        )

        velocity_ned = np.array([speed["speedX"], speed["speedY"], speed["speedZ"]])

        velocity_body = R_ned_to_body @ velocity_ned

        [
            velocity_msg.point.x,
            velocity_msg.point.y,
            velocity_msg.point.z
        ] = velocity_body

        self._add_msg_to_queue(velocity_msg, "velocity")

    def _collect_gps_data(self):
        gps_data_msg = sensor_msgs.msg.NavSatFix()
        gps_data_msg.header.stamp = rospy.Time.now()

        gps_fix = self.drone.get_state(
            olympe_msgs.ardrone3.GPSSettingsState.GPSFixStateChanged
        )["fixed"]

        gps_pos = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.GpsLocationChanged
        )

        gps_data_msg.status.status = gps_fix
        gps_data_msg.status.service = sensor_msgs.msg.NavSatStatus.SERVICE_GPS
        gps_data_msg.latitude = gps_pos["latitude"]
        gps_data_msg.longitude = gps_pos["longitude"]
        gps_data_msg.altitude = gps_pos["altitude"]

        gps_data_msg.position_covariance = [
            gps_pos["latitude_accuracy"], 0, 0,
            0, gps_pos["longitude_accuracy"], 0,
            0, 0, gps_pos["altitude_accuracy"]
        ]

        gps_data_msg.position_covariance_type = sensor_msgs.msg.NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self._add_msg_to_queue(gps_data_msg, "gps_data")

    def _collect_flying_state(self):
        flying_state_msg = diagnostic_msgs.msg.DiagnosticArray()
        flying_state_msg.header.stamp = rospy.Time.now()
        flying_state = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.FlyingStateChanged
        )["state"].name
        flying_status = diagnostic_msgs.msg.DiagnosticStatus()
        flying_status.name = "Anafi"
        flying_status.level = diagnostic_msgs.msg.DiagnosticStatus.OK
        flying_status.message = flying_state
        flying_state_msg.status = [flying_status]

        self._add_msg_to_queue(flying_state_msg, "flying_state")

    def _collect_gimbal_attitude(self):
        gimbal_attitude_msgs = geometry_msgs.msg.PointStamped()
        gimbal_attitude_msgs.header.stamp = rospy.Time.now()

        gimbal_attitude = self.drone.get_state(
            olympe_msgs.gimbal.attitude
        )

        gimbal_attitude_msgs.point.x = gimbal_attitude[0]["roll_relative"]
        gimbal_attitude_msgs.point.y = gimbal_attitude[0]["pitch_relative"]
        gimbal_attitude_msgs.point.z = gimbal_attitude[0]["yaw_relative"]

        self._add_msg_to_queue(gimbal_attitude_msgs, "gimbal_attitude")

    def _collect_image(self):
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

        image = cv.imread(f"{self.image_dir}/{resource_id}")

        image_msg = cv_bridge.CvBridge().cv2_to_imgmsg(image)
        image_msg.header.stamp = rospy.Time.now()

        self._add_msg_to_queue(image_msg, "image")
