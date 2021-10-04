
import rospy
import olympe
import olympe.messages as olympe_msgs
import geometry_msgs.msg
import sensor_msgs.msg
import diagnostic_msgs.msg
from scipy.spatial.transform import Rotation
import numpy as np
import os
import pathlib
import time
import cv2 as cv
import cv_bridge
import queue


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
        while not rospy.is_shutdown():
            self._publish_attitude()
            self._publish_velocity()
            self._publish_gps()
            self._publish_flying_state()
            self._publish_gimbal_attitude()
        # self._publish_image()
        # rospy.loginfo(f"Publishing took {time.time() - start} seconds")

    def collect_image(self):
        while not rospy.is_shutdown():
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