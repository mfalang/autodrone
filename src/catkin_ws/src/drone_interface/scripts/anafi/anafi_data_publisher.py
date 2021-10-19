
import os
import csv
import time
import queue
import pathlib
import threading

import rospy
import sensor_msgs.msg
import geometry_msgs.msg
import drone_interface.msg

import olympe
import olympe.messages as olympe_msgs
import olympe.enums as olympe_enums

import cv2 as cv
import cv_bridge
import numpy as np
from scipy.spatial.transform import Rotation

class GenericMessagePublisher():
    """
    Helper class which expands on the functionality of the ROS Publisher class.
    """

    def __init__(self, topic_name, msg_type, collect_function, publish_rate):
        """

        Parameters
        ----------
        topic_name : string
            Name of the topic this publisher should publish to
        msg_type : ROS message
            message type this publisher should publish
        collect_function : func
            Function where this publisher can get message to be published
        publish_rate : int
            Rate at which this publisher should publish data (in Hz). If rate is
            -1 the publisher will publish as often as possible
        """
        self.publisher = rospy.Publisher(
            topic_name, msg_type, queue_size=10
        )
        self.rate = publish_rate
        self.prev_publish_time = None
        self.collect_function = collect_function

    def publish(self):
        """
        Publish data on this publisher's topic.
        """
        self.prev_publish_time = rospy.Time.now()
        msg = self.collect_function()
        if msg is not None:
            self.publisher.publish(msg)

    def should_publish(self):
        """
        Check if this publisher is ready to publish yet (i.e. if more than
        1/rate time has passed since it last published).

        Returns
        -------
        bool
            True if more than 1/rate has passed, false if not
        """
        if self.prev_publish_time is None or self.rate == -1:
            return True
        else:
            return rospy.Time.now().to_time() >= self.prev_publish_time.to_time() + 1/self.rate


class TelemetryPublisher():
    """
    Class to publish telemetry data from Anafi drone
    """
    def __init__(self, drone):

        self.drone = drone
        self.image_dir = ""

        # Queue for storing messages ready for publishing
        self.publish_queue = queue.Queue()

        # List of all telemetry publishers
        self.telemetry_publishers = []

        # Topics to publish to
        self.telemetry_publishers.append(GenericMessagePublisher(
            "drone/out/attitude", geometry_msgs.msg.QuaternionStamped,
            self._collect_attitude, publish_rate=500
        ))

        self.telemetry_publishers.append(GenericMessagePublisher(
            "drone/out/velocity_body", geometry_msgs.msg.PointStamped,
            self._collect_velocity, publish_rate=500
        ))

        self.telemetry_publishers.append(GenericMessagePublisher(
            "drone/out/gps_data", sensor_msgs.msg.NavSatFix,
            self._collect_gps_data, publish_rate=10
        ))

        self.telemetry_publishers.append(GenericMessagePublisher(
            "drone/out/flying_state", drone_interface.msg.FlyingState,
            self._collect_flying_state, publish_rate=10
        ))

        self.telemetry_publishers.append(GenericMessagePublisher(
            "drone/out/gimbal_attitude", drone_interface.msg.GimbalAttitude,
            self._collect_gimbal_attitude, publish_rate=10
        ))

        self.telemetry_publishers.append(GenericMessagePublisher(
            "drone/out/battery_data", sensor_msgs.msg.BatteryState,
            self._collect_battery_data, publish_rate=0.2
        ))

        self.telemetry_publishers.append(GenericMessagePublisher(
            "drone/out/saturation_limits", drone_interface.msg.SaturationLimits,
            self._collect_saturation_limits, publish_rate=2
        ))

        rospy.loginfo("Initialized telemetry publisher")

    def publish(self):
        """
        Publish telemetry data from the Anafi drone. This function blocks so it
        should be run in its own thread.
        """
        while not rospy.is_shutdown():
            for publisher in self.telemetry_publishers:
                if publisher.should_publish():
                    publisher.publish()

    def _collect_attitude(self):
        """
        Get the attitude of the drone in Euler angles and convert a quaternion
        used in the ROS message for attitude.

        Returns
        -------
        geometry_msgs.msg.QuaternionStamped
            ROS message with drone attitude
        """
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

        return attitude_msg

    def _collect_velocity(self):
        """
        Get the body velocity of the the drone and put this in a ROS message.
        The format for the velocity is

        x = forward velocity
        y = left velocity
        z = down velocity

        Returns
        -------
        geometry_msgs.msg.PointStamped
            ROS message with drone body velocity
        """
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

        return velocity_msg

    def _collect_gps_data(self):
        """
        Get drone GPS position and other data and store this in a ROS message.

        Returns
        -------
        sensor_msgs.msg.NavSatFix
            ROS message containing drone location as well as other GPS related
            data
        """
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

        return gps_data_msg

    def _collect_flying_state(self):
        """
        Get the flying state of the Anafi and store this in a ROS message.

        Returns
        -------
        drone_interface.msg.PositionSetpointRelative
            ROS message containing the flying state of the drone
        """

        flying_state_msg = drone_interface.msg.FlyingState()
        flying_state_msg.header.stamp = rospy.Time.now()
        flying_state = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingState.FlyingStateChanged
        )["state"].name
        flying_state_msg.flying_state = flying_state

        return flying_state_msg

    def _collect_gimbal_attitude(self):
        """
        Get the gimbal attitude of the drone. Attitude is relative to the
        drone body frame (roll axis forward, pitch axis left, yaw-axis down).
        Counter clockwise angle positive. All angles zero when camera is
        pointing straight ahead.

        Returns
        -------
        drone_interface.msg.GimbalAttitude
            ROS message containing the gimbal attitude
        """
        gimbal_attitude_msg = drone_interface.msg.GimbalAttitude()
        gimbal_attitude_msg.header.stamp = rospy.Time.now()

        gimbal_attitude = self.drone.get_state(
            olympe_msgs.gimbal.attitude
        )

        gimbal_attitude_msg.roll = gimbal_attitude[0]["roll_relative"]
        gimbal_attitude_msg.pitch = gimbal_attitude[0]["pitch_relative"]
        gimbal_attitude_msg.yaw = gimbal_attitude[0]["yaw_relative"]

        return gimbal_attitude_msg

    def _collect_battery_data(self):
        """
        Get battery information about the drone and store this in a ROS message.

        Returns
        -------
        sensor_msgs.msg.BatteryState
            ROS message containing information about the drone battery
        """
        battery_msg = sensor_msgs.msg.BatteryState()
        battery_msg.header.stamp = rospy.Time.now()

        battery_msg.voltage = self.drone.get_state(
            olympe_msgs.battery.voltage
        )["voltage"]

        battery_msg.percentage = self.drone.get_state(
            olympe_msgs.common.CommonState.BatteryStateChanged
        )["percent"]/100

        battery_msg.power_supply_status = sensor_msgs.msg.BatteryState.POWER_SUPPLY_STATUS_DISCHARGING

        battery_alert = self.drone.get_state(
            olympe_msgs.battery.alert
        )

        no_alert = olympe_enums.battery.alert_level.none
        if battery_alert[olympe_enums.battery.alert.power_level]["level"] != no_alert:
            battery_msg.power_supply_health = sensor_msgs.msg.BatteryState.POWER_SUPPLY_HEALTH_DEAD
        elif battery_alert[olympe_enums.battery.alert.too_hot]["level"] != no_alert:
            battery_msg.power_supply_health = sensor_msgs.msg.BatteryState.POWER_SUPPLY_HEALTH_OVERHEAT
        elif battery_alert[olympe_enums.battery.alert.too_cold]["level"] != no_alert:
            battery_msg.power_supply_health = sensor_msgs.msg.BatteryState.POWER_SUPPLY_HEALTH_COLD
        else:
            battery_msg.power_supply_health = sensor_msgs.msg.BatteryState.POWER_SUPPLY_HEALTH_GOOD

        battery_msg.present = True

        battery_msg.serial_number = self.drone.get_state(olympe_msgs.battery.serial)["serial"]

        return battery_msg

    def _collect_saturation_limits(self):
        """
        Get the saturation limits for the different parameters attributed to
        control of the drone.

        Returns
        -------
        drone_interface.msg.SaturationLimits
            Message containing saturation limits
        """
        max_tilt = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingSettingsState.MaxTiltChanged
        )["current"]

        max_yaw_rot_speed = self.drone.get_state(
            olympe_msgs.ardrone3.SpeedSettingsState.MaxRotationSpeedChanged
        )["current"]

        max_roll_pitch_rot_speed = self.drone.get_state(
            olympe_msgs.ardrone3.SpeedSettingsState.MaxPitchRollRotationSpeedChanged
        )["current"]

        max_vertical_speed = self.drone.get_state(
            olympe_msgs.ardrone3.SpeedSettingsState.MaxVerticalSpeedChanged
        )["current"]

        saturation_limits_msg = drone_interface.msg.SaturationLimits()
        saturation_limits_msg.header.stamp = rospy.Time.now()
        saturation_limits_msg.max_tilt_angle = max_tilt
        saturation_limits_msg.max_yaw_rotation_speed = max_yaw_rot_speed
        saturation_limits_msg.max_roll_pitch_rotation_speed = max_roll_pitch_rot_speed
        saturation_limits_msg.max_vertical_speed = max_vertical_speed

        return saturation_limits_msg

class CameraPublisher():
    """
    Class to publish front camera of Anafi drone.
    """

    def __init__(self, drone):
        self.drone = drone

        # Make directory for data
        script_dir = os.path.dirname(os.path.realpath(__file__))
        today = time.localtime()
        self.image_dir = f"{script_dir}/../../../../../../out/images" \
            f"/{today.tm_year}-{today.tm_mon}-{today.tm_mday}" \
            f"/{today.tm_hour}-{today.tm_min}-{today.tm_sec}"
        pathlib.Path(self.image_dir).mkdir(parents=True, exist_ok=True)
        rospy.loginfo(f"Video stream to {self.image_dir}")

        # Set up video streaming
        self.h264_frame_stats = []
        self.h264_stats_file = open(
            os.path.join(self.image_dir, 'h264_stats.csv'), 'w+')
        self.h264_stats_writer = csv.DictWriter(
            self.h264_stats_file, ['fps', 'bitrate'])
        self.h264_stats_writer.writeheader()
        self.image_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()

        self.drone.set_streaming_output_files(
            h264_data_file=os.path.join(self.image_dir, 'h264_data.264'),
            h264_meta_file=os.path.join(self.image_dir, 'h264_metadata.json'),
        )

        # Callbacks for live processing of frames
        self.drone.set_streaming_callbacks(
            raw_cb=self.yuv_frame_cb,
            flush_raw_cb=self.flush_cb,
        )

        # Channel to publish to
        self.publisher = GenericMessagePublisher(
            "drone/out/image_rect_color", sensor_msgs.msg.Image,
            self._collect_image, publish_rate=-1
        )

        rospy.loginfo("Initialized camera publisher")

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

        Parameters
        ----------
        yuv_frame : olympe.VideoFrame
            Video frame
        """
        yuv_frame.ref()
        self.image_queue.put_nowait(yuv_frame)

    def flush_cb(self):
        """
        Function called by Olympe to flush the queue of frames.

        Returns
        -------
        bool
            True when flushing is finished
        """
        with self.flush_queue_lock:
            while not self.image_queue.empty():
                self.image_queue.get_nowait().unref()
        return True

    def publish(self):
        """
        Publishes video stream from Anafi front camera. Blocks, so this function
        Should be run in a separate thread.
        """

        self.drone.start_video_streaming()

        while not rospy.is_shutdown():
            if self.publisher.should_publish():
                self.publisher.publish()

    def _collect_image(self):
        """
        Gets the current image stored in the image queue. If there is no image,
        then this function returns None.

        Returns
        -------
        sensor_msgs.msg.Image
            ROS message containing the image and associated data
        """
        with self.flush_queue_lock:
            if not self.image_queue.empty():
                yuv_frame = self.image_queue.get()

                # Convert yuv frame to OpenCV compatible image array
                info = yuv_frame.info()
                cv_cvt_color_flag = {
                    olympe.PDRAW_YUV_FORMAT_I420: cv.COLOR_YUV2BGR_I420,
                    olympe.PDRAW_YUV_FORMAT_NV12: cv.COLOR_YUV2BGR_NV12,
                }[info["yuv"]["format"]]
                cv_frame = cv.cvtColor(yuv_frame.as_ndarray(), cv_cvt_color_flag)

                # Create ros message
                image_msg = cv_bridge.CvBridge().cv2_to_imgmsg(cv_frame)
                image_msg.header.stamp = rospy.Time.now()

                yuv_frame.unref()
                return image_msg
            else:
                return None