import rospy

import olympe.messages as olympe_msgs

import std_msgs.msg
import drone_interface.msg

class CommandListener():
    """
    Class to listen for Anafi commands via ROS topics and then communicate these
    to the drone.
    """
    def __init__(self, drone):

        self.drone = drone

        rospy.Subscriber("drone/cmd/takeoff", std_msgs.msg.Empty, self._takeoff_cb)
        rospy.Subscriber("drone/cmd/land", std_msgs.msg.Empty, self._land_cb)
        rospy.Subscriber("drone/cmd/set_position_relative",
            drone_interface.msg.PositionSetpointRelative, self._move_cb
        )
        rospy.Subscriber("drone/cmd/set_attitude",
            drone_interface.msg.AttitudeSetpoint, self._set_attitude_cb
        )
        rospy.Subscriber("drone/cmd/set_saturation_limits",
            drone_interface.msg.SaturationLimits, self._set_saturation_limits_cb
        )

        # Save default drone saturation limits
        self.max_tilt = self.drone.get_state(
            olympe_msgs.ardrone3.PilotingSettingsState.MaxTiltChanged
        )["current"]

        self.max_yaw_rot_speed = self.drone.get_state(
            olympe_msgs.ardrone3.SpeedSettingsState.MaxRotationSpeedChanged
        )["current"]

        self.max_roll_pitch_rot_speed = self.drone.get_state(
            olympe_msgs.ardrone3.SpeedSettingsState.MaxPitchRollRotationSpeedChanged
        )["current"]

        self.max_vertical_speed = self.drone.get_state(
            olympe_msgs.ardrone3.SpeedSettingsState.MaxVerticalSpeedChanged
        )["current"]

    def init(self, camera_angle):
        """
        Initialize the drone by pointing the camera down.

        Parameters
        ----------
        camera_angle : float
            Start angle of the camera in deg (negative down, 0 horizon)
        """
        # Init gimbal
        max_speed = 90 # Max speeds: Pitch 180, Roll/Yaw 0.

        self.drone(olympe_msgs.gimbal.set_max_speed(
            gimbal_id=0,
            yaw=0,
            pitch=max_speed,
            roll=0,
        ))

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
            pitch_frame_of_reference="relative"
        )).wait(5).success(), "Failed to pitch camera"

        rospy.loginfo(f"Initialized gimbal at {camera_angle} degrees")

        rospy.loginfo("Initialized drone")

    def _takeoff_cb(self, msg):
        """
        Callback function to take off when a message is received on the takeoff
        topic.

        Parameters
        ----------
        msg : std_msgs.msg.Empty
            Empty message indicating we should take off
        """
        rospy.loginfo("Taking off")
        self.drone(olympe_msgs.ardrone3.Piloting.TakeOff())

    def _land_cb(self, msg):
        """
        Callback function to land when a message is received on the land
        topic.

        Parameters
        ----------
        msg : std_msgs.msg.Empty
            Empty message indicating we should land
        """
        rospy.loginfo("Landing")
        self.drone(olympe_msgs.ardrone3.Piloting.Landing())

    def _move_cb(self, msg):
        """
        Callback function to move to a specific point relative to the drone body
        frame. Displacement in x,y,z as well as max horizontal, vertical and yaw
        rotation speed can be specified.

        Parameters
        ----------
        msg : drone_interface.msg.PositionSetpointRelative
            Message containing the displacements and constraints for the
            desired movement of the drone
        """
        rospy.loginfo(f"Moving dx: {msg.dx} dy: {msg.dy} dz: {msg.dz} dpsi: {msg.dpsi}")
        self.drone(olympe_msgs.move.extended_move_by(
            msg.dx, msg.dy, msg.dz, msg.dpsi,
            msg.max_horizontal_speed, msg.max_vertical_speed,
            msg.max_yaw_rotation_speed)
        )

    def _set_attitude_cb(self, msg):
        """
        Callback function to set a desired attitude for the drone. This has the
        same features as a regular radio controller has, namely
        - roll and pitch angles
        - yaw rotation speed
        - throttle

        See the message type drone_interface/msg/AttitudeSetpoint.msg for
        details on the values of the different parameters.

        Note: This should be called with 50 ms (20 Hz) delays, as Olympe only
        uploads new attitude setpoints at that interval.

        Parameters
        ----------
        msg : drone_interface.msg.AttitudeSetpoint
            Message including the setpoints for the desired attitude
        """
        roll_angle_as_percent = int((msg.roll_angle/self.max_tilt)*100)
        pitch_angle_as_percent = int((msg.pitch_angle/self.max_tilt)*100)
        yaw_rotation_speed_as_percent = int((msg.yaw_speed/self.max_yaw_rot_speed)*100)
        throttle_as_percent = int((msg.throttle/self.max_vertical_speed)*100)

        self.drone(olympe_msgs.ardrone3.Piloting.PCMD(
            1, roll_angle_as_percent, pitch_angle_as_percent,
            yaw_rotation_speed_as_percent, throttle_as_percent, 0
        ))

    def _set_saturation_limits_cb(self, msg):
        """
        Callback function to set the saturation limits for parameters on the
        drone related to control. The parameters that can be set are
        - max tilt angle (max roll and pitch angle)
        - max yaw rotational speed
        - max roll and pitch rotational speed
        - max vertical speed

        See the message drone_interface/msg/SaturationLimits.msg for details
        for the available intervals and values for the different parameters.
        All values outside the given intervals will be clamped to be inside of
        them.

        Parameters
        ----------
        msg : drone_interface.msg.SaturationLimits.msg
            Message including the new saturation limits
        """
        if msg.update_max_tilt_angle:
            self.max_tilt = msg.max_tilt_angle
            self.drone(olympe_msgs.ardrone3.PilotingSettings.MaxTilt(
                self.max_tilt
            ))

        if msg.update_max_yaw_rotation_speed:
            self.max_yaw_rot_speed = msg.max_yaw_rotation_speed
            self.drone(olympe_msgs.ardrone3.SpeedSettings.MaxRotationSpeed(
                self.max_yaw_rot_speed
            ))

        if msg.update_max_roll_pitch_rotation_speed:
            self.max_roll_pitch_rot_speed = msg.max_roll_pitch_rotation_speed
            self.drone(olympe_msgs.ardrone3.SpeedSettings.MaxPitchRollRotationSpeed(
                self.max_roll_pitch_rot_speed
            ))

        if msg.update_max_vertical_speed:
            self.max_vertical_speed = msg.max_vertical_speed
            self.drone(olympe_msgs.ardrone3.SpeedSettings.MaxVerticalSpeed(
                self.max_vertical_speed
            ))
