
import rospy

import olympe
import olympe.messages as olympe_msgs

import std_msgs.msg
from drone_interface.msg import PositionSetpointRelative

class CommandListener():

    def __init__(self, drone):

        rospy.Subscriber("anafi/cmd/takeoff", std_msgs.msg.Empty, self._takeoff_cb)
        rospy.Subscriber("anafi/cmd/land", std_msgs.msg.Empty, self._land_cb)
        rospy.Subscriber("anafi/cmd/set_position_relative",
            PositionSetpointRelative, self._move_cb
        )

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

        # TODO: Find out why this does not work with physical drone
        # assert self.drone(olympe_msgs.gimbal.max_speed(
        #     gimbal_id=0,
        #     current_yaw=0,
        #     current_pitch=max_speed,
        #     current_roll=0,
        # )).wait(_timeout=5).success(), "Failed to set max gimbal speed"

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
        rospy.loginfo("Intitialized drone")

    def _takeoff_cb(self, msg):
        rospy.loginfo("Taking off")
        self.drone(olympe_msgs.ardrone3.Piloting.TakeOff())

    def _land_cb(self, msg):
        rospy.loginfo("Landing")
        self.drone(olympe_msgs.ardrone3.Piloting.Landing())

    def _move_cb(self, msg):
        rospy.loginfo(f"Moving dx: {msg.dx} dy: {msg.dy} dz: {msg.dz} dpsi: {msg.dpsi}")
        self.drone(olympe_msgs.move.extended_move_by(
            msg.dx, msg.dy, msg.dz, msg.dpsi,
            msg.max_horizontal_speed, msg.max_vertical_speed,
            msg.max_yaw_rotation_speed)
        )