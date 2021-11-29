
import rospy
import geometry_msgs.msg

import numpy as np
from scipy.spatial.transform import Rotation

from generic_output_saver import GenericOutputSaver

class GroundTruthDataSaver(GenericOutputSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        self.initialized_offsets = False
        self.offsets = None # Format: numpy array [0,x,y,z,roll,pitch,yaw] (0 for timestamp)

    def _initialize_offsets(self, output_raw, object_type):
        self.offsets = output_raw
        self.offsets[0] = 0 # No offset in timestamp
        self.offsets[4:] = 0
        print(f"Offsets ({object_type}): " \
                f"x: {self.offsets[1]:.3f}m y: {self.offsets[2]:.3f}m " \
                f"z: {self.offsets[3]:.3f}m roll: {self.offsets[4]:.3f}deg " \
                f"pitch: {self.offsets[5]:.3f}deg yaw: {self.offsets[6]:.3f}deg"
        )
        self.initialized_offsets = True

    def _print_output(self, output, object_type):
        # Used for setup of motion capture system
        x, y, z, roll, pitch, yaw = output[1:]
        print(f"Pose ({object_type}):\tx: {x:.3f} y: {y:.3f} z: {z:.3f}\tRoll: {roll:.3f} Pitch: {pitch:.3f} Yaw: {yaw:.3f}")

    def _get_output_from_geometry_msg(self, msg):
        """
        Modifications made so that coordinate system is aligned with the on used
        by the pose estimate. Here:
        Motion capture:
            - x-axis of mocap is negative y-axis of pose estimate
            - y-axis of mocap is x-axis of pose estimate
        Simulator:
            - All axis are the same
        """
        # TODO: Make this work for simulator also
        quat = [msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]

        euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)

        if self.environment == "real":
            res = np.array([
                msg.header.stamp.to_sec(),
                -msg.pose.position.y, # conversion between frames
                msg.pose.position.x, # conversion between frames
                -msg.pose.position.z,
                euler[0],
                euler[1],
                euler[2]
            ])
        else:
            res = np.array([
                msg.header.stamp.to_sec(),
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                euler[0],
                euler[1],
                euler[2]
            ])

        return res

class DronePoseDataSaver(GroundTruthDataSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, geometry_msgs.msg.PoseStamped, self._drone_gt_pose_cb)


    def _drone_gt_pose_cb(self, msg):

        output_raw_ned = self._get_output_from_geometry_msg(msg)

        # Convert postion estimate to body frame
        # R_ned_to_body = np.array([
        #     [np.cos(output_raw_ned[6]), np.sin(output_raw_ned[6]), 0],
        #     [-np.sin(output_raw_ned[6]), np.cos(output_raw_ned[6]), 0],
        #     [0, 0, 1]
        # ])
        # pos_ned = np.array([output_raw_ned[1], output_raw_ned[2], output_raw_ned[3]])

        # x_body, y_body, z_body = R_ned_to_body @ pos_ned
        # output_raw = np.array([
        #     output_raw_ned[0],
        #     x_body,
        #     y_body,
        #     z_body,
        #     output_raw_ned[4],
        #     output_raw_ned[5],
        #     output_raw_ned[6],
        # ])
        output_raw = output_raw_ned
        # self._print_output(output_raw, "drone")

        if self.initialized_offsets == False:
            self._initialize_offsets(output_raw, "drone")

        output = output_raw - self.offsets
        print(output[6])
        # self._print_output(output, "drone")

        self._save_output(output)

class HelipadPoseDataSaver(GroundTruthDataSaver):

    def __init__(self, config, base_dir, output_category, output_type, environment):
        super().__init__(config, base_dir, output_category, output_type, environment)

        rospy.Subscriber(self.topic_name, geometry_msgs.msg.PoseStamped, self._helipad_gt_pose_cb)

    def _helipad_gt_pose_cb(self, msg):

        output_raw = self._get_output_from_geometry_msg(msg)

        # self._print_output(output_raw, "helipad")

        if self.initialized_offsets == False:
            self._initialize_offsets(output_raw, "helipad")

        output = output_raw - self.offsets

        # self._print_output(output, "helipad")

        self._save_output(output)
