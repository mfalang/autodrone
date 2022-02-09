#!/usr/bin/env python3

import os
import sys
import yaml

import numpy as np
from scipy.spatial.transform import Rotation

import rospy
import geometry_msgs.msg
import ground_truth.msg

class CoordinateTransform():

    def __init__(self):

        rospy.init_node("coordinate_transform", anonymous=False)

        script_dir = os.path.dirname(os.path.realpath(__file__))

        config_file = rospy.get_param("~config_file")
        self.environment = rospy.get_param("~environment")

        try:
            with open(f"{script_dir}/../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        # Set up raw ground truth subscribers
        rospy.Subscriber(self.config["topics"]["drone_pose_ned"][self.environment],
            geometry_msgs.msg.PoseStamped, self._drone_gt_pose_cb
        )

        self.new_drone_gt_pose = False
        self.newest_drone_pose_msg: geometry_msgs.msg.PoseStamped = None

        rospy.Subscriber(self.config["topics"]["helipad_pose_ned"][self.environment],
            geometry_msgs.msg.PoseStamped, self._helipad_gt_pose_cb
        )

        self.new_helipad_gt_pose = False
        self.newest_helipad_pose_msg: geometry_msgs.msg.PoseStamped = None

        # Set up publisher for new drone pose relative to helipad
        self.drone_pose_helipad_frame_publisher = rospy.Publisher(
            self.config["topics"]["drone_pose_helipad_frame"][self.environment],
            ground_truth.msg.PoseStampedEuler, queue_size=1
        )


    def _drone_gt_pose_cb(self, msg: geometry_msgs.msg.PoseStamped):
        # Do the job here, and assume that the previously stored helipad position
        # is correct as the helipad should not have any high speed maneuvers
        if self.newest_helipad_pose_msg != None:
            # print(f"Difference: {(msg.header.stamp - self.newest_helipad_pose_msg.header.stamp).to_sec()*1000}")
            drone_pose_ned_frame = self._geometry_msg_pose_to_array(msg)
            helipad_pose_ned_frame = self._geometry_msg_pose_to_array(self.newest_helipad_pose_msg)
            drone_pose_helipad_frame = self._ned_to_helipad_frame(drone_pose_ned_frame, helipad_pose_ned_frame)
            self._publish_drone_helipad_frame_pose(drone_pose_helipad_frame)

    def _helipad_gt_pose_cb(self, msg: geometry_msgs.msg.PoseStamped):
        self.newest_helipad_pose_msg = msg

    def _ned_to_helipad_frame(self, drone_pose_ned: np.ndarray, helipad_pose_ned: np.ndarray):

        # Create rotation matrix between NED and Helipad frame
        helipad_rotation = helipad_pose_ned[5]
        R_ned_to_heli = np.array([
            [np.cos(helipad_rotation), -np.sin(helipad_rotation), 0],
            [np.sin(helipad_rotation), np.cos(helipad_rotation), 0],
            [0, 0, 1]
        ])

        # Translation between Helipad and NED frame (negative since this by definition
        # is the translation Helipad->NED expressed in the Helipad frame, but the
        # simulator and motion capture system gives the distance expressed in the
        # NED frame).
        t_heli_to_ned_in_heli = - R_ned_to_heli @ np.array([
            helipad_pose_ned[0],
            helipad_pose_ned[1],
            helipad_pose_ned[2]
        ])

        # Convert position
        drone_position_ned = np.array([
            drone_pose_ned[0],
            drone_pose_ned[1],
            drone_pose_ned[2]
        ])
        drone_position_helipad = R_ned_to_heli @ drone_position_ned + t_heli_to_ned_in_heli

        # Convert orientation
        drone_orientation_helipad = np.array([
            drone_pose_ned[0],
            drone_pose_ned[2],
            (drone_pose_ned[5] - helipad_rotation + 180) % 360 - 180 # Use smalles signed angle
        ])

        # Store result
        drone_pose_helipad = np.array([
            drone_position_helipad[0],
            drone_position_helipad[1],
            drone_position_helipad[2],
            drone_orientation_helipad[0],
            drone_orientation_helipad[1],
            drone_orientation_helipad[2],
        ])

        return drone_pose_helipad

    def _geometry_msg_pose_to_array(self, pose: geometry_msgs.msg.PoseStamped):

        quat = [pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        ]
        euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)

        ret = np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            euler[0],
            euler[1],
            euler[2]
        ])

        return ret

    def _publish_drone_helipad_frame_pose(self, pose: np.ndarray):
        msg = ground_truth.msg.PoseStampedEuler()
        msg.header.stamp = rospy.Time.now()
        msg.x = pose[0]
        msg.y = pose[1]
        msg.z = pose[2]
        msg.phi = pose[3]
        msg.theta = pose[4]
        msg.psi = pose[5]
        self.drone_pose_helipad_frame_publisher.publish(msg)

    def start(self):
        rospy.loginfo("Starting ground truth coordinate frame alignment")

        rospy.spin()

def main():
    coordinate_transformer = CoordinateTransform()
    coordinate_transformer.start()

if __name__ == "__main__":
    main()