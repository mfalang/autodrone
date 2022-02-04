#!/usr/bin/env python3

import os
import sys
import yaml

import numpy as np
from scipy.spatial.transform import Rotation

import rospy
import geometry_msgs.msg

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
            geometry_msgs.msg.PoseStamped, queue_size=1
        )


    def _drone_gt_pose_cb(self, msg: geometry_msgs.msg.PoseStamped):
        self.newest_drone_pose_msg = msg
        self.new_drone_gt_pose = True

    def _helipad_gt_pose_cb(self, msg: geometry_msgs.msg.PoseStamped):
        self.newest_helipad_pose_msg = msg
        self.new_helipad_gt_pose = True

    def _ned_to_helipad_frame(self, drone_pose_ned: np.ndarray, helipad_pose_ned: np.ndarray):

        helipad_rotation = helipad_pose_ned[5]

        R_ned_to_heli = np.array([
            [np.cos(helipad_rotation), np.sin(helipad_rotation), 0],
            [-np.sin(helipad_rotation), np.cos(helipad_rotation), 0],
            [0, 0, 1]
        ])

        t_heli_to_ned = -np.array([
            helipad_pose_ned[0],
            helipad_pose_ned[1],
            helipad_pose_ned[2]
        ])


        drone_position_ned = np.array([
            drone_pose_ned[0],
            drone_pose_ned[1],
            drone_pose_ned[2]
        ])

        drone_position_helipad = R_ned_to_heli @ drone_position_ned + t_heli_to_ned

        drone_pose_helipad = np.array([
            drone_position_helipad[0],
            drone_position_helipad[0],
            drone_position_helipad[0],
            drone_pose_ned[3],
            drone_pose_ned[4],
            drone_pose_ned[5],
        ])

        return drone_pose_helipad

    def _geometry_msg_pose_to_array(self, pose: geometry_msgs.msg.PoseStamped()):

        quat = [pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)

        ret = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            euler[0],
            euler[1],
            euler[2]
        ])

        return ret

    def _publish_drone_helipad_frame_pose(self, pose: geometry_msgs.msg.PoseStamped().pose):
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose = pose
        self.drone_pose_helipad_frame_publisher.publish(msg)

    def start(self):
        rospy.loginfo("Starting ground truth coordinate frame alignment")

        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            if not (self.new_drone_gt_pose and self.new_helipad_gt_pose):
                continue

            drone_pose_helipad_frame = self._ned_to_helipad_frame(self.newest_drone_pose_msg.pose)
            self._publish_drone_helipad_frame_pose(drone_pose_helipad_frame)

            print(f"Difference: {(self.newest_drone_pose_msg.header.stamp - self.newest_helipad_pose_msg.header.stamp).to_sec()*1000}")



            self.new_drone_gt_pose = False
            self.new_helipad_gt_pose = False

            rate.sleep()

def main():
    coordinate_transformer = CoordinateTransform()
    coordinate_transformer.start()

if __name__ == "__main__":
    main()