#!/usr/bin/env python3

import os
import sys
import yaml

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

    def _helipad_to_ned_frame(self, pose: geometry_msgs.msg.PoseStamped().pose):

        return pose

    def _publish_drone_helipad_frame_pose(self):
        pass


    def start(self):
        rospy.loginfo("Starting ground truth coordinate frame alignment")

        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            if not (self.new_drone_gt_pose and self.new_helipad_gt_pose):
                continue



            print(f"Difference: {(self.newest_drone_pose_msg.header.stamp - self.newest_helipad_pose_msg.header.stamp).to_sec()*1000}")



            self.new_drone_gt_pose = False
            self.new_helipad_gt_pose = False

            rate.sleep()

def main():
    coordinate_transformer = CoordinateTransform()
    coordinate_transformer.start()

if __name__ == "__main__":
    main()