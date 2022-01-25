#!/usr/bin/env python3

import os
import sys
import yaml

import bb_pose_estimate

import rospy
import perception.msg
import darknet_ros_msgs.msg

class DnnPoseEstimator():

    def __init__(self, config_file=None):

        rospy.init_node("dnn_cv_estimator", anonymous=False)

        if config_file is None:
            config_file = rospy.get_param("~config_file")

        script_dir = os.path.dirname(os.path.realpath(__file__))

        try:
            with open(f"{script_dir}/../../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        self.focal_length = rospy.get_param("/drone/camera/focal_length")
        self.image_width = rospy.get_param("/drone/camera/img_width")
        self.image_height = rospy.get_param("/drone/camera/img_height")
        self.camera_offset_x_mm = rospy.get_param("/drone/camera/offset_x_mm")
        self.camera_offset_z_mm = rospy.get_param("/drone/camera/offset_z_mm")

        self.latest_bounding_boxes = None
        self.new_bounding_boxes_available = False

        self.latest_orientation = None # Format: TODO

        self.bb_pose_estimator = bb_pose_estimate.BoundingBoxPoseEstimator(
            self.image_height, self.image_width, self.focal_length
        )

        rospy.Subscriber(self.config["topics"]["input"]["bounding_boxes"],
            darknet_ros_msgs.msg.BoundingBoxes, self._get_bounding_box_cb
        )

        self.pose_estimate_publisher = rospy.Publisher(
            self.config["topics"]["output"]["pose_estimate"],
            perception.msg.DnnCvPoseEstimate, queue_size=1
        )

    def start(self):
        rospy.loginfo("Starting Deep Neural Network Pose Estimator")

        rate = rospy.Rate(10) # Hz

        while not rospy.is_shutdown():

            # Only perform estimation if we have a valid bounding box different from the last one
            if self.new_bounding_boxes_available == False:
                rate.sleep()
                continue


            bounding_boxes = self.bb_pose_estimator.remove_bad_bounding_boxes(self.latest_bounding_boxes)

            pose_msg = perception.msg.DnnCvPoseEstimate()
            pose_msg.header.stamp = rospy.Time.now()

            # Position
            pos_ned = self.bb_pose_estimator.estimate_position_from_helipad_perimeter(bounding_boxes)

            if pos_ned is not None:
                pose_msg.position_available = True
                pose_msg.x_n = pos_ned[0]
                pose_msg.y_n = pos_ned[1]
                pose_msg.z_n = pos_ned[2]
            else:
                pose_msg.position_available = False

            # Heading
            heading = self.bb_pose_estimator.estimate_rotation_from_helipad_arrow(bounding_boxes)

            if heading is not None:
                pose_msg.heading_available = True
                pose_msg.psi = heading
            else:
                pose_msg.heading_available = False

            self.pose_estimate_publisher.publish(pose_msg)

            self.new_bounding_boxes_available = False
            rate.sleep()

    def _get_bounding_box_cb(self, msg):
        self.latest_bounding_boxes = msg
        self.new_bounding_boxes_available = True

def main():

    estimator = DnnPoseEstimator()
    estimator.start()

if __name__ == "__main__":
    main()
