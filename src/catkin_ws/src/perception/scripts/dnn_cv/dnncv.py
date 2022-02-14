#!/usr/bin/env python3

import os
import sys
import yaml

import bb_pose_estimate

import rospy
import perception.msg
import geometry_msgs.msg
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
        camera_offset_x_mm = rospy.get_param("/drone/camera/offset_x_mm")
        camera_offset_y_mm = rospy.get_param("/drone/camera/offset_y_mm")
        camera_offset_z_mm = rospy.get_param("/drone/camera/offset_z_mm")
        self.camera_offsets = [camera_offset_x_mm, camera_offset_y_mm, camera_offset_z_mm]

        self.latest_bounding_boxes = None
        self.new_bounding_boxes_available = False

        self.bb_pose_estimator = bb_pose_estimate.BoundingBoxPoseEstimator(
            self.image_height, self.image_width, self.focal_length, self.camera_offsets
        )

        rospy.Subscriber(self.config["topics"]["input"]["bounding_boxes"],
            darknet_ros_msgs.msg.BoundingBoxes, self._get_bounding_box_cb
        )

        self.position_estimate_publisher = rospy.Publisher(
            self.config["topics"]["output"]["position"],
            geometry_msgs.msg.PointStamped, queue_size=1
        )

        self.heading_estimate_publisher = rospy.Publisher(
            self.config["topics"]["output"]["heading"],
            perception.msg.Heading, queue_size=1
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

            # Position
            pos_helipad_frame = self.bb_pose_estimator.estimate_position_from_helipad_perimeter(bounding_boxes)

            if pos_helipad_frame is not None:
                position_msg = geometry_msgs.msg.PointStamped()
                position_msg.header.stamp = rospy.Time.now()
                position_msg.header.frame_id = "helipad"
                position_msg.point.x = pos_helipad_frame[0]
                position_msg.point.y = pos_helipad_frame[1]
                position_msg.point.z = pos_helipad_frame[2]
                self.position_estimate_publisher.publish(position_msg)

            # Heading
            heading = self.bb_pose_estimator.estimate_rotation_from_helipad_arrow(bounding_boxes)

            if heading is not None:
                heading_msg = perception.msg.Heading()
                heading_msg.header.stamp = rospy.Time.now()
                heading_msg.header.frame_id = "helipad"
                heading_msg.heading = heading
                self.heading_estimate_publisher.publish(heading_msg)

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
