#!/usr/bin/env python3

import rospy
import sensor_msgs.msg

import os
import sys
import yaml
import time
import cv2 as cv
import numpy as np

import feature_detector
import pose_recovery

import perception.msg

class TcvPoseEstimator():

    def __init__(self, config_file=None):

        rospy.init_node("tcv_pose_estimator", anonymous=False)

        # In order to be able to run program without ROS launch so that it can
        # be run in vscode debugger
        if config_file is None:
            config_file = rospy.get_param("~config_file")

        script_dir = os.path.dirname(os.path.realpath(__file__))

        try:
            with open(f"{script_dir}/../../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        self.view_camera_output = rospy.get_param("~view_camera_output")
        self.calculate_run_times = rospy.get_param("~calculate_run_times")

        rospy.Subscriber("/drone/out/image_rect_color", sensor_msgs.msg.Image,
            self._new_image_cb
        )

        self.env = rospy.get_param("~environment")
        self.img_height = rospy.get_param("/drone/camera/img_height")
        self.img_width = rospy.get_param("/drone/camera/img_width")
        self.camera_offsets = np.array([
            rospy.get_param("/drone/camera/offset_x_mm")/1000,
            rospy.get_param("/drone/camera/offset_y_mm")/1000,
            rospy.get_param("/drone/camera/offset_z_mm")/1000
        ])
        self.K = np.array(rospy.get_param("/drone/camera/camera_matrix")).reshape(3,3)
        self.focal_length = (self.K[0,0] + self.K[1,1])/2

        self.feature_dists_metric = np.loadtxt(
            f"{script_dir}/../../{self.config['feature_dists_metric']['path']}"
        )

        self.latest_image = np.zeros((self.img_height, self.img_width, 3))
        self.new_image_available = False

        with open(f"{script_dir}/../../{self.config['shi_tomasi_params']['path']}") as f:
            shi_tomasi_params = yaml.safe_load(f)

        with open(f"{script_dir}/../../{self.config['hough_circle_params']['path']}") as f:
            hough_circle_params = yaml.safe_load(f)

        self.corner_detector = feature_detector.FeatureDetector(shi_tomasi_params, hough_circle_params)
        self.pose_recoverer = pose_recovery.PoseRecovery(self.K, self.camera_offsets)

        self.pose_estimate_publisher = rospy.Publisher(
            "/estimate/drone_pose/tcv", perception.msg.EulerPose, queue_size=10
        )


    def _new_image_cb(self, msg):
        self.latest_image = np.frombuffer(msg.data,
            dtype=np.uint8).reshape(msg.height, msg.width, -1
        )
        self.new_image_available = True


    def start(self):

        rospy.loginfo("Starting TCV pose estimator")

        if self.env == "sim":
            segment = False
        else:
            segment = True

        while not rospy.is_shutdown():

            if self.new_image_available:

                img = self.latest_image.astype(np.uint8)

                if self.calculate_run_times:
                    start_time = time.time()

                mask = self.corner_detector.create_helipad_mask(img)

                if self.calculate_run_times:
                    circle_detection_duration = time.time() - start_time
                    print(f"Used {circle_detection_duration:.4f} sec to detect circle")

                if self.calculate_run_times:
                    start_time = time.time()

                corners = self.corner_detector.find_corners_shi_tomasi(img, mask)

                if self.calculate_run_times:
                    corner_detection_duration = time.time() - start_time
                    print(f"Used {corner_detection_duration:.4f} sec to detect corners")

                if self.view_camera_output:
                    self.corner_detector.show_corners_found(img, corners, color="red")
                    cv.waitKey(1)

                if corners is None:
                    continue

                if self.calculate_run_times:
                    start_time = time.time()
                features_image = self.corner_detector.find_arrow_and_H(corners, self.feature_dists_metric)
                if self.calculate_run_times:
                    corner_identification_duration = time.time() - start_time
                    print(f"Used {corner_identification_duration:.4f} sec to identify corners")

                if features_image is None:
                    continue

                if self.view_camera_output:
                    self.corner_detector.show_known_points(img, features_image)

                if self.calculate_run_times:
                    start_time = time.time()
                R_cam, t_cam = self.pose_recoverer.find_R_t_pnp(self.feature_dists_metric, features_image)
                R_body, t_body = self.pose_recoverer.camera_to_drone_body_frame(R_cam, t_cam)
                pose_body = self.pose_recoverer.get_pose_from_R_t(R_body, t_body)
                if self.calculate_run_times:
                    pose_calculation_duration = time.time() - start_time
                    print(f"Used {pose_calculation_duration:.4f} sec to calculate pose")

                if self.calculate_run_times:
                    print(f"Total: {circle_detection_duration + corner_detection_duration + corner_identification_duration + pose_calculation_duration:.4f} sec")
                    print()

                self._publish_pose(pose_body)

                self.new_image_available = False

                cv.waitKey(1)

    def _pose_camera_to_ned(self, pose_camera):
        # Convert pose from camera frame with ENU coordinates, to NED coordinates

        pose_ned = np.zeros_like(pose_camera)
        pose_ned[0] = pose_camera[1]
        pose_ned[1] = pose_camera[0]
        pose_ned[2] = -pose_camera[2]
        pose_ned[3] = pose_camera[3]
        pose_ned[4] = pose_camera[4]
        pose_ned[5] = pose_camera[5]

        pose_ned[0] -= self.camera_offsets[0]
        pose_ned[1] -= self.camera_offsets[1]
        pose_ned[2] -= self.camera_offsets[2]

        return pose_ned


    def _publish_pose(self, pose):
        msg = perception.msg.EulerPose()
        msg.header.stamp = rospy.Time.now()

        msg.x = pose[0]
        msg.y = pose[1]
        msg.z = pose[2]
        msg.phi = pose[3]
        msg.theta = pose[4]
        msg.psi = pose[5]

        self.pose_estimate_publisher.publish(msg)


def main():
    estimator = TcvPoseEstimator(config_file="tcv_config.yaml")
    estimator.start()

if __name__ == "__main__":
    main()