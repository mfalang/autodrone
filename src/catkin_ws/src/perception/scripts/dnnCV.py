#!/usr/bin/env python3

import rospy
import geometry_msgs.msg
import darknet_ros_msgs.msg
import drone_interface.msg

import math
import numpy as np
from scipy.spatial.transform import Rotation

class DNNPoseEstimator():

    def __init__(self, image_dims, camera_offsets):

        rospy.init_node("dnn_cv_estimator", anonymous=False)

        self.image_width = image_dims[0]
        self.image_height = image_dims[1]

        self.camera_offset_x = camera_offsets[0]
        self.camera_offset_z = camera_offsets[1]

        self.latest_bounding_boxes = None
        self.new_bounding_boxes_available = False

        self.latest_orientation = None # Format: [roll, deg, yaw] all in rad

        rospy.Subscriber("darknet_ros/bounding_boxes",
            darknet_ros_msgs.msg.BoundingBoxes, self._get_bounding_box_cb
        )

        rospy.Subscriber("drone/out/attitude_euler",
            drone_interface.msg.AttitudeEuler, self._get_attitude_cb
        )

        self.pose_estimate_publisher = rospy.Publisher(
            "estimate/drone_pose/dnnCV", geometry_msgs.msg.TwistStamped, queue_size=10
        )

    def start(self):
        rospy.loginfo("Starting Deep Neural Network Pose Estimator")

        rate = rospy.Rate(10) # Hz

        while not rospy.is_shutdown():

            # Only perform estimation if we have a valid bounding box different from the last one
            if self.latest_bounding_boxes is None \
                or self.new_bounding_boxes_available == False \
                or self.latest_orientation is None:
                rate.sleep()
                continue

            bounding_boxes = self.latest_bounding_boxes
            center_px, radius_px, rotation = self._estimate_center_rotation_and_radius(
                bounding_boxes.bounding_boxes
            )

            if all(center_px) and radius_px:
                pose_estimate = self._transform_pixel_position_to_world_coordinates(center_px, radius_px)
                pose_estimate.angular.z = rotation if rotation is not None else 0.0
                pose_estimate_stamped = geometry_msgs.msg.TwistStamped()
                pose_estimate_stamped.header.stamp = rospy.Time.now()
                pose_estimate_stamped.twist = pose_estimate
                self.pose_estimate_publisher.publish(pose_estimate_stamped)

            self.new_bounding_boxes_available = False
            rate.sleep()

    def _get_bounding_box_cb(self, msg):
        self.latest_bounding_boxes = msg
        self.new_bounding_boxes_available = True

    def _get_attitude_cb(self, msg):

        self.latest_orientation = [
            msg.roll*3.14159/180,
            msg.pitch*3.14159/180,
            msg.yaw*3.14159/180
        ]

    def _estimate_center_rotation_and_radius(self, bounding_boxes):
        """
        Function which estimates the center and radius of the helipad in the camera
        frame, using YOLO BoundingBox of predicted locations of Helipad and Arrow from the camera frame.
        Estimates radius as 0.97*sidelength of Helipad BoundingBox.
        Estimates center as center of Helipad BoundingBox
        Estimates yaw using _est_rotation() with center and center of Arrow BoundingBox.

        input:
            bounding_boxes: [n]bouding_box - list of yolo bouding_boxes
        output:
            center: [int, int] - estimated pixel coordinates of center of landing platform.
            radius: int - estimated pixel length of radius of landing platform
            rotation: float degrees - estimated yaw of quadcopter
        """
        # H_bb_radius_scale_factor = 2.60 # TODO: Check if needed
        # rospy.loginfo(bounding_boxes)
        bounding_boxes = [bb for bb in bounding_boxes if self._is_good_bb(bb)]
        # classes = list(item.Class for item in bounding_boxes) # TODO: Check if needed
        # rospy.loginfo(classes)
        center = [None, None]
        radius = None
        rotation = None

        Helipad = self._find_best_bb_of_class(bounding_boxes, 'Helipad')
        # H = self._find_best_bb_of_class(bounding_boxes, 'H') # TODO: Find out if H is used for estimation at all
        Arrow = self._find_best_bb_of_class(bounding_boxes, 'Arrow')

        # downscale_bb = 0.97 if not cfg.is_simulator else 1 # TODO: Check if needed
        if Helipad != None:
            center = self._est_center_of_bb(Helipad)
            radius = 0.97*self._est_radius_of_bb(Helipad)
            if Arrow != None:
                rotation = self._est_rotation(center, self._est_center_of_bb(Arrow))
        return center, radius, rotation


    def _is_good_bb(self, bb):
        """
        Returns true for bounding boxes that are within the desired proportions,
        them being relatively square.
        Bbs that have one side being 5 times or longer than the other are discarded.

        input:
            bb: yolo bounding box

        output.
            discard or not: bool
        """
        bb_w = bb.xmax - bb.xmin
        bb_h = bb.ymax - bb.ymin
        if 0.2 > bb_w/bb_h > 5:
            return False
        else:
            return True

    def _est_radius_of_bb(self, bb):
        width = bb.xmax - bb.xmin
        height = bb.ymax - bb.ymin
        radius = (width + height)/4
        return radius

    def _est_center_of_bb(self, bb):
        width = bb.xmax - bb.xmin
        height = bb.ymax - bb.ymin
        center = [bb.xmin + width/2.0 ,bb.ymin + height/2.0]
        map(int, center)
        return center

    def _find_best_bb_of_class(self, bounding_boxes, classname):
        matches =  list(item for item in bounding_boxes if item.Class == classname)
        try:
            best = max(matches, key=lambda x: x.probability)
        except ValueError as e:
            best = None
        return best

    def _est_rotation(self, center, Arrow):
        """
        Estimates the quadcopter yaw rotation given the estimated center of the Helipad
        as well as the estimated center of the arrow. Quadcopter rotation is defined
        with respect to world frame coordinate axis.
        yaw=0 is when arrow is pointing at three-o-clock, and positively increasing when
            arrow is rotating the same direction as clock arm movement.
        y is defined 0 in top of the image, and increases positively downwards.

        input:
            center: np.array[2] - [x,y] pixel coordinates
            Arrow: np.array[2] - [x,y] pixel coordinates

        output:
            degs: float - estimated yaw angle of the quadcopter
        """
        dy = center[1] - Arrow[1]
        dx = Arrow[0] - center[0]
        rads = math.atan2(dy,dx)
        degs = rads*180 / math.pi
        degs *= -1
        return degs

    def _transform_pixel_position_to_world_coordinates(self, center_px, radius_px):
        center_px = (center_px[1], center_px[0]) # such that x = height, y = width for this

        # These are from ArDrone so probably wrong
        focal_length = 374.67 # if cfg.is_simulator else 720
        real_radius = 390 # mm (780mm in diameter / 2)

        # Center of image
        x_0 = self.image_height/2.0
        y_0 = self.image_width/2.0

        # Find distances from center of image to center of LP
        d_x = x_0 - center_px[0]
        d_y = y_0 - center_px[1]

        est_z = real_radius*focal_length / radius_px

        # Camera is placed 150 mm along x-axis of the drone
        # Since the camera is pointing down, the x and y axis of the drone
        # is the inverse of the x and y axis of the camera
        est_x = -((est_z * d_x / focal_length) + self.camera_offset_x) # mm adjustment for translated camera frame in x direction
        est_y = -(est_z * d_y / focal_length)
        est_z += self.camera_offset_z # mm adjustment for translated camera frame in z direction

        # Compensation for angled camera.

        cr, cp = math.cos(self.latest_orientation[0]), math.cos(self.latest_orientation[1])
        sr, sp = math.sin(self.latest_orientation[0]), math.sin(self.latest_orientation[1])

        x = est_x + est_z*sp*(0.5) # if not cfg.is_simulator else 1)
        y = est_y - est_z*sr*(0.7) # if not cfg.is_simulator else 1)
        z = est_z * cr * cp
        x = est_x
        y = est_y
        z = est_z

        est = geometry_msgs.msg.Twist()
        est.linear.x = x / 1000.0
        est.linear.y = y / 1000.0
        est.linear.z = z / 1000.0
        return est


    def _calculate_estimation_error(self, est, gt):
        if gt == None:
            return None
        est = geometry_msgs.msg.Twist() if est is None else est
        error = geometry_msgs.msg.Twist()
        error.linear.x = est.linear.x - gt.linear.x
        error.linear.y = est.linear.y - gt.linear.y
        error.linear.z = est.linear.z - gt.linear.z
        error.angular.x = est.angular.x - gt.angular.x
        error.angular.y = est.angular.y - gt.angular.y
        error.angular.z = est.angular.z - gt.angular.z
        return error

def main():

    image_dims = [1280, 720]
    camera_offsets = [-60, 45] # mm (taken from ArDrone so probably wrong)

    estimator = DNNPoseEstimator(image_dims, camera_offsets)
    estimator.start()

if __name__ == "__main__":
    main()
