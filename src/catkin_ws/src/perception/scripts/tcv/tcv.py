
import rospy
import sensor_msgs.msg

import os
import sys
import yaml
import cv2 as cv
import numpy as np

import corner_detector

class TcvPoseEstimator():

    def __init__(self, config_file=None):

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


        rospy.init_node("tcv_pose_estimator", anonymous=False)

        rospy.Subscriber("/drone/out/image_rect_color", sensor_msgs.msg.Image,
            self._new_image_cb
        )

        self.img_height = rospy.get_param("/drone/camera/img_height")
        self.img_width = rospy.get_param("/drone/camera/img_width")

        self.latest_image = np.zeros((self.img_height, self.img_width, 3))
        self.new_image_available = False

        self.corner_detector = corner_detector.CornerDetector(self.config["shi_tomasi"])


    def _new_image_cb(self, msg):
        self.latest_image = np.frombuffer(msg.data,
            dtype=np.uint8).reshape(msg.height, msg.width, -1
        )
        self.new_image_available = True


    def start(self):
        while not rospy.is_shutdown():

            if self.new_image_available:

                img = self.latest_image.astype(np.uint8)
                img_segmented = self.corner_detector.color_segment_image(img)

                corners = self.corner_detector.find_corners_shi_tomasi(img_segmented)

                if corners is None:
                    continue

                self.new_image_available = False

                self.corner_detector.show_corners_found(self.latest_image, corners)

                cv.waitKey(1)

def main():
    estimator = TcvPoseEstimator(config_file="tcv_config.yaml")
    estimator.start()

if __name__ == "__main__":
    main()