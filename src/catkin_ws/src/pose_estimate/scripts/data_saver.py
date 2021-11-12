#!/usr/bin/env python3

import os
import sys
import time
import yaml
import pathlib

import rospy
import geometry_msgs.msg

import numpy as np

class EstimateDataSaver():

    def __init__(self):
        """
        Saves estimated data coming from the different methods of estimation
        implemented. Currently saves data from
            - Deep Neural Network Estimator (dnnCV)
        """

        rospy.init_node("pose_estimate_data_saver", anonymous=False)

        script_dir = os.path.dirname(os.path.realpath(__file__))

        config_file = rospy.get_param("~config_file")

        try:
            with open(f"{script_dir}/../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        # TODO: Maybe add this to some separate modules, as this exact same code
        # is used in the drone interface
        today = time.localtime()
        self.estimate_dir = f"{script_dir}/../../../../../out" \
            f"/{today.tm_year}-{today.tm_mon}-{today.tm_mday}" \
            f"/{today.tm_hour}-{today.tm_min}-{today.tm_sec}/estimate"
        pathlib.Path(f"{self.estimate_dir}/drone").mkdir(parents=True, exist_ok=True)

        # Store values in RAM and then write results to file to avoid opening and closing files often
        self.max_values_stored_in_ram = 100

        self.dnn_cv_estimates = np.zeros((self.max_values_stored_in_ram, 4))
        self.dnn_cv_timestamps = np.zeros((self.max_values_stored_in_ram, 1))
        self.dnn_cv_data_index = 0

        self.dnn_cv_estimates_filename = f"{self.estimate_dir}/drone/estimates.txt"
        self.dnn_cv_timestamps_filename = f"{self.estimate_dir}/drone/timestamps.txt"

        self._write_format_header_to_file(self.dnn_cv_estimates_filename,
            self.dnn_cv_timestamps_filename, "dnn_cv"
        )

        rospy.Subscriber(self.config["subscribed_topics"]["dnnCV_estimator"],
            geometry_msgs.msg.Twist, self._dnn_cv_pose_cb
        )

    def start(self):
        rospy.loginfo(f"Saving output from estimates to {self.estimate_dir}")
        rospy.spin()

    def _write_format_header_to_file(self, estimates_filename, timestamps_filename, type):
        if type == "dnn_cv":
            estimates_header = "Format: [x[m], y[m], z[m], phi[deg], theta[deg], psi[deg]]\n"

        with open(estimates_filename, "w+") as file_desc:
            file_desc.write(estimates_header)

        timestamps_header = "Format: Timestamp in seconds\n"
        with open(timestamps_filename, "w+") as file_desc:
            file_desc.write(timestamps_header)


    def _dnn_cv_pose_cb(self, msg):

        # Write buffer array to file if full
        if self.dnn_cv_data_index >= self.max_values_stored_in_ram:
            with open(self.dnn_cv_timestamps_filename, "a") as file_desc:
                np.savetxt(file_desc, self.dnn_cv_timestamps)
            with open(self.dnn_cv_estimates_filename, "a") as file_desc:
                np.savetxt(file_desc, self.dnn_cv_estimates)

            self.dnn_cv_data_index = 0

        self.dnn_cv_timestamps[self.dnn_cv_data_index] = msg.header.stamp.to_sec()
        self.dnn_cv_estimates[self.dnn_cv_data_index] = [
            msg.Twist.linear.x,
            msg.Twist.linear.y,
            msg.Twist.linear.z,
            msg.Twist.angular.z
        ]

        self.dnn_cv_data_index += 1

def main():
    data_saver = EstimateDataSaver()
    data_saver.start()

if __name__ == "__main__":
    main()
