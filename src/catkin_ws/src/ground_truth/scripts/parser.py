#!/usr/bin/env python3

import os
import sys
import time
import yaml
import pathlib

import rospy
import geometry_msgs.msg

import numpy as np
from scipy.spatial.transform import Rotation


class Parser():

    def __init__(self):
        """
        Parses ground truth data coming from either the motion capture system or
        the simulator and saves it as numpy files. Can also visualize the data
        live.
        """

        rospy.init_node("ground_truth_parser", anonymous=False)

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
        self.ground_truth_dir = f"{script_dir}/../../../../../out" \
            f"/{today.tm_year}-{today.tm_mon}-{today.tm_mday}" \
            f"/{today.tm_hour}-{today.tm_min}-{today.tm_sec}/ground_truth"
        pathlib.Path(f"{self.ground_truth_dir}/drone").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{self.ground_truth_dir}/helipad").mkdir(parents=True, exist_ok=True)

        # Store values in RAM and then write results to file to avoid opening and closing files often
        self.max_values_stored_in_ram = 100

        self.drone_poses = np.zeros((self.max_values_stored_in_ram, 6))
        self.drone_timestamps = np.zeros((self.max_values_stored_in_ram, 1))
        self.drone_data_index = 0

        self.helipad_poses = np.zeros((self.max_values_stored_in_ram, 6))
        self.helipad_timestamps = np.zeros((self.max_values_stored_in_ram, 1))
        self.helipad_data_index = 0

        self.drone_pose_filename = f"{self.ground_truth_dir}/drone/pose.txt"
        self.drone_timestamps_filename = f"{self.ground_truth_dir}/drone/timestamps.txt"

        self.helipad_pose_filename = f"{self.ground_truth_dir}/helipad/pose.txt"
        self.helipad_timestamps_filename = f"{self.ground_truth_dir}/helipad/timestamps.txt"

        self._write_format_header_to_file(self.drone_pose_filename, "pose")
        self._write_format_header_to_file(self.drone_timestamps_filename, "timestamp")

        self._write_format_header_to_file(self.helipad_pose_filename, "pose")
        self._write_format_header_to_file(self.helipad_timestamps_filename, "timestamp")


        rospy.Subscriber(self.config["drone"]["pose_topic"],
            geometry_msgs.msg.PoseStamped, self._drone_pose_cb
        )

        rospy.Subscriber(self.config["helipad"]["pose_topic"],
            geometry_msgs.msg.PoseStamped, self._helipad_pose_cb
        )

    def start(self):
        rospy.loginfo(f"Parsing output from ground truth and saving to {self.ground_truth_dir}")
        rospy.spin()

    def _write_format_header_to_file(self, filename, type):
        if type == "pose":
            header = "Format: [x[m], y[m], z[m], phi[deg], theta[deg], psi[deg]]\n"
        elif type == "timestamp":
            header = "Format: Timestamp in seconds\n"

        with open(filename, "w+") as file_desc:
            file_desc.write(header)


    def _drone_pose_cb(self, msg):
        # Store in array if not full, save array if full
        if self.drone_data_index < self.max_values_stored_in_ram:
            self.drone_timestamps[self.drone_data_index] = msg.header.stamp.to_sec()
            self.drone_poses[self.drone_data_index] = self._get_pose_from_geometry_msg(msg)

            self.drone_data_index += 1
        else:
            with open(self.drone_timestamps_filename, "a") as file_desc:
                np.savetxt(file_desc, self.drone_timestamps)
            with open(self.drone_pose_filename, "a") as file_desc:
                np.savetxt(file_desc, self.drone_poses)

            self.drone_data_index = 0

    def _helipad_pose_cb(self, msg):
        # Store in array if not full, save array if full
        if self.helipad_data_index < self.max_values_stored_in_ram:
            self.helipad_timestamps[self.helipad_data_index] = msg.header.stamp.to_sec()
            self.helipad_poses[self.helipad_data_index] = self._get_pose_from_geometry_msg(msg)

            self.helipad_data_index += 1
        else:
            with open(self.helipad_timestamps_filename, "a") as file_desc:
                np.savetxt(file_desc, self.helipad_timestamps)
            with open(self.helipad_pose_filename, "a") as file_desc:
                np.savetxt(file_desc, self.helipad_poses)

            self.helipad_data_index = 0

    def _get_pose_from_geometry_msg(self, msg):

        res = []
        res.append(msg.pose.position.x)
        res.append(msg.pose.position.y)
        res.append(msg.pose.position.z)

        quat = [msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
        res.append(euler[0])
        res.append(euler[1])
        res.append(euler[2])

        return res

def main():
    parser = Parser()
    parser.start()

if __name__ == "__main__":
    main()
