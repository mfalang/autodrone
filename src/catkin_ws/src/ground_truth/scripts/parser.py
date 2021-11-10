#!/usr/bin/env python3

import os
import time
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

        # TODO: Maybe add this to some separate modules, as this exact same code
        # is used in the drone interface
        script_dir = os.path.dirname(os.path.realpath(__file__))
        today = time.localtime()
        self.ground_truth_dir = f"{script_dir}/../../../../../out" \
            f"/{today.tm_year}-{today.tm_mon}-{today.tm_mday}" \
            f"/{today.tm_hour}-{today.tm_min}-{today.tm_sec}/ground_truth"

        pathlib.Path(f"{self.ground_truth_dir}/drone").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{self.ground_truth_dir}/helipad").mkdir(parents=True, exist_ok=True)

        # Store values in RAM and then write results to file to avoid opening and closing files often.
        self.max_values_stored_in_ram = 100

        self.drone_poses = np.zeros((self.max_values_stored_in_ram,6))
        self.drone_timestamps = np.zeros((self.max_values_stored_in_ram,1))
        self.drone_data_index = 0

        self.helipad_poses = np.zeros((self.max_values_stored_in_ram,1))
        self.helipad_timestamps = np.zeros((self.max_values_stored_in_ram,1))
        self.helipad_data_index = 0

        self.drone_pose_filename = f"{self.ground_truth_dir}/drone/pose.txt"
        self.drone_timestamps_filename = f"{self.ground_truth_dir}/drone/timestamps.txt"

        self.drone_pose_file_desc = self._create_file_descriptor("drone/pose")
        self.drone_timestamps_file_desc = self._create_file_descriptor("drone/timestamps")

        self.helipad_pose_file_desc = self._create_file_descriptor("helipad/pose")
        self.helipad_timestamps_file_desc = self._create_file_descriptor("helipad/timestamps")

        rospy.Subscriber("ground_truth/pose/drone",
            geometry_msgs.msg.PoseStamped, self._drone_pose_cb
        )

        rospy.Subscriber("ground_truth/pose/helipad",
            geometry_msgs.msg.PoseStamped, self._helipad_pose_cb
        )

    def _create_file_descriptor(self, object_name):
        filename = f"{self.ground_truth_dir}/{object_name}.txt"
        file_desc = open(filename, "a+")
        return file_desc

    def _drone_pose_cb(self, msg):

        # Store in array if not full, save array if full
        if self.drone_data_index < self.max_values_stored_in_ram:
            self.drone_timestamps[self.drone_data_index] = msg.header.stamp.to_sec()
            self.drone_poses[self.drone_data_index][0] = msg.pose.position.x
            self.drone_poses[self.drone_data_index][1] = msg.pose.position.y
            self.drone_poses[self.drone_data_index][2] = msg.pose.position.z

            quat = [msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
            self.drone_poses[self.drone_data_index][3] = euler[0]
            self.drone_poses[self.drone_data_index][4] = euler[1]
            self.drone_poses[self.drone_data_index][5] = euler[2]

            self.drone_data_index += 1
        else:
            with open(self.drone_timestamps_filename, "a+") as file_desc:
                np.savetxt(file_desc, self.drone_timestamps)
            with open(self.drone_pose_filename, "a+") as file_desc:
                np.savetxt(file_desc, self.drone_poses)

            self.drone_data_index = 0

    def _helipad_pose_cb(self, msg):
        # print("Got helipad pose")
        pass

    def start(self):
        rospy.spin()
        rospy.on_shutdown(self._shutdown)

    def _shutdown(self):
        # self.drone_pose_file_desc.close()
        # self.helipad_pose_file_desc.close()
        rospy.loginfo(f"Ground truth data written to files in {self.ground_truth_dir}")

def main():
    parser = Parser()
    parser.start()

if __name__ == "__main__":
    main()
