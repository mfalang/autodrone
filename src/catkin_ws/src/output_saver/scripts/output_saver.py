#!/usr/bin/env python3

import os
import yaml
import sys
import time
import shutil

import rospy

import ground_truth
import pose_estimate

class OutputSaver():

    def __init__(self):
        
        rospy.init_node("output_saver", anonymous=False)

        script_dir = os.path.dirname(os.path.realpath(__file__))

        config_file = rospy.get_param("~config_file")
        self.environment = rospy.get_param("~environment")

        try:
            with open(f"{script_dir}/../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        today = time.localtime()
        self.output_base_dir = f"{script_dir}/../../../../../out" \
            f"/{today.tm_year}-{today.tm_mon}-{today.tm_mday}" \
            f"/{today.tm_hour}-{today.tm_min}-{today.tm_sec}/"

    def start(self):
        rospy.loginfo(f"Saving output to {self.output_base_dir}")

        pose_estimate.DNNCVDataSaver(self.config, self.output_base_dir, 
            "pose_estimates", "dnn_cv_pose", self.environment
        )

        ground_truth.DronePoseDataSaver(self.config, self.output_base_dir,
            "ground_truths", "drone_pose", self.environment
        )
        ground_truth.HelipadPoseDataSaver(self.config, self.output_base_dir,
            "ground_truths", "helipad_pose", self.environment
        )

        rospy.on_shutdown(self._on_shutdown)
        rospy.spin()

    def _on_shutdown(self):
        ans = input("Save output? [y/n] ")
        if ans.lower() == "n" or ans.lower() == "no":
            rospy.loginfo(f"Deleting directory {self.output_base_dir}")
            shutil.rmtree(self.output_base_dir)
        else:
            rospy.loginfo(f"Saved estimates to {self.output_base_dir}")


def main():
    output_saver = OutputSaver()
    output_saver.start()

if __name__ == "__main__":
    main()