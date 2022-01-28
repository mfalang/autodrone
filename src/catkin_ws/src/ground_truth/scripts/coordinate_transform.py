#!/usr/bin/env python3

import os
import sys
import yaml
import rospy

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




    def start(self):
        print("Running")
        pass


def main():
    coordinate_transformer = CoordinateTransform()
    coordinate_transformer.start()

if __name__ == "__main__":
    main()