#!/usr/bin/env python3

import os
import sys
import yaml

import rospy
import geometry_msgs.msg
import drone_interface.msg
import numpy as np

from ekf import EKF, EKFState
from dynamic_models.cv_model import ConstantVelocityModel
from measurement_models import DnnCvPosition, DroneVelocity

class EKFRosRunner():

    def __init__(self):

        rospy.init_node("perception_ekf", anonymous=False)

        config_file = rospy.get_param("~config_file")
        script_dir = os.path.dirname(os.path.realpath(__file__))

        try:
            with open(f"{script_dir}/../../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        self.dnn_cv_sigmas = np.array(self.config["measurements"]["dnn_cv_position"]["sigmas"])
        self.attitude_and_velocity_sigmas = np.array(self.config["measurements"]["drone_velocity"]["sigmas"])

        self.dt = self.config["ekf"]["dt"]

        self.estimate_publisher = rospy.Publisher(
            self.config["ekf"]["output"]["topic_name"],
            geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=10
        )

        rospy.Subscriber(
            self.config["measurements"]["drone_velocity"]["topic_name"],
            drone_interface.msg.AnafiTelemetry, self._drone_velocity_cb
        )

        rospy.Subscriber(
            self.config["measurements"]["dnn_cv_position"]["topic_name"],
            geometry_msgs.msg.PointStamped, self._dnn_cv_position_cb
        )

        # Set up filter
        measurement_models = {
            "dnn_cv_position": DnnCvPosition(self.dnn_cv_sigmas),
            "drone_velocity": DroneVelocity(self.attitude_and_velocity_sigmas)
        }

        dynamic_model = ConstantVelocityModel()

        self.filter = EKF(dynamic_model, measurement_models)

        x0 = np.array(self.config["ekf"]["init_values"]["x0"])
        P0 = np.array(self.config["ekf"]["init_values"]["P0"])

        self.ekf_estimate = EKFState(x0, P0)

    def run(self):
        rospy.loginfo("Starting perception EKF")

        while not rospy.is_shutdown():

            self.ekf_estimate = self.filter.predict(self.ekf_estimate, None, self.dt)

            output_msg = self._pack_estimate_msg(self.ekf_estimate)
            self.estimate_publisher.publish(output_msg)

            rospy.sleep(self.dt)

    def _pack_estimate_msg(self, ekfstate: EKFState):
        msg = geometry_msgs.msg.PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = ekfstate.mean[0]
        msg.pose.pose.position.y = ekfstate.mean[1]
        msg.pose.pose.position.z = ekfstate.mean[2]
        msg.pose.covariance = ekfstate.cov.flatten().tolist()

        return msg

    def _dnn_cv_position_cb(self, msg: geometry_msgs.msg.PointStamped):
        z = np.array([msg.point.x, msg.point.y, msg.point.z])

        self.ekf_estimate = self.filter.update(z, self.ekf_estimate, "dnn_cv_position")

    def _drone_velocity_cb(self, msg: drone_interface.msg.AnafiTelemetry):
        z = np.array([msg.vx, msg.vy, msg.vz])

        self.ekf_estimate = self.filter.update(z, self.ekf_estimate, "drone_velocity")

def main():
    ekf_runner = EKFRosRunner()
    ekf_runner.run()

if __name__ == "__main__":
    main()