#!/usr/bin/env python3

import ekf

import os
import sys
import yaml

import rospy
import perception.msg
import geometry_msgs.msg
import drone_interface.msg
import numpy as np

from ekf.ekf import EKF, EKFState
from ekf.dynamic_models.helipad_3D_movement import DynamicModel
from ekf.measurement_models.dnn_cv import DnnCvModel

class EKFRosRunner():

    def __init__(self):

        rospy.init_node("perception_ekf", anonymous=False)

        config_file = rospy.get_param("~config_file")
        script_dir = os.path.dirname(os.path.realpath(__file__))

        try:
            with open(f"{script_dir}/../config/{config_file}") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            sys.exit()

        self.z = np.zeros((4))
        self.new_messurement = False

        self.dnn_cv_sigmas = np.array(self.config["measurements"]["dnn_cv"]["sigmas"])

        self.dt = self.config["ekf"]["dt"]


        self.estimate_publisher = rospy.Publisher(
            self.config["ekf"]["output"]["topic_name"],
            perception.msg.EkfOutput, queue_size=10
        )

        self.latest_input = np.zeros(4)
        self.last_psi = None

        rospy.Subscriber(self.config["ekf"]["input"]["topic_name"],
            drone_interface.msg.EkfInput, self._input_cb
        )


    def run(self):
        rospy.loginfo("Starting perception EKF")

        filter = EKF(DynamicModel(), DnnCvModel(self.dnn_cv_sigmas))

        x0 = np.array(self.config["ekf"]["init_values"]["x0"])
        P0 = np.array(self.config["ekf"]["init_values"]["P0"])

        ekf_estimate = EKFState(x0, P0)

        while not rospy.is_shutdown():

            ekf_estimate = filter.predict(ekf_estimate, self.latest_input, self.dt)
            self.new_velocity_input = False
            self.new_attitude_input = False

            if self.new_messurement:
                ekf_estimate = filter.update(self.z, ekf_estimate)
                self.new_messurement = False

            output_msg = self._pack_estimate_msg(ekf_estimate)
            self.estimate_publisher.publish(output_msg)

            rospy.sleep(self.dt)

    def _pack_estimate_msg(self, ekfstate):
        msg = perception.msg.EkfOutput()
        msg.header.stamp = rospy.Time.now()
        msg.x = ekfstate.mean[0]
        msg.y = ekfstate.mean[1]
        msg.z = ekfstate.mean[2]
        msg.psi = ekfstate.mean[3]
        msg.v_x = ekfstate.mean[4]
        msg.v_y = ekfstate.mean[5]
        msg.v_z = ekfstate.mean[6]
        msg.covariance = ekfstate.cov.flatten().tolist()

        return msg

    def _input_cb(self, msg):
        self.latest_input[0] = msg.v_x
        self.latest_input[1] = msg.v_y
        self.latest_input[2] = msg.v_z

        if self.last_psi is not None:
            self.latest_input[3] = msg.psi - self.last_psi

        self.last_psi = msg.psi

        if self.latest_input[3] != 0:
            print(f"dpsi: {self.latest_input[3]}")


    def _dnn_cv_estimate_cb(self, msg):
        self.z = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.z
        ])
        self.new_messurement = True

def main():
    ekf_runner = EKFRosRunner()
    ekf_runner.run()

if __name__ == "__main__":
    main()