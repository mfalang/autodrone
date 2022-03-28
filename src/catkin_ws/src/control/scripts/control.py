# Main file starting the control algorithms

import os
import sys
import yaml

import numpy as np

import rospy
import std_msgs.msg
import drone_interface.msg

import control_util
import reference_model
import attitude_reference_generator

class Controller():

    def __init__(self, params: dict):

        self.config = params

        # Set up attitude reference generator
        self._attitude_controller = rospy.get_param("~attitude_controller")
        rospy.loginfo(f"Controller started with control method: {self._attitude_controller}")
        att_controller_params = self.config[self._attitude_controller]
        att_limits = self.config["attitude_limits"]

        attitude_generator = attitude_reference_generator.get_attitude_reference_generator(self._attitude_controller)
        self._attitude_reference_generator = attitude_generator(att_controller_params, att_limits)

        # Set up topic publishers
        self._attitude_ref_publisher = rospy.Publisher(
            rospy.get_param("/drone/topics/input/attitude_setpoint"),
            drone_interface.msg.AttitudeSetpoint, queue_size=2
        )

        self._takeoff_publisher = rospy.Publisher(
            rospy.get_param("/drone/topics/input/takeoff"),
            std_msgs.msg.Empty, queue_size=1
        )

        self._land_publisher = rospy.Publisher(
            rospy.get_param("/drone/topics/input/land"),
            std_msgs.msg.Empty, queue_size=1
        )

        self._reference_model = reference_model.VelocityReferenceModel(
            self.config["reference_model"]["omegas"], self.config["reference_model"]["zetas"]
        )

    def takeoff(self):
        control_util.await_user_confirmation("takeoff")
        rospy.loginfo("Taking off")
        self._takeoff_publisher.publish(std_msgs.msg.Empty())

    def land(self):
        control_util.await_user_confirmation("land")
        rospy.loginfo("Landing")
        self._land_publisher.publish(std_msgs.msg.Empty())

    def get_reference(self, ref_prev: np.ndarray, ref_raw: np.ndarray, dt: float):
        return self._reference_model.get_filtered_reference(ref_prev, ref_raw, dt)

    def set_attitude(self, x_d: np.ndarray, x: np.ndarray, ts: float, debug=False):
        att_ref = self._attitude_reference_generator.get_attitude_reference(x_d, x, ts, debug=debug)

        att_ref_msg = control_util.pack_attitude_ref_msg_horizontal(att_ref)
        self._attitude_ref_publisher.publish(att_ref_msg)

        return att_ref

def main():
    controller = Controller()

if __name__ == "__main__":
    main()