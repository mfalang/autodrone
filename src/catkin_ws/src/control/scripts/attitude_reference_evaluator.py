#!/usr/bin/env python3

# This file is used to evaluate the different methods in attitude_reference_generator.py
# to determine which is most suited

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import rospy
import drone_interface.msg

import control
import control_util

class AttitudeReferenceEvaluator():

    def __init__(self):
        node_name = "attitude_reference_evaluator"
        rospy.init_node(node_name)
        controller_params = control_util.load_control_params_config(node_name)

        self._controller = control.Controller(controller_params)

        self._prev_telemetry_timestamp: float = None
        self._prev_atttiude: np.ndarray = None # roll and pitch
        self._prev_velocity: np.ndarray = None # vx and vy

        rospy.Subscriber("/drone/out/telemetry", drone_interface.msg.AnafiTelemetry, self._drone_telemetry_cb)


    def _drone_telemetry_cb(self, msg: drone_interface.msg.AnafiTelemetry):

        self._prev_telemetry_timestamp = msg.header.stamp.to_sec()

        self._prev_atttiude = np.array([
            msg.roll,
            msg.pitch
        ])

        self._prev_velocity = np.array([
            msg.vx,
            msg.vy
        ])

    def evaluate_method(self):
        v_ref = control_util.get_reference_trajectory_safe()
        v_actual = np.zeros_like(v_ref)
        v_d = np.zeros_like(v_ref)
        att_actual = np.zeros_like(v_ref)
        att_ref = np.zeros_like(v_ref)
        time_refs = np.zeros(v_ref.shape[1])
        time_meas = np.zeros(v_ref.shape[1])

        x_d = np.zeros(4)
        dt = 0.05

        self._controller.takeoff()

        rate = rospy.Rate(20)

        control_util.await_user_confirmation("Start trajectory tracking")

        for i in range(v_ref.shape[1]):

            x_d = self._controller.get_smooth_reference(x_d, v_ref[:,i], dt)

            att_ref[:,i] = self._controller.set_attitude(
                x_d, self._prev_velocity, self._prev_telemetry_timestamp, debug=True
            )

            v_actual[:,i] = self._prev_velocity.copy()
            att_actual[:,i] = self._prev_atttiude.copy()
            time_refs[i] = rospy.Time.now().to_sec()
            time_meas[i] = self._prev_telemetry_timestamp
            v_d[:,i] = x_d[:2]

            rate.sleep()

        self._controller.land()

        import os

        script_dir = os.path.dirname(os.path.realpath(__file__))

        np.savetxt(f"{script_dir}/v_ref.txt", v_ref)
        np.savetxt(f"{script_dir}/v_d.txt", v_d)
        np.savetxt(f"{script_dir}/time_refs.txt", time_refs)
        np.savetxt(f"{script_dir}/v_actual.txt", v_actual)
        np.savetxt(f"{script_dir}/time_meas.txt", time_meas)
        np.savetxt(f"{script_dir}/att_ref.txt", att_ref)
        np.savetxt(f"{script_dir}/att_actual.txt", att_actual)

        control_util.plot_drone_velocity_vs_reference_trajectory(
            v_ref, v_d, time_refs, v_actual, time_meas
        )

        control_util.plot_drone_attitude_vs_reference(
            att_ref, time_refs, att_actual, time_meas, show_plot=True
        )

def visualize():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = f"{script_dir}/../../../../../out/controller_results/velocity_control_results"
    env = "sim"
    data_folder = "model_based/test_2"
    data_dir = f"{base_dir}/{env}/{data_folder}"

    v_ref = np.loadtxt(f"{data_dir}/v_ref.txt")
    v_d = np.loadtxt(f"{data_dir}/v_d.txt")
    time_refs = np.loadtxt(f"{data_dir}/time_refs.txt")
    v_actual = np.loadtxt(f"{data_dir}/v_actual.txt")
    time_meas = np.loadtxt(f"{data_dir}/time_meas.txt")
    att_ref = np.loadtxt(f"{data_dir}/att_ref.txt")
    att_actual = np.loadtxt(f"{data_dir}/att_actual.txt")


    if "model" in data_folder:
        vel_controller = "Linear drag model"
    else:
        vel_controller = "PID"

    velocity_title = f"Reference vs. measured horizontal velocities\nEnvironment: {env.upper()} - Velocity controller: {vel_controller}"
    attitude_title = f"Reference vs. measured roll and pitch angles\nEnvironment: {env.upper()} - Velocity controller: {vel_controller}"

    control_util.plot_drone_velocity_vs_reference_trajectory(
        v_ref, v_d, time_refs, v_actual, time_meas, plot_title=velocity_title,
        start_time_from_0=True, show_plot=False, save_fig=False
    )
    control_util.plot_drone_attitude_vs_reference(
        att_ref, time_refs, att_actual, time_meas, plot_title=attitude_title,
        start_time_from_0=True, show_plot=True, save_fig=False
    )





if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "plot":
        visualize()
    else:
        evaluator = AttitudeReferenceEvaluator()
        evaluator.evaluate_method()