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

def visualize():
    # Load velocity data to be used as reference and attitude data used for evaluation
    data_folder = "2022-3-10/17-52-44"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    gt_filename = f"{script_dir}/../../../../../out/{data_folder}/ground_truths/drone_velocity_body_frame_and_attitude.txt"
    gt_data = np.loadtxt(gt_filename, skiprows=1)

    telemetry_filename = f"{script_dir}/../../../../../out/{data_folder}/estimates/anafi_raw_data.txt"
    telemetry_data = np.loadtxt(telemetry_filename, skiprows=1)


    gt_df = pd.read_csv(gt_filename, sep=" ", skiprows=1)
    gt_df.columns = ["timestamp", "vx", "vy", "vz", "roll", "pitch", "yaw"]
    gt_df["time"] = gt_df["timestamp"] - gt_df["timestamp"][0]

    telemetry_df = pd.read_csv(telemetry_filename, sep=" ", skiprows=1)
    telemetry_df.columns = ["timestamp", "vx", "vy", "vz", "roll", "pitch", "yaw"]
    telemetry_df["time"] = telemetry_df["timestamp"] - telemetry_df["timestamp"][0]

    ## Plotting
    sns.set()

    # Plot ground truth and drone telemetry velocity and attitude
    fig, ax = plt.subplots(2, 3, sharex=True)
    sns.lineplot(ax=ax[0,0], data=gt_df, x="time", y="vx", label="gt")
    sns.lineplot(ax=ax[1,0], data=gt_df, x="time", y="pitch", label="gt")
    sns.lineplot(ax=ax[0,1], data=gt_df, x="time", y="vy", label="gt")
    sns.lineplot(ax=ax[1,1], data=gt_df, x="time", y="roll", label="gt")
    sns.lineplot(ax=ax[0,2], data=gt_df, x="time", y="vz", label="gt")
    sns.lineplot(ax=ax[1,2], data=gt_df, x="time", y="yaw", label="gt")

    sns.lineplot(ax=ax[0,0], data=telemetry_df, x="time", y="vx", label="telemetry")
    sns.lineplot(ax=ax[1,0], data=telemetry_df, x="time", y="pitch", label="telemetry")
    sns.lineplot(ax=ax[0,1], data=telemetry_df, x="time", y="vy", label="telemetry")
    sns.lineplot(ax=ax[1,1], data=telemetry_df, x="time", y="roll", label="telemetry")
    sns.lineplot(ax=ax[0,2], data=telemetry_df, x="time", y="vz", label="telemetry")
    sns.lineplot(ax=ax[1,2], data=telemetry_df, x="time", y="yaw", label="telemetry")

    ax[0,0].legend()
    ax[1,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()
    ax[0,2].legend()
    ax[1,2].legend()
    fig.suptitle("Ground truth and telemetry data")

    plt.show()

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

            x_d = self._controller.get_reference(x_d, v_ref[:,i], dt)

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

        control_util.plot_drone_velocity_vs_reference_trajectory(
            v_ref, v_d, time_refs, v_actual, time_meas
        )

        control_util.plot_drone_attitude_vs_reference(
            att_ref, time_refs, att_actual, time_meas, show_plot=True
        )

def main():
    evaluator = AttitudeReferenceEvaluator()
    evaluator.evaluate_method()

if __name__ == "__main__":
    main()