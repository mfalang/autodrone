#!/usr/bin/env python3

# This file is used to evaluate the different methods in attitude_reference_generator.py
# to determine which is most suited

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import rospy
import std_msgs.msg
import drone_interface.msg

from attitude_reference_generator import PIDReferenceGenerator

# # Load velocity data to be used as reference and attitude data used for evaluation
# data_folder = "2022-3-10/17-52-44"

# script_dir = os.path.dirname(os.path.realpath(__file__))
# gt_filename = f"{script_dir}/../../../../../out/{data_folder}/ground_truths/drone_velocity_body_frame_and_attitude.txt"
# gt_data = np.loadtxt(gt_filename, skiprows=1)

# telemetry_filename = f"{script_dir}/../../../../../out/{data_folder}/estimates/anafi_raw_data.txt"
# telemetry_data = np.loadtxt(telemetry_filename, skiprows=1)


# gt_df = pd.read_csv(gt_filename, sep=" ", skiprows=1)
# gt_df.columns = ["timestamp", "vx", "vy", "vz", "roll", "pitch", "yaw"]
# gt_df["time"] = gt_df["timestamp"] - gt_df["timestamp"][0]

# telemetry_df = pd.read_csv(telemetry_filename, sep=" ", skiprows=1)
# telemetry_df.columns = ["timestamp", "vx", "vy", "vz", "roll", "pitch", "yaw"]
# telemetry_df["time"] = telemetry_df["timestamp"] - telemetry_df["timestamp"][0]

# ## Plotting
# sns.set()

# # Plot ground truth and drone telemetry velocity and attitude
# fig, ax = plt.subplots(2, 3, sharex=True)
# sns.lineplot(ax=ax[0,0], data=gt_df, x="time", y="vx", label="gt")
# sns.lineplot(ax=ax[1,0], data=gt_df, x="time", y="pitch", label="gt")
# sns.lineplot(ax=ax[0,1], data=gt_df, x="time", y="vy", label="gt")
# sns.lineplot(ax=ax[1,1], data=gt_df, x="time", y="roll", label="gt")
# sns.lineplot(ax=ax[0,2], data=gt_df, x="time", y="vz", label="gt")
# sns.lineplot(ax=ax[1,2], data=gt_df, x="time", y="yaw", label="gt")

# sns.lineplot(ax=ax[0,0], data=telemetry_df, x="time", y="vx", label="telemetry")
# sns.lineplot(ax=ax[1,0], data=telemetry_df, x="time", y="pitch", label="telemetry")
# sns.lineplot(ax=ax[0,1], data=telemetry_df, x="time", y="vy", label="telemetry")
# sns.lineplot(ax=ax[1,1], data=telemetry_df, x="time", y="roll", label="telemetry")
# sns.lineplot(ax=ax[0,2], data=telemetry_df, x="time", y="vz", label="telemetry")
# sns.lineplot(ax=ax[1,2], data=telemetry_df, x="time", y="yaw", label="telemetry")

# ax[0,0].legend()
# ax[1,0].legend()
# ax[0,1].legend()
# ax[1,1].legend()
# ax[0,2].legend()
# ax[1,2].legend()
# fig.suptitle("Ground truth and telemetry data")

# plt.show()

class AttitudeReferenceEvaluator():


    def __init__(self):
        self._prev_telemetry_timestamp: float = None
        self._prev_atttiude: np.ndarray = None # roll and pitch
        self._prev_velocity: np.ndarray = None # vx and vy

        rospy.init_node("attitude_reference_evaluator", anonymous=False)

        rospy.Subscriber("/drone/out/telemetry", drone_interface.msg.AnafiTelemetry, self._drone_telemetry_cb)

        self._attitude_ref_publisher = rospy.Publisher(
            "drone/cmd/set_attitude", drone_interface.msg.AttitudeSetpoint, queue_size=1
        )
        self._takeoff_publisher = rospy.Publisher(
            "drone/cmd/takeoff", std_msgs.msg.Empty, queue_size=1
        )

        self._land_publisher = rospy.Publisher(
            "drone/cmd/land", std_msgs.msg.Empty, queue_size=1
        )

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

    def _pack_attitude_ref_msg(self, att_ref: np.ndarray):
        msg = drone_interface.msg.AttitudeSetpoint()
        msg.header.stamp = rospy.Time.now()
        msg.climb_rate = 0
        msg.roll = att_ref[0]
        msg.pitch = att_ref[1]
        msg.yaw_rate = 0

        return msg

    def get_reference(self) -> np.ndarray:
        v_ref = np.zeros((2,180)) # 9 second reference signal
        v_ref[0,:60] = 1
        v_ref[1,60:120] = 1
        v_ref[:, 120:] = 1

        return v_ref

    def _takeoff(self, await_confirmation=True):
        if await_confirmation:
            input("Press enter to take off ")
        rospy.loginfo("Taking off")
        self._takeoff_publisher.publish(std_msgs.msg.Empty())

    def _land(self, await_confirmation=True):
        if await_confirmation:
            input("Press enter to land ")
        rospy.loginfo("Landing")
        self._land_publisher.publish(std_msgs.msg.Empty())

    def run(self):
        v_ref = self.get_reference()
        v_actual = np.zeros_like(v_ref)
        att_actual = np.zeros_like(v_ref)
        att_ref = np.zeros_like(v_ref)
        time = np.zeros(v_ref.shape[1])

        ref_generator = PIDReferenceGenerator(3,2,1)

        self._takeoff()

        input("Press enter to begin test ")

        rate = rospy.Rate(20)

        for i in range(v_ref.shape[1]):
            att_ref[:,i] = ref_generator.get_attitude_reference(
                v_ref[:,i], self._prev_velocity, self._prev_telemetry_timestamp
            )
            v_actual[:,i] = self._prev_velocity.copy()
            att_actual[:,i] = self._prev_atttiude.copy()
            time[i] = self._prev_telemetry_timestamp

            msg = self._pack_attitude_ref_msg(att_ref[:,i])

            self._attitude_ref_publisher.publish(msg)

            rate.sleep()

        self._land()

        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_ref.txt", v_ref)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_actual.txt", v_actual)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_ref.txt", att_ref)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_actual.txt", att_actual)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/time.txt", time)

        self.plot_reference_vs_actual(v_ref, v_actual, att_ref, att_actual, time)

        return v_ref, v_actual, att_ref, att_actual, time

    def plot_reference_vs_actual(self, v_ref, v_actual, att_ref, att_actual, time):
        sns.set()
        fig, ax = plt.subplots(2, 2, sharex=True)

        # Velocity references
        ax[0,0].plot(time - time[0], v_ref[0,:], label="vx_ref")
        ax[0,1].plot(time - time[0], v_ref[1,:], label="vy_ref")

        # Actual velocities
        ax[0,0].plot(time - time[0], v_actual[0,:], label="vx")
        ax[0,1].plot(time - time[0], v_actual[1,:], label="vy")

        # Attitude references
        ax[1,0].plot(time - time[0], att_ref[0,:], label="roll_ref")
        ax[1,1].plot(time - time[0], att_ref[1,:], label="pitch_ref")

        # Actual attitude
        ax[1,0].plot(time - time[0], att_actual[0,:], label="roll")
        ax[1,1].plot(time - time[0], att_actual[1,:], label="pitch")

        # Legends
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()

        plt.show()



def main():
    evaluator = AttitudeReferenceEvaluator()
    evaluator.run()

    # v_ref = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_ref.txt")
    # v_actual = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_actual.txt")
    # att_ref = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_ref.txt")
    # att_actual = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_actual.txt")
    # time = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/time.txt")

    # evaluator.plot_reference_vs_actual(v_ref, v_actual, att_ref, att_actual, time)

if __name__ == "__main__":
    main()