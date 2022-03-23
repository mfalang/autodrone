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

import attitude_reference_generator

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
        self._prev_telemetry_timestamp: float = None
        self._prev_atttiude: np.ndarray = None # roll and pitch
        self._prev_velocity: np.ndarray = None # vx and vy

        rospy.init_node("attitude_reference_evaluator", anonymous=False)

        rospy.Subscriber("/drone/out/telemetry", drone_interface.msg.AnafiTelemetry, self._drone_telemetry_cb)

        self._attitude_ref_publisher = rospy.Publisher(
            "drone/cmd/set_attitude", drone_interface.msg.AttitudeSetpoint, queue_size=2
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
        v_ref = np.zeros((2,900)) # 45 second reference signal

        # Forward backwards x
        v_ref[0,:100] = 0.5
        v_ref[0,100:150] = 0
        v_ref[0,150:250] = -0.5

        # Forward backwards y
        v_ref[1,300:400] = 0.5
        v_ref[1,400:450] = 0
        v_ref[1,450:550] = -0.5

        # # Forward backwards x and y
        v_ref[:,600:700] = 0.5
        v_ref[:,700:750] = 0
        v_ref[:,750:850] = -0.5

        return v_ref

    def _get_reference_short(self) -> np.ndarray:
        v_ref = np.zeros((2, 360)) # 18 seconds

        # Forward backwards x
        v_ref[0,:40] = 0.5
        v_ref[0,40:80] = 0
        v_ref[0,80:120] = -0.5

        # Forward backwards y
        v_ref[1,120:160] = 0.5
        v_ref[1,160:200] = 0
        v_ref[1,200:240] = -0.5

        # # Forward backwards x and y
        v_ref[:,240:280] = 0.5
        v_ref[:,280:320] = 0
        v_ref[:,320:] = -0.5

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

    def evaluate_PID_method(self):
        # v_ref = self.get_reference()
        v_ref = self._get_reference_short()
        v_actual = np.zeros_like(v_ref)
        v_d = np.zeros_like(v_ref)
        att_actual = np.zeros_like(v_ref)
        att_ref = np.zeros_like(v_ref)
        time_refs = np.zeros(v_ref.shape[1])
        time_meas = np.zeros(v_ref.shape[1])

        pitch_limits = (-20, 20)
        roll_limits = (-20, 20)

        pitch_gains = (-7, -0.001, -10)
        roll_gains = (7, 0.001, 10)

        ref_generator = attitude_reference_generator.PIDReferenceGenerator(
            roll_gains, pitch_gains, roll_limits, pitch_limits
        )

        omegas = (10, 10)
        zetas = (1, 1)

        ref_model = attitude_reference_generator.VelocityReferenceModel(
            omegas, zetas
        )
        x_d = np.zeros(4)

        dt = 0.05

        self._takeoff()

        input("Press enter to begin test ")

        rate = rospy.Rate(20)
        # Use the reference model here
        for i in range(v_ref.shape[1]):

            x_d = ref_model.get_filtered_reference(x_d, v_ref[:,i], dt)

            att_ref[:,i] = ref_generator.get_attitude_reference(
                x_d[:2], self._prev_velocity, self._prev_telemetry_timestamp,
                debug=True
            )
            v_actual[:,i] = self._prev_velocity.copy()
            att_actual[:,i] = self._prev_atttiude.copy()
            time_refs[i] = rospy.Time.now().to_sec() # self._prev_telemetry_timestamp
            time_meas[i] = self._prev_telemetry_timestamp
            print(f"Time diff s: {time_refs[i] - time_meas[i]}")
            v_d[:,i] = x_d[:2]

            msg = self._pack_attitude_ref_msg(att_ref[:,i])

            self._attitude_ref_publisher.publish(msg)

            rate.sleep()

        self._land()

        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_ref.txt", v_ref)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_actual.txt", v_actual)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_ref.txt", att_ref)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_actual.txt", att_actual)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/time.txt", time)

        sns.set()
        fig, ax = plt.subplots(2, 2, sharex=True)

        # time_refs -= time_refs[0]
        # time_meas -= time_meas[0]
        # time_meas += np.average(time_refs - time_meas)

        # Vx
        ax[0,0].plot(time_refs, v_ref[0,:], label="vx_ref")
        ax[0,0].plot(time_refs, v_d[0,:], label="vd_x")
        ax[0,0].plot(time_meas, v_actual[0,:], label="vx")
        ax[0,0].set_title(f"Kp: {pitch_gains[0]} Ki: {pitch_gains[1]} Kd: {pitch_gains[2]}")

        # Vy
        ax[0,1].plot(time_refs, v_ref[1,:], label="vy_ref")
        ax[0,1].plot(time_refs, v_d[1,:], label="vd_y")
        ax[0,1].plot(time_meas, v_actual[1,:], label="vy")
        ax[0,1].set_title(f"Kp: {roll_gains[0]} Ki: {roll_gains[1]} Kd: {roll_gains[2]}")


        # Pitch
        ax[1,0].plot(time_refs, att_ref[1,:], label="pitch_ref")
        ax[1,0].plot(time_meas, att_actual[1,:], label="pitch")

        # Roll
        ax[1,1].plot(time_refs, att_ref[0,:], label="roll_ref")
        ax[1,1].plot(time_meas, att_actual[0,:], label="roll")

        # Legends
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()

        plt.show()

        return v_ref, v_actual, att_ref, att_actual, time_refs, time_meas, roll_gains, pitch_gains

    def evaluate_linear_drag_model_based_method(self):

        v_ref = self._get_reference_short()
        # v_ref = self.get_reference()
        v_d = np.zeros_like(v_ref)
        v_actual = np.zeros_like(v_ref)
        att_actual = np.zeros_like(v_ref)
        att_ref = np.zeros_like(v_ref)
        time_refs = np.zeros(v_ref.shape[1])
        time_meas = np.zeros(v_ref.shape[1])
        acc_d = np.zeros_like(v_ref)

        drone_mass = rospy.get_param("/drone/mass_g") / 1000
        d_x = 0.08063504
        d_y = 0.09929089


        ref_generator = attitude_reference_generator.LinearDragModelReferenceGenerator(
            drone_mass, d_x, d_y
        )

        omegas = (3, 3)
        zetas = (1, 1)

        ref_model = attitude_reference_generator.VelocityReferenceModel(
            omegas, zetas
        )
        x_d = np.zeros(4)

        dt = 0.05

        self._takeoff()

        input("Press enter to begin test ")

        rate = rospy.Rate(20)

        for i in range(v_ref.shape[1]):

            x_d = ref_model.get_filtered_reference(x_d, v_ref[:,i], dt)

            att_ref[:,i] = ref_generator.get_attitude_reference(
                x_d[:2], self._prev_velocity, self._prev_telemetry_timestamp,
                debug=True
            )
            v_actual[:,i] = self._prev_velocity.copy()
            att_actual[:,i] = self._prev_atttiude.copy()
            time_refs[i] = rospy.Time.now().to_sec()
            time_meas[i] = self._prev_telemetry_timestamp
            v_d[:,i] = x_d[:2]
            acc_d[:,i] = x_d[2:]

            msg = self._pack_attitude_ref_msg(att_ref[:,i])

            self._attitude_ref_publisher.publish(msg)

            rate.sleep()

        self._land()

        sns.set()
        fig, ax = plt.subplots(2, 2, sharex=True)

        # time -= time[0] # set time to start at 0 seconds

        # Vx
        ax[0,0].plot(time_refs, v_ref[0,:], label="vx_ref")
        ax[0,0].plot(time_refs, v_d[0,:], label="vd_x")
        # ax[0,0].plot(time, acc_d[0,:], label="acc_d_x")
        ax[0,0].plot(time_meas, v_actual[0,:], label="vx")

        # Vy
        ax[0,1].plot(time_refs, v_ref[1,:], label="vy_ref")
        ax[0,1].plot(time_refs, v_d[1,:], label="vd_y")
        # ax[0,1].plot(time, acc_d[1,:], label="acc_d_y")
        ax[0,1].plot(time_meas, v_actual[1,:], label="vy")

        # Pitch
        ax[1,0].plot(time_refs, att_ref[1,:], label="pitch_ref")
        ax[1,0].plot(time_meas, att_actual[1,:], label="pitch")

        # Roll
        ax[1,1].plot(time_refs, att_ref[0,:], label="roll_ref")
        ax[1,1].plot(time_meas, att_actual[0,:], label="roll")

        # Legends
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()

        plt.show()

        return v_ref, v_actual, att_ref, att_actual, time_refs

    def evaluate_iP_method(self):
        # v_ref = self.get_reference()
        v_ref = self._get_reference_short()
        v_actual = np.zeros_like(v_ref)
        v_d = np.zeros_like(v_ref)
        att_actual = np.zeros_like(v_ref)
        att_ref = np.zeros_like(v_ref)
        time_refs = np.zeros(v_ref.shape[1])
        time_meas = np.zeros(v_ref.shape[1])

        pitch_limits = (-5, 5)
        roll_limits = (-5, 5)

        # pitch_gains = (-7, -0.001, -10)
        # roll_gains = (7, 0.001, 10)
        pitch_gains = (7, 0.5, 0.1)
        roll_gains = (-7, -0.5, -0.1)

        # Kp_pitch = 7
        # Kp_roll = -7
        alpha_pitch = 1
        alpha_roll = 1

        ref_generator = attitude_reference_generator.iPReferenceGenerator(
            roll_gains, pitch_gains, alpha_roll, alpha_pitch, roll_limits, pitch_limits
        )

        omegas = (10, 10)
        zetas = (1, 1)

        ref_model = attitude_reference_generator.VelocityReferenceModel(
            omegas, zetas
        )
        x_d = np.zeros(4)

        dt = 0.05

        self._takeoff()

        input("Press enter to begin test ")

        rate = rospy.Rate(20)
        # Use the reference model here
        for i in range(v_ref.shape[1]):

            x_d = ref_model.get_filtered_reference(x_d, v_ref[:,i], dt)

            att_ref[:,i] = ref_generator.get_attitude_reference(
                x_d[:2], self._prev_velocity, x_d[2:], self._prev_telemetry_timestamp, debug=True
            )

            v_actual[:,i] = self._prev_velocity.copy()
            att_actual[:,i] = self._prev_atttiude.copy()
            time_refs[i] = rospy.Time.now().to_sec() # self._prev_telemetry_timestamp
            time_meas[i] = self._prev_telemetry_timestamp
            v_d[:,i] = x_d[:2]

            msg = self._pack_attitude_ref_msg(att_ref[:,i])

            self._attitude_ref_publisher.publish(msg)

            rate.sleep()

        self._land()

        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_ref.txt", v_ref)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_actual.txt", v_actual)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_ref.txt", att_ref)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_actual.txt", att_actual)
        # np.savetxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/time.txt", time)

        sns.set()
        fig, ax = plt.subplots(2, 2, sharex=True)

        # time_refs -= time_refs[0]
        # time_meas -= time_meas[0]
        # time_meas += np.average(time_refs - time_meas)

        # Vx
        ax[0,0].plot(time_refs, v_ref[0,:], label="vx_ref")
        ax[0,0].plot(time_refs, v_d[0,:], label="vd_x")
        ax[0,0].plot(time_meas, v_actual[0,:], marker=".", label="vx")
        ax[0,0].set_title(f"Kp: {pitch_gains[0]} Ki: {pitch_gains[1]} Kd: {pitch_gains[2]}")

        # Vy
        ax[0,1].plot(time_refs, v_ref[1,:], label="vy_ref")
        ax[0,1].plot(time_refs, v_d[1,:], label="vd_y")
        ax[0,1].plot(time_meas, v_actual[1,:], marker=".", label="vy")
        ax[0,1].set_title(f"Kp: {roll_gains[0]} Ki: {roll_gains[1]} Kd: {roll_gains[2]}")


        # Pitch
        ax[1,0].plot(time_refs, att_ref[1,:], label="pitch_ref")
        ax[1,0].plot(time_meas, att_actual[1,:], marker=".", label="pitch")

        # Roll
        ax[1,1].plot(time_refs, att_ref[0,:], label="roll_ref")
        ax[1,1].plot(time_meas, att_actual[0,:], marker=".", label="roll")

        # Legends
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()

        plt.show()

        return v_ref, v_actual, att_ref, att_actual, time_refs, time_meas

    def test_reference_model(self):

        omegas = (10, 10)
        zetas = (1, 1)

        ref_model = attitude_reference_generator.VelocityReferenceModel(omegas, zetas)

        v_ref = self._get_reference_short()

        v_d = np.zeros_like(v_ref)
        acc_d = np.zeros_like(v_ref)

        time = np.linspace(0, 9, num=v_ref.shape[1])

        x_d = np.zeros(4)

        dt = 0.05

        for i in range(v_ref.shape[1]):

            x_d = ref_model.get_filtered_reference(x_d, v_ref[:,i], dt)

            v_d[:,i] = x_d[:2]
            acc_d[:,i] = x_d[2:]

        sns.set()
        fig, ax = plt.subplots(2, 2, sharex=True)

        fig.suptitle("Velocity and acceleration references from second order velocity reference model")

        # x-axis velocity reference
        ax[0,0].plot(time, v_d[0,:], label="vd_x")
        ax[0,0].plot(time, v_ref[0,:], label="vx_ref")

        # y-axis velocity reference
        ax[0,1].plot(time, v_d[1,:], label="vd_x")
        ax[0,1].plot(time, v_ref[1,:], label="vy_ref")

        # x-axis acceleration reference
        ax[1,0].plot(time, acc_d[0,:], label="acc_d_x")

        # y-axis acceleration reference
        ax[1,1].plot(time, acc_d[1,:], label="acc_d_y")

        # Legends
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()

        plt.show()



def main():
    evaluator = AttitudeReferenceEvaluator()
    # evaluator.evaluate_PID_method()
    # evaluator.evaluate_linear_drag_model_based_method()
    evaluator.evaluate_iP_method()
    # evaluator.test_reference_model()

    # visualize()

    # v_ref = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_ref.txt")
    # v_actual = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/v_actual.txt")
    # att_ref = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_ref.txt")
    # att_actual = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/att_actual.txt")
    # time = np.loadtxt("/home/martin/code/autodrone/src/catkin_ws/src/control/scripts/time.txt")

    # evaluator.plot_reference_vs_actual(v_ref, v_actual, att_ref, att_actual, time)

if __name__ == "__main__":
    main()