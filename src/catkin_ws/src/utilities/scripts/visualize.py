#!/usr/bin/env python3

from audioop import rms
import os
import argparse
from cv2 import merge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Plotter():

    def __init__(self, data_dir):

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = f"{script_dir}/../../../../../out/{data_dir}"

        self.pose3D_figure = plt.figure(1)
        self.pose3D_ax = plt.axes(projection='3d')
        self.pose3D_ax.set_xlabel("x [m]")
        self.pose3D_ax.set_ylabel("y [m]")
        self.pose3D_ax.set_zlabel("z [m]")
        self.pose3D_ax.set_xlim(-2, 2)
        self.pose3D_ax.set_ylim(-2, 2)

        self.orientation_fig, self.orientation_ax = plt.subplots(3, 1)
        # self.orientation_fig.suptitle("Estimated (DNN CV) vs. ground truth angles")
        # self.orientation_ax[0].set_title("Estimated vs. ground truth roll")
        # self.orientation_ax[1].set_title("Estimated vs. ground truth pitch")
        # self.orientation_ax[2].set_title("Estimated vs. ground truth yaw")
        self.orientation_ax[0].set_xlabel("Time [sec]")
        self.orientation_ax[0].set_ylabel("Roll angle [deg]")
        self.orientation_ax[1].set_xlabel("Time [sec]")
        self.orientation_ax[1].set_ylabel("Pitch angle [deg]")
        self.orientation_ax[2].set_xlabel("Time [sec]")
        self.orientation_ax[2].set_ylabel("Yaw angle [deg]")

        self.dnn_cv_yaw_fig = plt.figure(2)
        self.dnn_cv_yaw_ax = plt.axes()
        self.dnn_cv_yaw_ax.set_xlabel("Time [sec]")
        self.dnn_cv_yaw_ax.set_ylabel("Yaw angle [deg]")
        self.dnn_cv_yaw_ax.set_title("Estimated (DNN CV) vs. ground truth yaw angle")

        self.gt_first_index = 0
        self.tcv_first_index = 0
        self.dnn_cv_first_index = 0

    def plot_drone_ground_truth(self):
        print("Plotting drone ground truth")
        # skiprows=2 instead of 1 because for some reason the first timestamp is negative so skip it
        pose = np.loadtxt(f"{self.data_dir}/ground_truths/helipad_pose_body_frame.txt", skiprows=2)
        self.pose3D_ax.plot3D(pose[:,1], pose[:,2], pose[:,3], label="Drone ground truth")
        # self.pose3D_figure.legend()
        self.orientation_ax[0].plot(pose[self.gt_first_index:,0] - pose[self.gt_first_index,0], pose[self.gt_first_index:,4], label="GT")
        self.orientation_ax[1].plot(pose[self.gt_first_index:,0] - pose[self.gt_first_index,0], pose[self.gt_first_index:,5], label="GT")
        self.orientation_ax[2].plot(pose[self.gt_first_index:,0] - pose[self.gt_first_index,0], pose[self.gt_first_index:,6], label="GT")
        self.orientation_ax[0].legend(loc="lower right")
        self.orientation_ax[1].legend(loc="lower right")
        self.orientation_ax[2].legend(loc="lower right")

        self.pos2D_ax[0].plot(pose[self.gt_first_index:,0] - pose[self.gt_first_index,0], pose[self.gt_first_index:,1], label="GT")
        self.pos2D_ax[1].plot(pose[self.gt_first_index:,0] - pose[self.gt_first_index,0], pose[self.gt_first_index:,2], label="GT")
        self.pos2D_ax[2].plot(pose[self.gt_first_index:,0] - pose[self.gt_first_index,0], pose[self.gt_first_index:,3], label="GT")
        self.pos2D_ax[0].legend(loc="lower right")
        self.pos2D_ax[1].legend(loc="lower right")
        self.pos2D_ax[2].legend(loc="lower right")

        self.dnn_cv_yaw_ax.plot(pose[self.gt_first_index:,0] - pose[self.gt_first_index,0], pose[self.gt_first_index:,6], label="GT")
        self.dnn_cv_yaw_ax.legend(loc="lower right")

    def plot_drone_dnncv_estimates(self):
        print("Plotting estimates from DNN CV")
        position = np.loadtxt(f"{self.data_dir}/estimates/dnn_cv_position.txt", skiprows=1)

        # Plot in 3D plot
        self.pose3D_ax.scatter(position[:,1], position[:,2], position[:,3], s=1, c="red", label="Drone DNN CV estimate")
        self.pose3D_figure.legend()

        # Plot each component individually
        self.pos2D_ax[0].scatter(position[:,0] - position[self.dnn_cv_first_index,0], position[:,1], s=5, c="red", label="DNN CV")
        self.pos2D_ax[1].scatter(position[:,0] - position[self.dnn_cv_first_index,0], position[:,2], s=5, c="red", label="DNN CV")
        self.pos2D_ax[2].scatter(position[:,0] - position[self.dnn_cv_first_index,0], position[:,3], s=5, c="red", label="DNN CV")
        self.pos2D_ax[0].legend(loc="lower right")
        self.pos2D_ax[1].legend(loc="lower right")
        self.pos2D_ax[2].legend(loc="lower right")

        # Plot heading
        heading = np.loadtxt(f"{self.data_dir}/estimates/dnn_cv_heading.txt", skiprows=1)
        self.dnn_cv_yaw_ax.scatter(heading[:,0] - heading[self.dnn_cv_first_index,0], heading[:,1], s=5, c="red", label="DNN CV")
        self.dnn_cv_yaw_ax.legend(loc="lower right")


    # def plot_helipad_ground_truth(self):
    #     print("Plotting helipad ground truth")
    #     pose = np.loadtxt(f"{self.data_dir}/ground_truths/helipad_pose.txt", skiprows=1)
    #     self.pose3D_ax.plot3D(pose[:,1], pose[:,2], -pose[:,3], label="Helipad ground truth")
    #     self.pose3D_figure.legend()

    # def plot_drone_pose_dnn_cv(self):
    #     print("Plotting drone estimate raw from DNN CV")
    #     pose = np.loadtxt(f"{self.data_dir}/estimates/dnn_cv_pose.txt", skiprows=1)
    #     # self.pose3D_ax.plot3D(pose[:,1], -pose[:,2], -pose[:,3], label="Drone DNN CV estimate")
    #     self.pose3D_ax.scatter(pose[:,1], -pose[:,2], -pose[:,3], s=1, c="red", label="Drone DNN CV estimate")
    #     self.pose3D_figure.legend()

    #     # self.orientation_ax[2].scatter(pose[:,0] - pose[0,0], pose[:,4], s=5, c="red", label="DNN CV")
    #     # self.orientation_ax[2].legend(loc="lower right")

    #     self.pos2D_ax[0].scatter(pose[:,0] - pose[0,0], pose[:,1], s=5, c="red", label="DNN CV")
    #     self.pos2D_ax[1].scatter(pose[:,0] - pose[0,0], -pose[:,2], s=5, c="red", label="DNN CV")
    #     self.pos2D_ax[2].scatter(pose[:,0] - pose[0,0], pose[:,3], s=5, c="red", label="DNN CV")
    #     self.pos2D_ax[0].legend(loc="lower right")
    #     self.pos2D_ax[1].legend(loc="lower right")
    #     self.pos2D_ax[2].legend(loc="lower right")

    #     # SSA
    #     yaw = (pose[:,4] + 90 + 180) % 360 - 180

    #     self.dnn_cv_yaw_ax.scatter(pose[:,0] - pose[0,0], yaw, s=5, c="red", label="DNN CV")
    #     self.dnn_cv_yaw_ax.legend(loc="lower right")


    def plot_drone_pose_tcv(self):
        print("Plotting drone estimate raw from TCV")
        pose = np.loadtxt(f"{self.data_dir}/estimates/tcv_pose.txt", skiprows=1)
        self.pose3D_ax.scatter(pose[:,1], pose[:,2], -pose[:,3], s=1, c="red", label="Drone TCV estimate")
        self.pose3D_figure.legend()
        self.orientation_ax[0].scatter(pose[:,0] - pose[0,0], pose[:,4], s=5, c="red", label="TCV")
        self.orientation_ax[1].scatter(pose[:,0] - pose[0,0], pose[:,5], s=5, c="red", label="TCV")
        self.orientation_ax[2].scatter(pose[:,0] - pose[0,0], pose[:,6], s=5, c="red", label="TCV")
        self.orientation_ax[0].legend(loc="lower right")
        self.orientation_ax[1].legend(loc="lower right")
        self.orientation_ax[2].legend(loc="lower right")

        self.pos2D_ax[0].scatter(pose[:,0] - pose[0,0], pose[:,1], s=5, c="red", label="TCV")
        self.pos2D_ax[1].scatter(pose[:,0] - pose[0,0], pose[:,2], s=5, c="red", label="TCV")
        self.pos2D_ax[2].scatter(pose[:,0] - pose[0,0], pose[:,3], s=5, c="red", label="TCV")
        self.pos2D_ax[0].legend(loc="lower right")
        self.pos2D_ax[1].legend(loc="lower right")
        self.pos2D_ax[2].legend(loc="lower right")

    def plot_drone_pose_ekf(self):
        print("Plotting drone estimate from EKF and covariance")
        output = np.loadtxt(f"{self.data_dir}/estimates/ekf_output.txt", skiprows=1)
        self.pose3D_ax.plot(output[:,1], output[:,2], -output[:,3], label="Drone EKF estimate")
        plt.legend()

    def plot_drone_pose_cov(self):
        print("Plotting drone pose estimate covariance")
        output = np.loadtxt(f"{self.data_dir}/estimates/ekf_output.txt", skiprows=1)



    def plot_helipad_estimate(self):
        print("Plotting helipad estimate (not implemented)")
        # pose = np.loadtxt(f"{self.data_dir}/estimate/helipad/pose.txt", skiprows=1)
        # self.pose3D_ax.plot3D(pose[:,1], pose[:,2], pose[:,3], label="Helipad estimate")
        # plt.legend()

    def synch_tcv_and_gt_timestamps(self):
        tcv_timestamps = np.loadtxt(f"{self.data_dir}/estimates/tcv_pose.txt", skiprows=1)[:,0]
        gt_timestamps = np.loadtxt(f"{self.data_dir}/ground_truths/drone_pose.txt", skiprows=2)[:,0]

        tcv_first_timestamp = tcv_timestamps[0]
        self.gt_first_index = np.argmin(np.abs(gt_timestamps-tcv_first_timestamp))
        self.tcv_first_index = 0

    def synch_dnn_cv_and_gt_timestamps(self):
        dnn_cv_timestamps = np.loadtxt(f"{self.data_dir}/estimates/dnn_cv_position.txt", skiprows=1)[:,0]
        gt_timestamps = np.loadtxt(f"{self.data_dir}/ground_truths/drone_pose_helipad_frame.txt", skiprows=2)[:,0]

        dnn_cv_first_timestamp = dnn_cv_timestamps[0]
        self.gt_first_index = np.argmin(np.abs(gt_timestamps-dnn_cv_first_timestamp))
        self.dnn_cv_first_index = 0

    def sync_dnncv_and_gt_data(self):
        dnncv_data = np.loadtxt(f"{self.data_dir}/estimates/dnn_cv_position.txt", skiprows=1)
        gt_data = np.loadtxt(f"{self.data_dir}/ground_truths/helipad_pose_body_frame.txt", skiprows=2)

        dnncv_df = pd.DataFrame()
        # dnncv_df["time"] = pd.to_datetime(dnncv_data[:,0], unit="s")
        dnncv_df["time"] = dnncv_data[:,0]
        dnncv_df["est_x"] = dnncv_data[:,1]
        dnncv_df["est_y"] = dnncv_data[:,2]
        dnncv_df["est_z"] = dnncv_data[:,3]

        gt_df = pd.DataFrame()
        # gt_df["time"] = pd.to_datetime(gt_data[:,0], unit="s")
        gt_df["time"] = gt_data[:,0]
        gt_df["gt_x"] = gt_data[:,1]
        gt_df["gt_y"] = gt_data[:,2]
        gt_df["gt_z"] = gt_data[:,3]

        merged = pd.merge_asof(dnncv_df, gt_df, on="time", allow_exact_matches=True, direction="nearest")

        timestamps = merged["time"].to_numpy()
        estimates = np.array([merged["est_x"].to_numpy(), merged["est_y"].to_numpy(), merged["est_z"].to_numpy()]).T
        ground_truths = np.array([merged["gt_x"].to_numpy(), merged["gt_y"].to_numpy(), merged["gt_z"].to_numpy()]).T

        np.savetxt("ground_truths_from_pandas.txt", ground_truths)

        return estimates, ground_truths, timestamps

    def plot_dnncv_estimate_vs_ground_truth(self, estimates, ground_truths, timestamps):

        rmse_all = np.sqrt(np.mean((estimates - ground_truths)**2))
        rmse_x = np.sqrt(np.mean((estimates[:,0] - ground_truths[:,0])**2))
        rmse_y = np.sqrt(np.mean((estimates[:,1] - ground_truths[:,1])**2))
        rmse_z = np.sqrt(np.mean((estimates[:,2] - ground_truths[:,2])**2))

        pos2D_fig, pos2D_ax = plt.subplots(3, 1)
        pos2D_fig.suptitle(f"DNN CV helipad position estimate vs. ground truth. Total RMSE: {rmse_all:.3f}m")
        # pos2D_ax[0].set_xlabel("Time [sec]")
        pos2D_ax[0].set_ylabel("x [m]")
        pos2D_ax[0].set_title(f"RMSE: {rmse_x:.3f}m")
        # pos2D_ax[1].set_xlabel("Time [sec]")
        pos2D_ax[1].set_ylabel("y [m]")
        pos2D_ax[1].set_title(f"RMSE: {rmse_y:.3f}m")
        pos2D_ax[2].set_xlabel("Time [sec]")
        pos2D_ax[2].set_ylabel("z [m]")
        pos2D_ax[2].set_title(f"RMSE: {rmse_z:.3f}m")

        # Estimates
        pos2D_ax[0].scatter(timestamps - timestamps[0], estimates[:,0], s=5, c="red", label="DNN CV")
        pos2D_ax[1].scatter(timestamps - timestamps[0], estimates[:,1], s=5, c="red", label="DNN CV")
        pos2D_ax[2].scatter(timestamps - timestamps[0], estimates[:,2], s=5, c="red", label="DNN CV")

        # Ground truths
        pos2D_ax[0].plot(timestamps - timestamps[0], ground_truths[:,0], label="GT")
        pos2D_ax[1].plot(timestamps - timestamps[0], ground_truths[:,1], label="GT")
        pos2D_ax[2].plot(timestamps - timestamps[0], ground_truths[:,2], label="GT")

        pos2D_ax[0].legend(loc="lower right")
        pos2D_ax[1].legend(loc="lower right")
        pos2D_ax[2].legend(loc="lower right")




def main():
    parser = argparse.ArgumentParser(description="Visualize data.")
    parser.add_argument("data_dir", metavar="data_dir", type=str, help="Base directory of data")
    parser.add_argument("--plots", choices=["all", "ekf", "dnncv", "tcv", "drone_gt", "helipad_gt", "none"],
        help="What estimation data to visualize (default: all)", default="all"
    )

    args = parser.parse_args()

    plotter = Plotter(args.data_dir)

    if (args.plots == "dnncv" and args.plots == "drone_gt") or args.plots == "all":
        est, gt, ts = plotter.sync_dnncv_and_gt_data()
        plotter.plot_dnncv_estimate_vs_ground_truth(est, gt, ts)
        # plotter.synch_dnn_cv_and_gt_timestamps()
        # plotter.plot_drone_dnncv_estimates()
        # plotter.plot_drone_ground_truth()


    # if args.gt_plots == "drone_pose" and args.estimate_plots == "tcv_pose":
    #     plotter.synch_tcv_and_gt_timestamps()

    # if args.gt_plots == "drone_pose" and args.estimate_plots == "dnn_cv_pose":
    #     plotter.synch_dnn_cv_and_gt_timestamps()

    # if args.gt_plots == "all":
    #     plotter.plot_drone_ground_truth()
    #     plotter.plot_helipad_ground_truth()
    # elif args.gt_plots == "drone_pose":
    #     plotter.plot_drone_ground_truth()
    # elif args.gt_plots == "helipad_pose":
    #     plotter.plot_helipad_ground_truth()

    # if args.estimate_plots == "all":
    #     plotter.plot_drone_pose_dnn_cv()
    #     plotter.plot_drone_pose_ekf()
    #     plotter.plot_drone_pose_tcv()
    # elif args.estimate_plots == "dnn_cv_pose":
    #     plotter.plot_drone_pose_dnn_cv()
    # elif args.estimate_plots == "ekf_pose":
    #     plotter.plot_drone_pose_ekf()
    # elif args.estimate_plots == "tcv_pose":
    #     plotter.plot_drone_pose_tcv()

    plt.show()

if __name__ == "__main__":
    main()