#!/usr/bin/env python3

import os
import argparse
import numpy as np
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

    def plot_drone_ground_truth(self):
        print("Plotting drone ground truth")
        pose = np.loadtxt(f"{self.data_dir}/ground_truths/drone_pose.txt", skiprows=1)
        self.pose3D_ax.plot3D(pose[:,1], pose[:,2], -pose[:,3], label="Drone ground truth")
        plt.legend()

    def plot_helipad_ground_truth(self):
        print("Plotting helipad ground truth")
        pose = np.loadtxt(f"{self.data_dir}/ground_truths/helipad_pose.txt", skiprows=1)
        self.pose3D_ax.plot3D(pose[:,1], pose[:,2], -pose[:,3], label="Helipad ground truth")
        plt.legend()

    def plot_drone_pose_dnn_cv(self):
        print("Plotting drone estimate raw from DNN CV")
        pose = np.loadtxt(f"{self.data_dir}/estimates/dnn_cv_pose.txt", skiprows=1)
        self.pose3D_ax.plot3D(pose[:,1], pose[:,2], -pose[:,3], label="Drone DNN CV estimate")
        plt.legend()

    def plot_drone_pose_tcv(self):
        print("Plotting drone estimate raw from TCV")
        pose = np.loadtxt(f"{self.data_dir}/estimates/tcv_pose.txt", skiprows=1)
        self.pose3D_ax.plot3D(pose[:,1], pose[:,2], -pose[:,3], label="Drone TCV estimate")
        plt.legend()

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


def main():
    parser = argparse.ArgumentParser(description="Visualize data.")
    parser.add_argument("data_dir", metavar="data_dir", type=str, help="Base directory of data")
    parser.add_argument("--estimate_plots", choices=["all", "ekf_pose", "dnn_cv_pose", "tcv_pose", "none"],
        help="What estimation data to visualize (default: all)", default="all"
    )
    parser.add_argument("--gt_plots", choices=["all", "drone_pose", "helipad_pose", "none"],
        help="What ground truth data to visualize (default: all)", default="all"
    )

    args = parser.parse_args()

    plotter = Plotter(args.data_dir)

    if args.gt_plots == "all":
        plotter.plot_drone_ground_truth()
        plotter.plot_helipad_ground_truth()
    elif args.gt_plots == "drone_pose":
        plotter.plot_drone_ground_truth()
    elif args.gt_plots == "helipad_pose":
        plotter.plot_helipad_ground_truth()

    if args.estimate_plots == "all":
        plotter.plot_drone_pose_dnn_cv()
        plotter.plot_drone_pose_ekf()
        plotter.plot_drone_pose_tcv()
    elif args.estimate_plots == "dnn_cv_pose":
        plotter.plot_drone_pose_dnn_cv()
    elif args.estimate_plots == "ekf_pose":
        plotter.plot_drone_pose_ekf()
    elif args.estimate_plots == "tcv_pose":
        plotter.plot_drone_pose_tcv()

    plt.show()

if __name__ == "__main__":
    main()