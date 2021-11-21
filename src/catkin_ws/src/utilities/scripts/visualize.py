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

        self.figure = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")


    def plot_drone_ground_truth(self):
        print("Plotting drone ground truth")
        pose = np.loadtxt(f"{self.data_dir}/ground_truths/drone_pose.txt", skiprows=1)
        self.ax.plot3D(pose[:,1], pose[:,2], pose[:,3], label="Drone ground truth")
        plt.legend()

    def plot_helipad_ground_truth(self):
        print("Plotting helipad ground truth")
        pose = np.loadtxt(f"{self.data_dir}/ground_truths/helipad_pose.txt", skiprows=1)
        self.ax.plot3D(pose[:,1], pose[:,2], pose[:,3], label="Helipad ground truth")
        plt.legend()

    def plot_drone_estimate(self):
        print("Plotting drone estimate raw from DNN CV")
        pose = np.loadtxt(f"{self.data_dir}/estimates/dnn_cv_pose.txt", skiprows=1)
        self.ax.plot3D(pose[:,1], pose[:,2], pose[:,3], label="Drone DNN CV estimate")
        plt.legend()

    def plot_helipad_estimate(self):
        print("Plotting helipad estimate (not implemented)")
        # pose = np.loadtxt(f"{self.data_dir}/estimate/helipad/pose.txt", skiprows=1)
        # self.ax.plot3D(pose[:,1], pose[:,2], pose[:,3], label="Helipad estimate")
        # plt.legend()


def main():
    parser = argparse.ArgumentParser(description="Visualize data.")
    parser.add_argument("data_dir", metavar="data_dir", type=str, help="Base directory of data")
    parser.add_argument("--drone_plots", choices=["all", "est", "gt", "none"],
        help="What drone data to visualize (default: all)", default="all"
    )
    parser.add_argument("--helipad_plots", choices=["all", "est", "gt", "none"],
        help="What helipad data to visualize (default: all)", default="all"
    )

    args = parser.parse_args()

    plotter = Plotter(args.data_dir)

    if args.drone_plots == "all":
        plotter.plot_drone_ground_truth()
        plotter.plot_drone_estimate()
    elif args.drone_plots == "est":
        plotter.plot_drone_estimate()
    elif args.drone_plots == "gt":
        plotter.plot_drone_ground_truth()

    if args.helipad_plots == "all":
        plotter.plot_helipad_ground_truth()
        plotter.plot_helipad_estimate()
    elif args.helipad_plots == "est":
        plotter.plot_helipad_estimate()
    elif args.helipad_plots == "gt":
        plotter.plot_helipad_ground_truth()

    plt.show()

if __name__ == "__main__":
    main()