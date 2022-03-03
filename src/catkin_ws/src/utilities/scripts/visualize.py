#!/usr/bin/env python3

import os
import argparse
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Plotter():

    def match_two_dataseries_one_to_one(self, data1: np.ndarray, data2: np.ndarray):

        if data1.shape[0] > data2.shape[0]:
            dense = data1
            dense_prefix = "data1"
            sparse = data2
            sparse_prefix = "data2"
        else:
            dense = data2
            dense_prefix = "data2"
            sparse = data1
            sparse_prefix = "data1"

        dense_df = pd.DataFrame()
        dense_df["time"] = dense[:,0]
        for i in range(1, dense.shape[1]):
            dense_df[f"{dense_prefix}_{i-1}"] = dense[:,i]

        sparse_df = pd.DataFrame()
        sparse_df["time"] = sparse[:,0]
        for i in range(1, sparse.shape[1]):
            sparse_df[f"{sparse_prefix}_{i-1}"] = sparse[:,i]

        merged = pd.merge_asof(sparse_df, dense_df, on="time", allow_exact_matches=True, direction="nearest")

        timestamps = merged["time"].to_numpy()

        data1_out = []
        for i in range(data1.shape[1] - 1):
            data1_out.append(merged[f"data1_{i}"].to_numpy())
        data1_out = np.array(data1_out).T

        data2_out = []
        for i in range(data2.shape[1] - 1):
            data2_out.append(merged[f"data2_{i}"].to_numpy())
        data2_out = np.array(data2_out).T

        return timestamps, data1_out, data2_out

    def sync_multiple_data_series_based_on_timestamps(self, data: list):

        data_frames = []

        for i, dataseries in enumerate(data):
            df = pd.DataFrame()
            df["time"] = dataseries[:,0]
            for j in range(1, dataseries.shape[1]):
                df[f"data_{i}_{j-1}"] = dataseries[:,j]
            data_frames.append(df)

        df_merged = functools.reduce(lambda left,right: pd.merge(left, right, on="time", how="outer"), data_frames)

        data_out = []

        for i in range(len(data)):
            data_i = []
            data_i.append(df_merged["time"])

            num_columns = data[i].shape[1] - 1 # not including time
            for j in range(num_columns):
                data_i.append(df_merged[f"data_{i}_{j}"].to_numpy())
            data_i = np.array(data_i).T

            data_out.append(data_i)

        return data_out

    def calculate_rmse_and_plot(self, timestamps, data1, data2, suptitle="", data1_label="", data2_label="",
        xlabels=[], ylabels=[], plot_std_devs=False, std_devs=[]
    ):
        if data1.shape == data2.shape:
            rmse_all = np.sqrt(np.mean((data1 - data2)**2))
            rmse_per_column = []
            for i in range(data1.shape[1]):
                rmse_per_column.append(np.sqrt(np.mean((data1[:,i] - data2[:,i])**2)))

            print("="*10, f"RMSE calculations ({suptitle})", "="*10)
            print(f"Total RMSE ({data1.shape[1]} axis): {rmse_all:.4f}\n")
            for i, rmse in enumerate(rmse_per_column):
                print(f"Column {i}: {rmse:.4f}")

        fig, ax = plt.subplots(data1.shape[1], 1)
        fig.suptitle(suptitle)

        for i in range(ax.shape[0]):
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_xlabel(xlabels[i])
            ax[i].plot(timestamps - timestamps[0], data1[:,i], label=data1_label)
            ax[i].plot(timestamps - timestamps[0], data2[:,i], label=data2_label)

            if plot_std_devs:
                ax[i].plot(timestamps - timestamps[0], data1[:,i] + std_devs[i], c="red", ls="--", label="std")
                ax[i].plot(timestamps - timestamps[0], data1[:,i] - std_devs[i], c="red", ls="--", label="std")
                # ax[i].plot(timestamps - timestamps[0], std_devs[i], c="red", label="std")

            ax[i].legend(loc="lower right")

    def plot_multiple_data_series(self, data: list, numplots: int, suptitle: str,
        legends: list, xlabels: list, ylabels: list, use_scatter: list
    ):
        fig, ax = plt.subplots(numplots, 1)
        fig.suptitle(suptitle)
        for i in range(ax.shape[0]):
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_xlabel(xlabels[i])
            for j in range(len(data)):
                if use_scatter[j]:
                    ax[i].scatter(data[j][:,0] - data[j][0,0], data[j][:,i+1], s=4, label=legends[j])
                else:
                    ax[i].plot(data[j][:,0] - data[j][0,0], data[j][:,i+1], label=legends[j])
            ax[i].legend(loc="lower right")
            ax[i].grid()

def main():
    parser = argparse.ArgumentParser(description="Visualize data.")
    parser.add_argument("data_dir", metavar="data_dir", type=str, help="Base directory of data")

    args = parser.parse_args()

    plotter = Plotter()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{script_dir}/../../../../../out/{args.data_dir}"

    # Load data
    dnncv_data = np.loadtxt(f"{data_dir}/estimates/dnn_cv_position.txt", skiprows=1)
    gt_data = np.loadtxt(f"{data_dir}/ground_truths/helipad_pose_body_frame.txt", skiprows=2)
    ekfpos_data = np.loadtxt(f"{data_dir}/estimates/ekf_position.txt", skiprows=1)

    # Calculate accuracy of DNNCV
    ts, dnncv_pos, gt_pos = plotter.match_two_dataseries_one_to_one(dnncv_data, gt_data)
    plotter.calculate_rmse_and_plot(
        ts,
        dnncv_pos[:,:3],
        gt_pos[:,:3],
        suptitle="DNNCV pos estimate vs. ground truth",
        data1_label="DNNCV",
        data2_label="GT",
        ylabels=["x[m]", "y[m]", "z[m]"],
        xlabels=["t [sec]", "t [sec]", "t [sec]"]
    )

    # Calculate accuracy of EKF
    ts, ekf_pos, gt_pos = plotter.match_two_dataseries_one_to_one(ekfpos_data, gt_data)
    plotter.calculate_rmse_and_plot(
        ts,
        ekf_pos[:,:3],
        gt_pos[:,:3],
        suptitle="EKF pos estimate vs. ground truth",
        data1_label="EKF",
        data2_label="GT",
        ylabels=["x[m]", "y[m]", "z[m]"],
        xlabels=["t [sec]", "t [sec]", "t [sec]"],
        plot_std_devs=True,
        std_devs = [np.sqrt(ekf_pos[:,3]), np.sqrt(ekf_pos[:,7]), np.sqrt(ekf_pos[:,11])]
    )

    # Compare DNNCV and EKF estimate to ground truth
    synced_gt_dnncv_ekf_data = plotter.sync_multiple_data_series_based_on_timestamps([gt_data, ekfpos_data, dnncv_data])
    plotter.plot_multiple_data_series(
        synced_gt_dnncv_ekf_data, 3, "Position - GT vs. EKF vs. DNNCV raw",
        ["GT", "EKF", "DNNCV"], ["t [sec]", "t [sec]", "t [sec]"], ["x[m]", "y[m]", "z[m]"],
        [False, True, True]
    )

    plt.show()

if __name__ == "__main__":
    main()