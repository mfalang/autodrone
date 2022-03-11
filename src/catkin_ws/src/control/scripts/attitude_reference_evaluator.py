#!/usr/bin/env python3

# This file is used to evaluate the different methods in attitude_reference_generator.py
# to determine which is most suited

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from attitude_reference_generator import PIDReferenceGenerator

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


# Calculate output from PID method
time = gt_data[:,0] - gt_data[0,0]
v_ref = gt_data[:,1:3]

att_target = gt_data[:,4:6]

att_ref = np.zeros_like(v_ref)

ref_generator = PIDReferenceGenerator(1,2,3)

for i in range(len(v_ref)):
    # x = ref_generator.get_attitude_reference(v_ref)
    pass

plt.show()