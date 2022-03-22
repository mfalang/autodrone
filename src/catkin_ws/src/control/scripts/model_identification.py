import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import least_squares

def pitch_est(z, acc_x, vx, g=9.81, m=315/1000):
    return np.rad2deg(np.arctan(-(acc_x / g + (z * vx) / (m * g))))

def roll_est(z, acc_y, vy, g=9.81, m=315/1000):
    return np.rad2deg(np.arctan(acc_y / g + (z * vy) / (m * g)))

def residual(z, gt, est_fun, acc, v):
    return est_fun(z, acc, v) - gt

def get_attitude_refs_from_model(acc_x, acc_y, ax, bx, ay, vx, vy):

    g = 9.81
    m = 315 / 1000

    pitch_ref = np.rad2deg(np.arctan(-(acc_x / g + (ax * vx + bx) / (m * g))))
    roll_ref = np.rad2deg(np.arctan(acc_y / g + ay * vy / (m * g)))

    return roll_ref, pitch_ref

def get_drag_force(roll, pitch, acc_x, acc_y, g=9.81, m=315/1000):

    drag_x = m * (g*np.tan(np.deg2rad(pitch)) - acc_x)
    drag_y = m * (g*np.tan(np.deg2rad(roll)) - acc_y)

    return drag_x, drag_y

def plot():
    pitch_only_folder = "2022-3-22/9-1-2"
    roll_and_pitch_folder = "2022-3-22/13-29-42"
    roll_only_folder = "2022-3-22/13-36-30"

    data_folder = roll_and_pitch_folder

    script_dir = os.path.dirname(os.path.realpath(__file__))
    gt_filename = f"{script_dir}/../../../../../out/{data_folder}/ground_truths/drone_velocity_body_frame_and_attitude.txt"
    gt_data = np.loadtxt(gt_filename, skiprows=1)

    time = gt_data[:,0] - gt_data[0,0]
    roll = gt_data[:-1,4]
    pitch = gt_data[:-1,5]
    dt = np.diff(time)
    vel = gt_data[:,1:3]
    vx = vel[:-1,0]
    vy = vel[:-1,1]
    dv = np.diff(vel, axis=0)

    acc_x = dv[:,0] / dt
    acc_x = savgol_filter(acc_x, 101, 3)
    acc_y = dv[:,1] / dt
    acc_y = savgol_filter(acc_y, 101, 3)


    df = pd.read_csv(gt_filename, sep=" ", skiprows=1)
    df.columns = ["timestamp", "vx", "vy", "vz", "roll", "pitch", "yaw"]
    df["time"] = df["timestamp"] - df["timestamp"][0]
    df["acc_x"] = acc_x
    df["acc_y"] = acc_y

    # Generate data from the model
    # r_ref, p_ref = get_attitude_refs_from_model(acc_x, acc_y, 0.1, -0.05, 0.1, vx[:-1], vy[:-1])
    z0_pitch = 0.1
    pitch_res = least_squares(residual, z0_pitch, args=[pitch, pitch_est, acc_x, vx])
    z_pitch = pitch_res.x
    print(pitch_res)

    z0_roll = 0.1
    roll_res = least_squares(residual, z0_roll, args=[roll, roll_est, acc_y, vy])
    z_roll = roll_res.x
    print(roll_res)

    r_ref = roll_est(z_roll, acc_y, vy)
    p_ref = pitch_est(z_pitch, acc_x, vx)

    df["roll_ref"] = r_ref
    df["pitch_ref"] = p_ref

    # drag_x, drag_y = get_drag_force(roll[:-1], pitch[:-1], acc_x, acc_y)

    # df["drag_x"] = drag_x
    # df["drag_y"] = drag_y

    sns.set()

    # Plot ground truth and drone telemetry velocity and attitude
    fig, ax = plt.subplots(3, 2, sharex=True)
    sns.lineplot(ax=ax[0,0], data=df, x="time", y="vx", label="vx")
    sns.lineplot(ax=ax[1,0], data=df, x="time", y="acc_x", label="ax")
    sns.lineplot(ax=ax[2,0], data=df, x="time", y="pitch", label="gt")
    sns.lineplot(ax=ax[2,0], data=df, x="time", y="pitch_ref", label="ref")
    sns.lineplot(ax=ax[0,1], data=df, x="time", y="vy", label="vy")
    sns.lineplot(ax=ax[1,1], data=df, x="time", y="acc_y", label="ay")
    sns.lineplot(ax=ax[2,1], data=df, x="time", y="roll", label="gt")
    sns.lineplot(ax=ax[2,1], data=df, x="time", y="roll_ref", label="ref")
    # sns.lineplot(ax=ax[3,0], data=df, x="time", y="drag_x", label="drag_x")
    # sns.lineplot(ax=ax[3,1], data=df, x="time", y="drag_y", label="drag_y")

    ax[0,0].legend()
    ax[1,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()
    ax[2,0].legend()
    ax[2,1].legend()
    fig.suptitle("Ground truth data")

    plt.show()


if __name__ == "__main__":
    plot()