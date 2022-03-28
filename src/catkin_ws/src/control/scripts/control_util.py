# Standalone functions useful for use in different control scripts

import os
import sys
import yaml

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import rospy
import drone_interface.msg

def pack_attitude_ref_msg_horizontal(att_ref: np.ndarray) -> drone_interface.msg.AttitudeSetpoint:
    """Helper function for creating an atttiude setpoint message when only horizontal motion is
    desired.

    Parameters
    ----------
    att_ref : np.ndarray
        Attitude setpoint. Format: [roll, pitch] both in degrees.

    Returns
    -------
    drone_interface.msg.AttitudeSetpoint
        ROS message with the attitude setpoint
    """

    att_ref = np.array([att_ref[0], att_ref[1], 0, 0])
    return pack_attitude_ref_msg(att_ref)


def pack_attitude_ref_msg(att_ref: np.ndarray) -> drone_interface.msg.AttitudeSetpoint:
    """Generate a attitude setpoint message attitude setpoints.

    Parameters
    ----------
    att_ref : np.ndarray
        Attitude setpoint. Format: [roll, pitch, yaw_rate, climb_rate]
        All angles in degrees, climb_rate positive up.

    Returns
    -------
    drone_interface.msg.AttitudeSetpoint
        ROS message with the attitude setpoint
    """

    msg = drone_interface.msg.AttitudeSetpoint()
    msg.header.stamp = rospy.Time.now()
    msg.roll = att_ref[0]
    msg.pitch = att_ref[1]
    msg.yaw_rate = att_ref[2]
    msg.climb_rate = att_ref[3]

    return msg

def get_reference_trajectory_sim() -> np.ndarray:
    """For SIM only (too far for dronelab)! Trajectory where the drone flies first forward and
    backwards, then right and left, and then across forward/to the right and back across backwards
    and to the left.

    Returns
    -------
    np.ndarray
        Reference trajectory. Format: [vx_ref, vy_ref] in m/s
    """

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

def get_reference_trajectory_safe() -> np.ndarray:
    """Safe for dronelab and simulation. Same trajectory as in get_reference_trajectory_sim, but
    the distances travelled are smaller meaning the drone can safely fly this trajectory in the
    dronelab.

    Returns
    -------
    np.ndarray
        Reference trajectory. Format: [vx_ref, vy_ref] in m/s
    """
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

def await_user_confirmation(msg: str):
    """Await user confirmation to continue.

    Parameters
    ----------
    msg : str
        What to await confirmation for
    """
    ans = input(f"Press Enter to: {msg}, type [q] to quit ")
    if ans != "":
        sys.exit(0)

def plot_drone_velocity_vs_reference_trajectory(
    v_ref: np.ndarray, v_d: np.ndarray, ts_refs: np.ndarray,
    v: np.ndarray, ts_meas: np.ndarray, start_time_from_0=False, show_plot=False
):
    """Plot a veloctiy reference trajectory vs. the actual measured velocity.

    Parameters
    ----------
    v_ref : np.ndarray
        Raw velocity reference. Format: [vx_ref, vy_ref] in m/s
    v_d : np.ndarray
        Velocity reference from reference model. Format: [vx_d, vy_d] in m/s
    ts_refs : np.ndarray
        ROS timestamps for v_ref and v_d
    v : np.ndarray
        Velocity measurements
    ts_meas : np.ndarray
        ROS timestamps for v in seconds
    start_time_from_0 : bool, optional
        Set to true to show seconds on the x-axis and not the actual timestamps.
        This could mess up the scales since the refs and measurements use different
        time series. By default False
    show_plot : bool, optional
        Set to true to run plt.show() to show the plot. This will block, by default False
    """
    sns.set()
    fig, ax = plt.subplots(2, 1, sharex=True)

    fig.suptitle("Reference vs. measured horizontal velocities")

    if start_time_from_0:
        ts_refs -= ts_refs[0]
        ts_meas -= ts_meas[0]

    # Vx
    ax[0].plot(ts_refs, v_ref[0,:], label="vx_ref")
    ax[0].plot(ts_refs, v_d[0,:], label="vd_x")
    ax[0].plot(ts_meas, v[0,:], label="vx")
    ax[0].set_title("X-axis")

    # Vy
    ax[1].plot(ts_refs, v_ref[1,:], label="vy_ref")
    ax[1].plot(ts_refs, v_d[1,:], label="vd_y")
    ax[1].plot(ts_meas, v[1,:], label="vy")
    ax[1].set_title("Y-axis")

    if show_plot:
        plt.show()

def plot_drone_attitude_vs_reference(
    att_ref: np.ndarray, ts_refs: np.ndarray,
    att_meas: np.ndarray, ts_meas: np.ndarray, start_time_from_0=False, show_plot=False
):
    """Plot a roll and pitch reference angles vs. the measured angles.

    Parameters
    ----------
    att_ref : np.ndarray
        Reference angles. Format: [roll, pitch]
    ts_refs : np.ndarray
        ROS timestamps for att_ref
    att_meas : np.ndarray
        Measured angles. Format: [roll, pitch]
    ts_meas : np.ndarray
        ROS timestamps for att_meas
    start_time_from_0 : bool, optional
        Set to true to show seconds on the x-axis and not the actual timestamps.
        This could mess up the scales since the refs and measurements use different
        time series. By default False
    show_plot : bool, optional
        Set to true to run plt.show() to show the plot. This will block, by default False
    """
    sns.set()
    fig, ax = plt.subplots(2, 1, sharex=True)

    fig.suptitle("Reference vs. measured roll and pitch angles")

    if start_time_from_0:
        ts_refs -= ts_refs[0]
        ts_meas -= ts_meas[0]

    # Pitch
    ax[0].plot(ts_refs, att_ref[1,:], label="pitch_ref")
    ax[0].plot(ts_meas, att_meas[1,:], label="pitch")
    ax[0].set_title("Pitch")

    # Roll
    ax[1].plot(ts_refs, att_ref[0,:], label="roll_ref")
    ax[1].plot(ts_meas, att_meas[0,:], label="roll")
    ax[1].set_title("Roll")

    if show_plot:
        plt.show()


def load_control_params_config(node_name: str) -> dict:
    """Load the control parameters when the controller is run from an arbitrary node.

    Parameters
    ----------
    node_name : str
        The name of the node the starting the controller

    Returns
    -------
    dict
        The control parameters config
    """
    config_file = rospy.get_param(f"/{node_name}/config_file")
    script_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        with open(f"{script_dir}/../config/{config_file}") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        rospy.logerr(f"Failed to load config: {e}")
        sys.exit()

    return config
