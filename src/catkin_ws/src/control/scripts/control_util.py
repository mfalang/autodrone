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

def generate_random_reference_trajectory(duration: int) -> np.ndarray:
    """Get a random horizontal trajectory lasting in a given amount of seconds.

    Parameters
    ----------
    duration : int
        Seconds the trajectory lasts

    Returns
    -------
    np.ndarray
        Reference trajectory. Format: [vx_ref, vy_ref] in m/s
    """
    v_ref = np.zeros((2, duration * 20))
    print(v_ref.shape)
    v_ref_i = np.random.uniform(0.2,0.5, [2,1])

    v_ref[:,:-1] = v_ref_i # leave last reference to 0

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
    v: np.ndarray, ts_meas: np.ndarray,
    plot_title="Velocity vs. reference trajectory", start_time_from_0=False,
    show_plot=False, save_fig=False
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

    # LaTex must be installed for this to work
    # sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(2, 1, sharex=True)

    fig.suptitle(plot_title)

    if start_time_from_0:
        ts_refs -= ts_refs[0]
        ts_meas -= ts_meas[0]

    # Vx
    ax[0].plot(ts_refs, v_ref[0,:], label=r"$v_{r,x}$")
    ax[0].plot(ts_refs, v_d[0,:], label=r"$v_{d,x}$")
    ax[0].plot(ts_meas, v[0,:], label=r"$v_x$")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel(r"X-axis velocity [m/s]")

    # Vy
    ax[1].plot(ts_refs, v_ref[1,:], label=r"$v_{r,y}$")
    ax[1].plot(ts_refs, v_d[1,:], label=r"$v_{d,y}$")
    ax[1].plot(ts_meas, v[1,:], label=r"$v_y$")
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel(r"Y-axis velocity [m/s]")
    ax[1].set_xlabel(r"Time [sec]")

    if save_fig:
        plt.savefig("veloctiy_vs_reference_trajectory.png", dpi=300)

    if show_plot:
        plt.show()

def plot_drone_attitude_vs_reference(
    att_ref: np.ndarray, ts_refs: np.ndarray,
    att_meas: np.ndarray, ts_meas: np.ndarray, plot_title="Reference vs. measured roll and pitch angles",
    start_time_from_0=False, show_plot=False, save_fig=False
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

    # LaTex must be installed for this to work
    # sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(2, 1, sharex=True)

    fig.suptitle(plot_title)

    if start_time_from_0:
        ts_refs -= ts_refs[0]
        ts_meas -= ts_meas[0]

    # Pitch
    ax[0].plot(ts_refs, att_ref[1,:], label=r"$\theta_r$")
    ax[0].plot(ts_meas, att_meas[1,:], label=r"$\theta$")
    ax[0].set_ylabel(r"Pitch angle [deg]")
    ax[0].legend()

    # Roll
    ax[1].plot(ts_refs, att_ref[0,:], label=r"$\phi_r$")
    ax[1].plot(ts_meas, att_meas[0,:], label=r"$\phi$")
    ax[1].set_ylabel(r"Roll angle [deg]")
    ax[1].set_xlabel(r"Time [sec]")
    ax[1].legend()

    if save_fig:
        plt.savefig("attitude_vs_reference.png", dpi=300)

    if show_plot:
        plt.show()

def plot_drone_position_vs_reference(
    pos_ref: np.ndarray, ts_refs: np.ndarray,
    pos_meas: np.ndarray, ts_meas: np.ndarray, start_time_from_0=False, show_plot=False
):
    """Plot a position reference vs. the measured position.

    Parameters
    ----------
    pos_ref : np.ndarray
        Reference position. Format (body frame): [x, y] or [x, y, z]
    ts_refs : np.ndarray
        ROS timestamps for pos_ref
    pos_meas : np.ndarray
        Measured position. Format (body frame): [x, y] or [x, y, z]
    ts_meas : np.ndarray
        ROS timestamps for pos_meas
    start_time_from_0 : bool, optional
        Set to true to show seconds on the x-axis and not the actual timestamps.
        This could mess up the scales since the refs and measurements use different
        time series. By default False
    show_plot : bool, optional
        Set to true to run plt.show() to show the plot. This will block, by default False
    """
    sns.set()

    if start_time_from_0:
        ts_refs -= ts_refs[0]
        ts_meas -= ts_meas[0]

    if pos_ref.shape[0] == 2:
        fig, ax = plt.subplots(2, 1, sharex=True)

        fig.suptitle("Reference vs. measured horizontal position")

        # x-axis
        ax[0].plot(ts_refs, pos_ref[0,:], label="x_ref")
        ax[0].plot(ts_meas, pos_meas[0,:], label="x")
        ax[0].legend()
        ax[0].set_title("X")

        # y-axis
        ax[1].plot(ts_refs, pos_ref[1,:], label="y_ref")
        ax[1].plot(ts_meas, pos_meas[1,:], label="y")
        ax[1].legend()
        ax[1].set_title("Y")

    else:
        fig, ax = plt.subplots(3, 1, sharex=True)

        fig.suptitle("Reference vs. measured position")

        # x-axis
        ax[0].plot(ts_refs, pos_ref[0,:], label="x_ref")
        ax[0].plot(ts_meas, pos_meas[0,:], label="x")
        ax[0].legend()
        ax[0].set_title("X")

        # y-axis
        ax[1].plot(ts_refs, pos_ref[1,:], label="y_ref")
        ax[1].plot(ts_meas, pos_meas[1,:], label="y")
        ax[1].legend()
        ax[1].set_title("Y")

        # z-axis
        ax[2].plot(ts_refs, pos_ref[2,:], label="z_ref")
        ax[2].plot(ts_meas, pos_meas[2,:], label="z")
        ax[2].legend()
        ax[2].set_title("Z")

    if show_plot:
        plt.show()

def plot_drone_position_error_vs_gt(
    pos_error: np.ndarray, ts_pos_error: np.ndarray, gt_pos_error: np.ndarray,
    ts_gt_pos_error: np.ndarray, plot_title="Reference vs. measured horizontal position",
    start_time_from_0=False, show_plot=False, save_fig=False
):
    """Plot a position error.

    Parameters
    ----------
    pos_error : np.ndarray
        Position error. Format (body frame): [x, y] or [x, y, z]
    ts_error : np.ndarray
        ROS timestamps for pos_error
    gt_pos_error : np.ndarray
        GT position error. Format (body frame): [x, y] or [x, y, z]
    ts_gt : np.ndarray
        ROS timestamps for gt_pos_error
    start_time_from_0 : bool, optional
        Set to true to show seconds on the x-axis and not the actual timestamps.
        This could mess up the scales since the refs and measurements use different
        time series. By default False
    show_plot : bool, optional
        Set to true to run plt.show() to show the plot. This will block, by default False
    """
    sns.set()

    # LaTex must be installed for this to work
    # sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
    plt.rcParams['text.usetex'] = True

    if start_time_from_0:
        ts_pos_error -= ts_pos_error[0]
        ts_gt_pos_error -= ts_gt_pos_error[0]

    zero = np.zeros_like(ts_pos_error)

    if pos_error.shape[0] == 2:
        fig, ax = plt.subplots(2, 1, sharex=True)

        fig.suptitle(plot_title)

        # x-axis
        ax[0].plot(ts_pos_error, zero, linestyle="--", label=r"$e_{r,x}$")
        ax[0].plot(ts_gt_pos_error, gt_pos_error[0,:], label=r"$e_x$")
        ax[0].plot(ts_pos_error, pos_error[0,:], label=r"$\hat e_x$")
        ax[0].set_ylabel(r"X-axis position [m]")
        ax[0].legend()

        # y-axis
        ax[1].plot(ts_pos_error, zero, linestyle="--", label=r"$e_{r,y}$")
        ax[1].plot(ts_gt_pos_error, gt_pos_error[1,:], label=r"$e_y$")
        ax[1].plot(ts_pos_error, pos_error[1,:], label=r"$\hat e_y$")
        ax[1].set_ylabel(r"Y-axis position [m]")
        ax[1].set_xlabel(r"Time [sec]")
        ax[1].legend()

    else:
        fig, ax = plt.subplots(3, 1, sharex=True)

        fig.suptitle(plot_title)

        # x-axis
        ax[0].plot(ts_gt_pos_error, gt_pos_error[0,:], label=r"$e_x$")
        ax[0].plot(ts_pos_error, pos_error[0,:], label=r"$\hat e_x$")
        ax[0].set_ylabel(r"X-axis position [m]")
        ax[0].legend()

        # y-axis
        ax[1].plot(ts_gt_pos_error, gt_pos_error[1,:], label=r"$e_y$")
        ax[1].plot(ts_pos_error, pos_error[1,:], label=r"$\hat e_y$")
        ax[1].set_ylabel(r"Y-axis position [m]")
        ax[1].legend()

        # z-axis
        ax[2].plot(ts_gt_pos_error, gt_pos_error[2,:], label=r"$e_z$")
        ax[2].plot(ts_pos_error, pos_error[2,:], label=r"$\hat e_z$")
        ax[2].set_ylabel(r"Z-axis position [m]")
        ax[2].set_xlabel(r"Time [sec]")
        ax[2].legend()

    if save_fig:
        plt.savefig("pos_error_gt_vs_est.png", dpi=300)

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

def load_config(node_name: str, rosparam_name: str) -> dict:
    """Load the control parameters when the controller is run from an arbitrary node.

    Parameters
    ----------
    node_name : str
        The name of the node the starting the controller
    rosparam_name : str
        The name of the ROS parameter where the filename of the config file is stored

    Returns
    -------
    dict
        The control parameters config
    """
    config_file = rospy.get_param(f"/{node_name}/{rosparam_name}")
    script_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        with open(f"{script_dir}/../config/{config_file}") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        rospy.logerr(f"Failed to load config: {e}")
        sys.exit()

    return config