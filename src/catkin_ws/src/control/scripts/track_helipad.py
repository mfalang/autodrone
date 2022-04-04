#!/usr/bin/env python3

import numpy as np

import control
import guidance
import control_util

import rospy
import perception.msg
import drone_interface.msg
import ground_truth.msg

class Tracker():

    def __init__(self):
        node_name = "helipad_tracker"
        rospy.init_node(node_name)
        control_params = control_util.load_control_params_config(node_name)

        self._controller = control.Controller(control_params)

        self._guidance_law_type = rospy.get_param("~guidance_law")
        rospy.loginfo(f"Tracker started with guidance law: {self._guidance_law_type}")
        guidance_law_params = control_params["guidance"][self._guidance_law_type]
        velocity_limits = control_params["guidance"]["velocity_limits"]
        guidance_law = guidance.get_guidance_law(self._guidance_law_type)
        self._guidance_law = guidance_law(guidance_law_params, velocity_limits)

        self._prev_telemetry_timestamp: float = None
        self._prev_atttiude: np.ndarray = None # roll and pitch
        self._prev_velocity: np.ndarray = None # vx and vy

        self._prev_pos_timestamp: float = None
        self._prev_pos: np.ndarray = None

        rospy.Subscriber("/drone/out/telemetry", drone_interface.msg.AnafiTelemetry, self._drone_telemetry_cb)
        rospy.Subscriber("/estimate/ekf", perception.msg.PointWithCovarianceStamped, self._ekf_cb)
        # rospy.Subscriber("/ground_truth/body_frame/helipad_pose", ground_truth.msg.PoseStampedEuler, self._gt_position_cb)

    def _drone_telemetry_cb(self, msg: drone_interface.msg.AnafiTelemetry) -> None:

        self._prev_telemetry_timestamp = msg.header.stamp.to_sec()

        self._prev_atttiude = np.array([
            msg.roll,
            msg.pitch
        ])

        self._prev_velocity = np.array([
            msg.vx,
            msg.vy
        ])

    def _ekf_cb(self, msg: perception.msg.PointWithCovarianceStamped) -> None:

        self._prev_pos_timestamp = msg.header.stamp.to_sec()

        self._prev_pos = np.array([
            msg.position.x,
            msg.position.y
        ])

    def _gt_position_cb(self, msg: ground_truth.msg.PoseStampedEuler) -> None:

        self._prev_pos_timestamp = msg.header.stamp.to_sec()

        self._prev_pos = np.array([
            msg.x,
            msg.y
        ])

    def start(self, debug=False):

        self._controller.takeoff()

        control_util.await_user_confirmation("Move up 1m")
        self._controller.move_relative(0, 0, -1, 0)

        control_util.await_user_confirmation("Move away from the helipad")
        self._controller.move_relative(-1, -1, 0, 0)

        control_util.await_user_confirmation("Start tracking")

        rate = rospy.Rate(20)
        dt = 0.05
        v_d = np.zeros(4)

        n_sec_to_save = 100
        n_entries = n_sec_to_save * 20
        self._vrefs = np.zeros((2, n_entries))
        self._vds = np.zeros_like(self._vrefs)
        self._v_meas = np.zeros_like(self._vrefs)
        self._att_meas = np.zeros_like(self._vrefs)
        self._att_refs = np.zeros_like(self._vrefs)
        self._time_refs = np.zeros(self._vrefs.shape[1])
        self._time_meas = np.zeros(self._vrefs.shape[1])
        self._counter = 0

        rospy.on_shutdown(self._shutdown)


        while not rospy.is_shutdown():
            v_ref = self._guidance_law.get_velocity_reference(self._prev_pos, self._prev_pos_timestamp, debug=True)
            v_d = self._controller.get_reference(v_d, v_ref, dt)

            att_ref = self._controller.set_attitude(
                v_d, self._prev_velocity, self._prev_telemetry_timestamp
            )

            if self._counter < n_entries:
                self._vrefs[:, self._counter] = v_ref
                self._vds[:, self._counter] = v_d[:2]
                self._att_refs[:, self._counter] = att_ref
                self._time_refs[ self._counter] = rospy.Time.now().to_sec()
                self._v_meas[:, self._counter] = self._prev_velocity
                self._att_meas[:, self._counter] = self._prev_atttiude
                self._time_meas[ self._counter] = self._prev_telemetry_timestamp

                self._counter += 1

            if debug:
                print(f"Pos error: x: \t{self._prev_pos[0]:.3f} y: \t{self._prev_pos[1]:.3f}")
                print(f"Vref: vx: \t{v_ref[0]:.3f} vy: \t{v_ref[1]:.3f}")
                print(f"Vd: x: \t\t{v_d[0]:.3f} y: \t{v_d[1]:.3f}")
                print(f"Attref: r: \t{att_ref[0]:.3f} p: \t{att_ref[1]:.3f}")
                print()

            rate.sleep()

    def _shutdown(self):
        self._controller.land()

        output_dir = "/home/martin/code/autodrone/out/temp_guidance_ouput"
        print(f"Saving output data to: {output_dir}")
        np.savetxt(f"{output_dir}/vrefs.txt", self._vrefs[:, :self._counter])
        np.savetxt(f"{output_dir}/vds.txt", self._vds[:, :self._counter])
        np.savetxt(f"{output_dir}/v_meas.txt", self._v_meas[:, :self._counter])
        np.savetxt(f"{output_dir}/att_refs.txt", self._att_refs[:, :self._counter])
        np.savetxt(f"{output_dir}/att_meas.txt", self._att_meas[:, :self._counter])
        np.savetxt(f"{output_dir}/time_refs.txt", self._time_refs[:self._counter])
        np.savetxt(f"{output_dir}/time_meas.txt", self._time_meas[:self._counter])

    def _plot_output(self):
        output_dir = "/home/martin/code/autodrone/out/temp_guidance_ouput"
        v_ref = np.loadtxt(f"{output_dir}/vrefs.txt")
        v_d = np.loadtxt(f"{output_dir}/vds.txt")
        v_meas = np.loadtxt(f"{output_dir}/v_meas.txt")
        t_refs = np.loadtxt(f"{output_dir}/time_refs.txt")
        t_meas = np.loadtxt(f"{output_dir}/time_meas.txt")
        att_refs = np.loadtxt(f"{output_dir}/att_refs.txt")
        att_meas = np.loadtxt(f"{output_dir}/att_meas.txt")

        control_util.plot_drone_velocity_vs_reference_trajectory(
            v_ref, v_d, t_refs, v_meas, t_meas, start_time_from_0=True
        )

        control_util.plot_drone_attitude_vs_reference(
            att_refs, t_refs, att_meas, t_meas, start_time_from_0=True, show_plot=True
        )

def main():
    tracker = Tracker()
    # tracker.start(debug=False)
    tracker._plot_output()

if __name__ == "__main__":
    main()