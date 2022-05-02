#!/usr/bin/env python3

import rospy
import numpy as np

import control
import guidance
import control_util

import perception.msg
import drone_interface.msg

class MissionController():

    def __init__(self) -> None:
        node_name = "mission_control"
        rospy.init_node(node_name, anonymous=False)
        control_params = control_util.load_control_params_config(node_name)

        self._controller = control.Controller(control_params)

        self._guidance_law_type = rospy.get_param("~guidance_law")
        rospy.loginfo(f"Using guidance law: {self._guidance_law_type}")
        guidance_law_params = control_params["guidance"][self._guidance_law_type]
        velocity_limits = control_params["guidance"]["velocity_limits"]
        guidance_law = guidance.get_guidance_law(self._guidance_law_type)
        self._guidance_law = guidance_law(guidance_law_params, velocity_limits)

        self._prev_telemetry_timestamp: float = None
        self._prev_telemetry: drone_interface.msg.AnafiTelemetry = None
        self._new_telemetry_available: bool = False
        self._prev_atttiude: np.ndarray = None # roll and pitch
        self._prev_velocity: np.ndarray = None # vx and vy

        self._prev_pos_timestamp: float = None
        self._prev_pos: np.ndarray = None

        rospy.Subscriber("/drone/out/telemetry", drone_interface.msg.AnafiTelemetry, self._drone_telemetry_cb)
        rospy.Subscriber("/estimate/ekf", perception.msg.PointWithCovarianceStamped, self._ekf_cb)

    def _drone_telemetry_cb(self, msg: drone_interface.msg.AnafiTelemetry) -> None:
        self._prev_telemetry_timestamp = msg.header.stamp.to_sec()
        self._prev_telemetry = msg
        self._new_telemetry_available = True

    def _ekf_cb(self, msg: perception.msg.PointWithCovarianceStamped) -> None:

        self._prev_pos_timestamp = msg.header.stamp.to_sec()

        self._prev_pos = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])

    def _wait_for_hovering(self):
        rospy.loginfo("Waiting for drone to hover")
        flying_state = ""
        while flying_state != "hovering":
            if self._new_telemetry_available:
                flying_state = self._prev_telemetry.flying_state
                self._new_telemetry_available = False
            rospy.sleep(0.1)
        rospy.loginfo("Hovering")

    def _get_reliable_altitude_estimate(self):
        # Use EKF if altitude is above 1m
        if self._prev_pos[2] > 1:
            print("Using EKF")
            return self._prev_pos[2]
        else:
            print("Using ultrasonic sensor")
            return -self._prev_telemetry.relative_altitude # negative to get it in the BODY frame

    def start(self):

        control_util.await_user_confirmation("Start mission")

        self.takeoff()
        control_util.await_user_confirmation("Continue to tracking")
        self.track_helipad()
        self.land()


    def takeoff(self):
        # Take off and wait for drone to be stable in the air
        self._controller.takeoff()
        self._wait_for_hovering()

        # Move up to a total of 3m altitude
        rospy.loginfo("Moving up 2m")
        self._controller.move_relative(0, 0, -2, 0)
        self._wait_for_hovering()

    def land(self):
        # Assuming that the altitude above the helipad is about 0.5m (done by the tracking
        # helipad action) and therefore we can just execute the landing here.
        self._controller.land()

    def move(self, whereto: np.ndarray, use_gps_coordinates=False):
        # use_gps_coordinates should only be set to true in the simulator and if used in real
        # life one must be very careful to actually select the correct GPS location.

        if not use_gps_coordinates:
            self._controller.move_relative(whereto[0], whereto[1], whereto[2])
        else:
            print("GPS coordinates not implemented.")

        self._wait_for_hovering()

    def track_helipad(self):
        rate = rospy.Rate(20)
        dt = 0.05
        v_d = np.zeros(4)

        pos_error_threshold = 0.2 # m

        control_util.await_user_confirmation("Move away from the helipad")
        self._controller.move_relative(-1, -1, 0, 0)
        control_util.await_user_confirmation("Start tracking")

        # First align the drone with the helipad horizontally
        rospy.loginfo("Aligning horizontally")
        descending = False
        landing_position_ref = np.array([0, 0, 0.5]) # in body frame
        while not rospy.is_shutdown():

            if np.linalg.norm(self._prev_pos[:2]) < pos_error_threshold:
                descending = True

            if descending:
                alt = self._get_reliable_altitude_estimate()
                alt_error = alt - landing_position_ref[2]
                # Sign of position errro in z must be switched as positive climb rate is defined as upwards
                # in the drone interface, but since these measurements are in BODY, being above the desired
                # altitude will result in a positive error, hence this error must be made negative to work with
                # the control
                alt_error *= -1

                pos_error = np.hstack((self._prev_pos[:2], alt_error))

                if np.abs(pos_error[2]) < 0.1:
                    break
            else:
                pos_error = np.hstack((self._prev_pos[:2], 0))

            v_ref = self._guidance_law.get_velocity_reference(pos_error, self._prev_pos_timestamp, debug=True)
            v_d = self._controller.get_smooth_reference(v_d, v_ref[:2], dt)

            prev_vel = np.array([
                self._prev_telemetry.vx,
                self._prev_telemetry.vy,
                self._prev_telemetry.vz
            ])

            vd_3D = np.hstack((v_d[:2], v_ref[2]))
            self._controller.set_attitude3D(
                vd_3D, prev_vel, self._prev_telemetry_timestamp
            )

            rate.sleep()

        rospy.loginfo("Ready to land")

def main():
    mission_controller = MissionController()
    mission_controller.start()

if __name__ == "__main__":
    main()