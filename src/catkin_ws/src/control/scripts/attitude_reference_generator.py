# This file holds the methods for converting a velocity reference to an attitude
# reference that can be forwarded to the Anafi's internal control system.

import numpy as np

class VelocityReferenceModel():

    def __init__(self, omegas: tuple, zetas: tuple):
        self._w_x = omegas[0]
        self._w_y = omegas[1]

        self._zeta_x = zetas[0]
        self._zeta_y = zetas[1]

        self._Ad = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-self._w_x**2, 0, -2*self._zeta_x*self._w_x, 0],
            [0, -self._w_y**2, 0, -2*self._zeta_y*self._w_y]
        ])

        self._Bd = np.array([
            [0, 0],
            [0, 0],
            [self._w_x**2, 0],
            [0, self._w_y**2]
        ])

    def get_filtered_reference(self, xd_prev: np.ndarray, v_ref_raw: np.ndarray, dt: float):
        r = v_ref_raw

        xd_dot = self._Ad @ xd_prev + self._Bd @ r

        xd_next = xd_prev + dt * xd_dot

        return xd_next

class GenericAttitudeReferenceGenerator():

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float):
        raise NotImplementedError

    def _clamp(self, value: float, limits: tuple):
        if value < limits[0]:
            return limits[0]
        elif value > limits[1]:
            return limits[1]
        else:
            return value

class PIDReferenceGenerator(GenericAttitudeReferenceGenerator):

    def __init__(self, roll_Kp_Ki_Kd: tuple, pitch_Kp_Ki_Kd: tuple,
        roll_min_max_angles: tuple, pitch_min_max_angles: tuple):

        self._Kp_roll = roll_Kp_Ki_Kd[0]
        self._Ki_roll = roll_Kp_Ki_Kd[1]
        self._Kd_roll = roll_Kp_Ki_Kd[2]

        self._Kp_pitch = pitch_Kp_Ki_Kd[0]
        self._Ki_pitch = pitch_Kp_Ki_Kd[1]
        self._Kd_pitch = pitch_Kp_Ki_Kd[2]

        self._roll_min = roll_min_max_angles[0]
        self._roll_max = roll_min_max_angles[1]
        self._pitch_min = pitch_min_max_angles[0]
        self._pitch_max = pitch_min_max_angles[1]

        self._prev_timestamp = None
        self._error_integral = np.zeros(2)
        self._prev_error = np.zeros(2)

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float, debug=False):

        error = v_ref - v_actual

        error_x = error[0]
        error_y = error[1]

        gain_x = self._Kp_pitch * error_x
        gain_y = self._Kp_roll * error_y

        if self._prev_timestamp is not None and timestamp != self._prev_timestamp:
            dt = timestamp - self._prev_timestamp

            derivative_x = self._Kd_pitch * (error_x - self._prev_error[0]) / dt
            derivative_y = self._Kd_roll * (error_y - self._prev_error[1]) / dt

            self._prev_error = error

            # Avoid integral windup
            if self._pitch_min <= self._error_integral[0] <= self._pitch_max:
                self._error_integral[0] += self._Ki_pitch * error_x * dt # should probably not add Ki*e, only e

            if self._roll_min <= self._error_integral[1] <= self._roll_max:
                self._error_integral[1] += self._Ki_roll * error_y * dt # should probably not add Ki*e, only e


        else:
            derivative_x = derivative_y = 0
            self._prev_timestamp = timestamp # this should be done outside of the if/else
            # as it stands now it is only updated if the previous timestamp is not available or equal

        if debug:
            print(f"Pitch gains:\tP: {gain_x:.3f}\tI: {self._error_integral[0]:.3f}\tD: {derivative_x:.3f} ")
            print(f"Roll gains:\tP: {gain_y:.3f}\tI: {self._error_integral[1]:.3f}\tD: {derivative_y:.3f} ")
            print()

        pitch_reference = gain_x + derivative_x + self._error_integral[0]
        roll_reference = gain_y + derivative_y + self._error_integral[1]

        attitude_reference = np.array([roll_reference, pitch_reference])

        return attitude_reference

class LinearDragModelReferenceGenerator(GenericAttitudeReferenceGenerator):

    def __init__(self, drone_mass: float, d_x: float, d_y: float, gravity=9.81):

        self._m = drone_mass
        self._g = gravity

        self._d_x = d_x
        self._d_y = d_y

        self._prev_timestamp: float = None
        self._prev_roll_ref: float = None
        self._prev_pitch_ref: float = None

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float, debug=False):
        vx = v_actual[0]
        vy = v_actual[1]

        vx_ref = v_ref[0]
        vy_ref = v_ref[1]

        accel_x_desired = vx_ref - vx
        accel_y_desired = vy_ref - vy

        # Negative on x axis due to inverse relationship between pitch angle and x-velocity
        pitch_ref = np.rad2deg(np.arctan(-(accel_x_desired / self._g + (self._d_x * vx) / (self._m * self._g))))
        roll_ref = np.rad2deg(np.arctan(accel_y_desired / self._g + (self._d_y * vy) / (self._m * self._g)))

        if debug:
            print(f"ts:{timestamp}\tRefs: R: {roll_ref:.3f}\tP: {pitch_ref:.3f}\tax_des: {accel_x_desired:.3f}\tay_des: {accel_y_desired:.3f}")
            print()

        return np.array([roll_ref, pitch_ref])

class iPReferenceGenerator(GenericAttitudeReferenceGenerator):

    def __init__(self, roll_kp_kd: tuple, pitch_kp_kd: tuple, alpha_roll, alpha_pitch,
        roll_limits: tuple, pitch_limits: tuple
    ):
        self._Kp_roll = roll_kp_kd[0]
        self._Ki_roll = roll_kp_kd[1]
        self._Kd_roll = roll_kp_kd[2]

        self._Kp_pitch = pitch_kp_kd[0]
        self._Ki_pitch = pitch_kp_kd[1]
        self._Kd_pitch = pitch_kp_kd[2]

        self._alpha_roll = alpha_roll
        self._alpha_pitch = alpha_pitch

        self._pitch_limits = pitch_limits
        self._roll_limits = roll_limits

        self._F_roll = 0
        self._F_pitch = 0

        self._error_int = np.zeros(2)
        self._prev_error = np.zeros(2)
        self._prev_ts: float = None

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, v_dot_star: np.ndarray, ts: float, debug=False):

        vx = v_actual[0]
        vy = v_actual[1]

        vx_ref = v_ref[0]
        vy_ref = v_ref[1]

        e_x = vx_ref - vx
        e_y = vy_ref - vy

        ax_star = v_dot_star[0]
        ay_star = v_dot_star[1]

        if self._prev_ts is not None and ts != self._prev_ts:
            dt = ts - self._prev_ts

            e_dot_x = (e_x - self._prev_error[0]) / dt
            e_dot_y = (e_y - self._prev_error[1]) / dt

            # Avoid integral windup
            if self._pitch_limits[0] <= self._error_int[0] <= self._pitch_limits[1]:
                self._error_int[0] += e_x * dt

            if self._roll_limits[0] <= self._error_int[0] <= self._roll_limits[1]:
                self._error_int[1] += e_y * dt
        else:
            e_dot_x = e_dot_y = 0

        pitch_ref = (self._F_pitch - ax_star + self._Kp_pitch * e_x + self._Kd_pitch * e_dot_x + self._Ki_pitch * self._error_int[0]) / self._alpha_pitch
        roll_ref = (self._F_roll - ay_star + self._Kp_roll * e_y + self._Kd_roll * e_dot_y + self._Ki_roll * self._error_int[1]) / self._alpha_roll

        # pitch_ref = self._clamp(pitch_ref, self._pitch_limits)
        # roll_ref = self._clamp(roll_ref, self._roll_limits)

        # Update F
        # if self._prev_ts is not None:
            # dt = ts - self._prev_ts

        self._F_pitch += (ax_star - self._alpha_pitch*pitch_ref - self._Kp_pitch*e_x - self._Kd_pitch * e_dot_x - self._Ki_pitch * self._error_int[0])
        self._F_roll += (ay_star - self._alpha_roll*roll_ref - self._Kp_roll*e_y - self._Kd_roll * e_dot_y - self._Ki_roll * self._error_int[1])

        self._prev_error = np.array([e_x, e_y])
        self._prev_ts = ts

        # else:
        # self._F_pitch = self._F_roll = 0
        # self._F_roll = 0

        # self._prev_ts = ts

        if debug:
            print(f"ts: {ts}")
            print(f"Refs:\tPitch: {pitch_ref:.3f}\tRoll: {roll_ref:.3f}")
            print(f"F pitch:\t{self._F_pitch}\tF roll:\t{self._F_roll}")
            print(f"ax_star: {ax_star}\tay_star: {ay_star}")
            print(f"ex: {e_x}\tey: {e_y}")
            print(f"Pitch gains:\tP: {self._Kp_pitch * e_x:.3f}\tI: {self._Ki_pitch * self._error_int[0]:.3f}\tD: {self._Kd_pitch * e_dot_x:.3f} ")
            print(f"Roll gains:\tP: {self._Kp_roll * e_y:.3f}\tI: {self._Ki_roll * self._error_int[1]:.3f}\tD: {self._Kd_roll * e_dot_y:.3f} ")
            print()

        return np.array([roll_ref, pitch_ref])
