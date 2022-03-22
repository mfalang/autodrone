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
                self._error_integral[0] += self._Ki_pitch * error_x * dt

            if self._roll_min <= self._error_integral[0] <= self._roll_max:
                self._error_integral[1] += self._Ki_roll * error_y * dt


        else:
            derivative_x = derivative_y = 0
            self._prev_timestamp = timestamp

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