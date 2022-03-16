# This file holds the methods for converting a velocity reference to an attitude
# reference that can be forwarded to the Anafi's internal control system.

import numpy as np

class GenericAttitudeReferenceGenerator():

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float):
        raise NotImplementedError

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
