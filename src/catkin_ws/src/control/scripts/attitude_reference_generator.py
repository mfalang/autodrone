# This file holds the methods for converting a velocity reference to an attitude
# reference that can be forwarded to the Anafi's internal control system.

import numpy as np

class GenericAttitudeReferenceGenerator():

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float):
        raise NotImplementedError

class PIDReferenceGenerator(GenericAttitudeReferenceGenerator):

    def __init__(self, roll_Kp_Ki_Kd: tuple, pitch_Kp_Ki_Kd: tuple):

        self._Kp_roll = roll_Kp_Ki_Kd[0]
        self._Ki_roll = roll_Kp_Ki_Kd[1]
        self._Kd_roll = roll_Kp_Ki_Kd[2]

        self._Kp_pitch = pitch_Kp_Ki_Kd[0]
        self._Ki_pitch = pitch_Kp_Ki_Kd[1]
        self._Kd_pitch = pitch_Kp_Ki_Kd[2]

        self._prev_timestamp = None
        self._error_integral = np.zeros(2)
        self._prev_error = np.zeros(2)

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float):

        error = v_ref - v_actual

        error_x = error[0]
        error_y = error[1]

        gain_x = self._Kp_pitch * error_x
        gain_y = self._Kp_roll * error_y

        if self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp

            derivative_x = self._Kd_pitch * (error_x - self._prev_error[0]) / dt
            derivative_y = self._Kd_roll * (error_y - self._prev_error[1]) / dt

            derivative = self._Kd * (error - self._prev_error) / dt
            self._prev_error = error

            self._error_integral[0] += self._Ki_pitch * dt
            self._error_integral[1] += self._Ki_roll * dt

        else:
            derivative_x = derivative_y = 0

        pitch_reference = gain_x + derivative_x + self._error_integral[0]
        roll_reference = gain_y + derivative_y + self._error_integral[1]

        attitude_reference = np.array([roll_reference, pitch_reference])

        return attitude_reference
