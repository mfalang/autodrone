# This file holds the methods for converting a velocity reference to an attitude
# reference that can be forwarded to the Anafi's internal control system.

import numpy as np

class GenericAttitudeReferenceGenerator():

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float):
        raise NotImplementedError

class PIDReferenceGenerator(GenericAttitudeReferenceGenerator):

    def __init__(self, Kp, Kd, Ki):

        self._Kp = Kp
        self._Kd = Kd
        self._Ki = Ki

        self._prev_timestamp = None
        self._error_integral = np.zeros((2,1))
        self._prev_error = np.zeros((2,1))

    def get_attitude_reference(self, v_ref: np.ndarray, v_actual: np.ndarray, timestamp: float):

        error = v_ref - v_actual
        gain = self._Kp * error

        if self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp

            derivative = self._Kd * (error - self._prev_error) / dt
            self._prev_error = error

            self._error_integral += self._Ki * dt
        else:
            derivative = np.zeros((2,1))

        attitude_reference = gain + derivative + self._error_integral

        return attitude_reference
