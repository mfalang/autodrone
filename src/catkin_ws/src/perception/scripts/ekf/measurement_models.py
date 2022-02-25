
import numpy as np

class MeasurementModel():

    def __init__(self, sigmas):
        """
        Generic measurement model class.
        """

        self._R = np.diag(sigmas)**2
        self._H = None

    def h(self, x):
        """
        This function must be implemented for each instance of the class.
        """
        raise NotImplementedError

    def H(self):
        return self._H

    def R(self):
        return self._R

class DnnCvPosition(MeasurementModel):

    def __init__(self, sigmas):
        """
        Measurement model for position measurements coming from the DNN CV module

        Frame of reference: All positions correspond to the helipad position
        relative to the drone body frame
        z = [x, y, z]

        """
        super().__init__(sigmas)

        self._H = np.hstack((np.eye(3), np.zeros((3,3))))

    def h(self, x):
        return x[:3]

class DroneVelocity(MeasurementModel):
    def __init__(self, sigmas):
        """
        Measurement model for drone velocity measurements coming from
        drone internal EKF.

        Frame of reference: All positions correspond to the helipad position
        relative to the drone body frame
        z = [v_x, v_y, v_z]

        """
        super().__init__(sigmas)

        self._H = np.hstack((np.zeros((3,3)), np.eye(3)))

    def h(self, x):
        return x[3:]
