
import numpy as np

class DnnCvModel():

    def __init__(self, sigmas):
        """
        Measurement model for measurements coming from computer vision methods.

        z = [x, y, z, psi]

        """
        self._R = np.diag(sigmas)**2
        self._H = np.hstack((np.eye(4), np.zeros((4,3))))

    def h(self, x):
        return x[:4]

    def H(self):
        return self._H

    def R(self):
        return self._R