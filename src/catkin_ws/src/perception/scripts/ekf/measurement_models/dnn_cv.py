
import numpy as np

class DnnCvModel():

    def __init__(self, sigmas):
        """
        Measurement model for measurements coming from computer vision methods.

        z = [x, y, z, psi]
        
        """
        self.R = np.diag(sigmas)**2
        self.H = np.hstack((np.eye(4), np.zeros((4,3))))

        self.z

    def h(self, x):
        return x[:3]
    
    def H(self):
        return self.H

    def R(self):
        return self.R