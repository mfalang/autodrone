
import numpy as np

def get_guidance_law(guidance_law_type: str):
    if guidance_law_type == "pp":
        return PurePursuit
    else:
        raise ValueError(f"Invalid guidance law: {guidance_law_type}")

class PurePursuit():

    def __init__(self, params: dict) -> None:
        self._kappa = params["kappa"]

    def get_velocity_reference(self, pos_error_body: np.ndarray) -> np.ndarray:
        """Generate a velocity reference from a position error using the pure
        pursuit guidance law as defined in Fossen 2021.

        Parameters
        ----------
        pos_error_body : np.ndarray (shape: 2x1)
            Position error between drone and object to track, expressed in drone
            body frame

        Returns
        -------
        np.ndarray
            Velocity reference [vx, vy] expressed in drone body frame.
        """
        assert pos_error_body.shape == (2,), f"Incorrect pos_error_body shape. Should be (2,), was {pos_error_body.shape}."

        vel_ref = self._kappa * pos_error_body / np.linalg.norm(pos_error_body)

        return vel_ref