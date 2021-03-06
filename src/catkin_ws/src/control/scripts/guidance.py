
import numpy as np

def get_guidance_law(guidance_law_type: str):
    if guidance_law_type == "pp":
        return PurePursuit
    elif guidance_law_type == "pid":
        return PID
    else:
        raise ValueError(f"Invalid guidance law: {guidance_law_type}")

class GenericGuidanceLaw():

    def get_velocity_reference(self, pos_error_body: np.ndarray, ts: float, debug=False) -> np.ndarray:
        raise NotImplementedError

    def _clamp(self, value: float, limits: tuple):
        if value < limits[0]:
            return limits[0]
        elif value > limits[1]:
            return limits[1]
        else:
            return value

class PurePursuit(GenericGuidanceLaw):

    def __init__(self, params: dict, limits: dict) -> None:
        self._kappa = params["kappa"]

        self._vx_limits = limits["vx"]
        self._vy_limits = limits["vy"]

        print(10*"=", "Pure pursuit guiance law", 10*"=")
        print(f"Kappa: {self._kappa}\t Limits: vx: {self._vx_limits} vy: {self._vy_limits}")
        print(46*"=")

    def get_velocity_reference(self, pos_error_body: np.ndarray, ts: float, debug=False) -> np.ndarray:
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

        vel_ref_unclamped = self._kappa * pos_error_body / np.linalg.norm(pos_error_body)

        vel_ref_x = self._clamp(vel_ref_unclamped[0], self._vx_limits)
        vel_ref_y = self._clamp(vel_ref_unclamped[1], self._vy_limits)

        velocity_reference = np.array([vel_ref_x, vel_ref_y])

        if debug:
            print(f"Timestamp: {ts}")
            print(f"Velocity references:\t vx: {velocity_reference[0]:.3f}\t vy: {velocity_reference[1]:.3f}")
            if not np.array_equal(vel_ref_unclamped, velocity_reference):
                print(f"Unclamped reference:\t vx: {vel_ref_unclamped[0]:.3f}\t vy: {vel_ref_unclamped[1]:.3f}")
            print()

        return velocity_reference

class PID(GenericGuidanceLaw):

    def __init__(self, params: dict, limits: dict):

        self._Kp_x = params["x_axis"]["kp"]
        self._Ki_x = params["x_axis"]["ki"]
        self._Kd_x = params["x_axis"]["kd"]

        self._Kp_y = params["y_axis"]["kp"]
        self._Ki_y = params["y_axis"]["ki"]
        self._Kd_y = params["y_axis"]["kd"]

        self._Kp_z = params["z_axis"]["kp"]
        self._Ki_z = params["z_axis"]["ki"]
        self._Kd_z = params["z_axis"]["kd"]

        self._vx_limits = limits["vx"]
        self._vy_limits = limits["vy"]
        self._vz_limits = limits["vz"]

        print(10*"=", "Position controller control params", 10*"=")
        print(f"X-axis: \tKp: {self._Kp_x} \tKi: {self._Ki_x} \tKd: {self._Kd_x} \tLimits: {self._vx_limits}")
        print(f"Y-axis: \tKp: {self._Kp_y} \tKi: {self._Ki_y} \tKd: {self._Kd_y} \tLimits: {self._vy_limits}")
        print(f"Z-axis: \tKp: {self._Kp_z} \tKi: {self._Ki_z} \tKd: {self._Kd_z} \tLimits: {self._vz_limits}")
        print(56*"=")

        self._prev_ts = None
        self._error_int = np.zeros(3)
        self._prev_error = np.zeros(3)

    def get_velocity_reference(self, pos_error_body: np.ndarray, ts: float, debug=False) -> np.ndarray:

        if pos_error_body.shape[0] == 2:
            control3D = False
        elif pos_error_body.shape[0] == 3:
            control3D = True
        else:
            print(f"Position error has wrong shape, should be 2 or 3, is: {pos_error_body.shape[0]}")

        e_x = pos_error_body[0]
        e_y = pos_error_body[1]

        if control3D:
            e_z = pos_error_body[2]

        if self._prev_ts is not None and ts != self._prev_ts:
            dt = ts - self._prev_ts

            e_dot_x = (e_x - self._prev_error[0]) / dt
            e_dot_y = (e_y - self._prev_error[1]) / dt

            if control3D:
                e_dot_z = (e_z - self._prev_error[2]) / dt

            if control3D:
                self._prev_error = pos_error_body
            else:
                self._prev_error = np.hstack((pos_error_body, 0))

            # Avoid integral windup
            if self._vx_limits[0] <= self._error_int[0] <= self._vx_limits[1]:
                self._error_int[0] += e_x * dt

            if self._vy_limits[0] <= self._error_int[1] <= self._vy_limits[1]:
                self._error_int[1] += e_y * dt

            if control3D:
                if self._vz_limits[0] <= self._error_int[2] <= self._vz_limits[1]:
                    self._error_int[2] += e_z * dt

        else:
            e_dot_x = e_dot_y = e_dot_z = 0

        self._prev_ts = ts

        vx_reference = self._Kp_x*e_x + self._Kd_x*e_dot_x + self._Ki_x*self._error_int[0]
        vy_reference = self._Kp_y*e_y + self._Kd_y*e_dot_y + self._Ki_y*self._error_int[1]

        vx_reference = self._clamp(vx_reference, self._vx_limits)
        vy_reference = self._clamp(vy_reference, self._vy_limits)

        if control3D:
            vz_reference = self._Kp_z*e_z + self._Kd_z*e_dot_z + self._Ki_z*self._error_int[2]
            vz_reference = self._clamp(vz_reference, self._vz_limits)
            velocity_reference = np.array([vx_reference, vy_reference, vz_reference])
        else:
            velocity_reference = np.array([vx_reference, vy_reference])

        if debug:
            print(f"Timestamp: {ts}")
            print(f"Vx gains:\tP: {self._Kp_x*e_x:.3f}\tI: {self._Ki_x*self._error_int[0]:.3f}\tD: {self._Kd_x*e_dot_x:.3f} ")
            print(f"Vy gains:\tP: {self._Kp_y*e_y:.3f}\tI: {self._Ki_y*self._error_int[1]:.3f}\tD: {self._Kd_y*e_dot_y:.3f} ")
            if control3D:
                print(f"Vz gains:\tP: {self._Kp_z*e_z:.3f}\tI: {self._Ki_z*self._error_int[2]:.3f}\tD: {self._Kd_z*e_dot_y:.3f} ")
                print(f"Velocity references:\t vx: {vx_reference:.3f}\t vy: {vy_reference:.3f} vz: {vz_reference:.3f}")
            else:
                print(f"Velocity references:\t vx: {vx_reference:.3f}\t vy: {vy_reference:.3f}")
            print()

        return velocity_reference
