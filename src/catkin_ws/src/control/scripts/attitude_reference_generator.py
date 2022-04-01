# This file holds the methods for converting a velocity reference to an attitude
# reference that can be forwarded to the Anafi's internal control system.

import numpy as np

def get_attitude_reference_generator(generator_type: str):
    if generator_type == "pid":
        return PIDReferenceGenerator
    elif generator_type == "linear_drag_model":
        return LinearDragModelReferenceGenerator
    elif generator_type == "ipid":
        return iPIDReferenceGenerator
    else:
        raise ValueError("Invalid generator type")

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

    def __init__(self, params: dict, limits: dict):

        self._Kp_x = params["x_axis"]["kp"]
        self._Ki_x = params["x_axis"]["ki"]
        self._Kd_x = params["x_axis"]["kd"]

        self._Kp_y = params["y_axis"]["kp"]
        self._Ki_y = params["y_axis"]["ki"]
        self._Kd_y = params["y_axis"]["kd"]

        self._pitch_limits = limits["pitch"]
        self._roll_limits = limits["roll"]

        print(10*"=", "Control params", 10*"=")
        print(f"Pitch: \tKp: {self._Kp_x} \tKi: {self._Ki_x} \tKd: {self._Kd_x} \tLimits: {self._pitch_limits}")
        print(f"Roll: \tKp: {self._Kp_y} \tKi: {self._Ki_y} \tKd: {self._Kd_y} \tLimits: {self._roll_limits}")
        print(36*"=")

        self._prev_ts = None
        self._error_int = np.zeros(2)
        self._prev_error = np.zeros(2)

    def get_attitude_reference(self, v_ref: np.ndarray, v: np.ndarray, ts: float, debug=False):

        error = v_ref[:2] - v

        e_x = error[0]
        e_y = error[1]

        if self._prev_ts is not None and ts != self._prev_ts:
            dt = ts - self._prev_ts

            e_dot_x = (e_x - self._prev_error[0]) / dt
            e_dot_y = (e_y - self._prev_error[1]) / dt

            self._prev_error = error

            # Avoid integral windup
            if self._pitch_limits[0] <= self._error_int[0] <= self._pitch_limits[1]:
                self._error_int[0] += e_x * dt

            if self._roll_limits[0] <= self._error_int[1] <= self._roll_limits[1]:
                self._error_int[1] += e_y * dt

        else:
            e_dot_x = e_dot_y = 0

        self._prev_ts = ts

        pitch_reference = self._Kp_x*e_x + self._Kd_x*e_dot_x + self._Ki_x*self._error_int[0]
        roll_reference = self._Kp_y*e_y + self._Kd_y*e_dot_y + self._Ki_y*self._error_int[1]

        pitch_reference = self._clamp(pitch_reference, self._pitch_limits)
        roll_reference = self._clamp(roll_reference, self._roll_limits)

        attitude_reference = np.array([roll_reference, pitch_reference])

        if debug:
            print(f"Timestamp: {ts}")
            print(f"Pitch gains:\tP: {self._Kp_x*e_x:.3f}\tI: {self._Ki_x*self._error_int[0]:.3f}\tD: {self._Kd_x*e_dot_x:.3f} ")
            print(f"Roll gains:\tP: {self._Kp_y*e_y:.3f}\tI: {self._Ki_y*self._error_int[1]:.3f}\tD: {self._Kd_y*e_dot_y:.3f} ")
            print()

        return attitude_reference

class LinearDragModelReferenceGenerator(GenericAttitudeReferenceGenerator):

    def __init__(self, params: dict, limits: dict):

        self._m = params["drone_mass"]
        self._g = params["g"]
        self._d_x = params["dx"]
        self._d_y = params["dy"]

        self._pitch_limits = limits["pitch"]
        self._roll_limits = limits["roll"]

        print(10*"=", "Control params", 10*"=")
        print(f"Pitch:\tdx: {self._d_x}\tLimits: {self._pitch_limits}")
        print(f"Roll:\tdy: {self._d_y}\tLimits: {self._roll_limits}")
        print(36*"=")

    def get_attitude_reference(self, v_ref: np.ndarray, v: np.ndarray, ts: float, debug=False):
        vx = v[0]
        vy = v[1]

        vx_ref = v_ref[0]
        vy_ref = v_ref[1]

        accel_x_desired = vx_ref - vx
        accel_y_desired = vy_ref - vy

        # Negative on x axis due to inverse relationship between pitch angle and x-velocity
        pitch_ref = np.rad2deg(np.arctan(-(accel_x_desired / self._g + (self._d_x * vx) / (self._m * self._g))))
        roll_ref = np.rad2deg(np.arctan(accel_y_desired / self._g + (self._d_y * vy) / (self._m * self._g)))

        pitch_ref = self._clamp(pitch_ref, self._pitch_limits)
        roll_ref = self._clamp(roll_ref, self._roll_limits)

        attitude_ref = np.array([roll_ref, pitch_ref])

        if debug:
            print(f"ts:{ts}\tRefs: R: {roll_ref:.3f}\tP: {pitch_ref:.3f}\tax_des: {accel_x_desired:.3f}\tay_des: {accel_y_desired:.3f}")
            print()

        return attitude_ref

class iPIDReferenceGenerator(GenericAttitudeReferenceGenerator):

    def __init__(self, params: dict, limits: dict):

        self._Kp_x = params["x_axis"]["kp"]
        self._Ki_x = params["x_axis"]["ki"]
        self._Kd_x = params["x_axis"]["kd"]
        self._alpha_x = params["x_axis"]["alpha"]

        self._Kp_y = params["y_axis"]["kp"]
        self._Ki_y = params["y_axis"]["ki"]
        self._Kd_y = params["y_axis"]["kd"]
        self._alpha_y = params["y_axis"]["alpha"]

        self._pitch_limits = limits["pitch"]
        self._roll_limits = limits["roll"]

        print(10*"=", "Control params", 10*"=")
        print(f"Pitch:\tKp: {self._Kp_x}\tKi: {self._Ki_x}\tKd: {self._Kd_x}\tAlpha: {self._alpha_x}\tLimits: {self._pitch_limits}")
        print(f"Roll:\tKp: {self._Kp_y}\tKi: {self._Ki_y}\tKd: {self._Kd_y}\tAlpha: {self._alpha_y}\tLimits: {self._roll_limits}")
        print(36*"=")

        self._F_roll = 0
        self._F_pitch = 0

        self._error_int = np.zeros(2)
        self._prev_error = np.zeros(2)
        self._prev_ts: float = None

    def get_attitude_reference(self, v_ref: np.ndarray, v: np.ndarray, ts: float, debug=False):

        vx = v[0]
        vy = v[1]

        vx_ref = v_ref[0]
        vy_ref = v_ref[1]

        e_x = vx_ref - vx
        e_y = vy_ref - vy

        ax_star = v_ref[3]
        ay_star = v_ref[4]

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

        pitch_ref = self._clamp(pitch_ref, self._pitch_limits)
        roll_ref = self._clamp(roll_ref, self._roll_limits)

        attitude_reference = np.array([roll_ref, pitch_ref])

        self._F_pitch += (ax_star - self._alpha_pitch*pitch_ref - self._Kp_pitch*e_x - self._Kd_pitch * e_dot_x - self._Ki_pitch * self._error_int[0])
        self._F_roll += (ay_star - self._alpha_roll*roll_ref - self._Kp_roll*e_y - self._Kd_roll * e_dot_y - self._Ki_roll * self._error_int[1])

        self._prev_error = np.array([e_x, e_y])
        self._prev_ts = ts

        if debug:
            print(f"ts: {ts}")
            print(f"Refs:\tPitch: {pitch_ref:.3f}\tRoll: {roll_ref:.3f}")
            print(f"F pitch:\t{self._F_pitch}\tF roll:\t{self._F_roll}")
            print(f"ax_star: {ax_star}\tay_star: {ay_star}")
            print(f"ex: {e_x}\tey: {e_y}")
            print(f"Pitch gains:\tP: {self._Kp_pitch * e_x:.3f}\tI: {self._Ki_pitch * self._error_int[0]:.3f}\tD: {self._Kd_pitch * e_dot_x:.3f} ")
            print(f"Roll gains:\tP: {self._Kp_roll * e_y:.3f}\tI: {self._Ki_roll * self._error_int[1]:.3f}\tD: {self._Kd_roll * e_dot_y:.3f} ")
            print()

        return attitude_reference
