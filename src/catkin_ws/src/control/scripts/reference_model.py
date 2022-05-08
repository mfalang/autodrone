
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


if __name__ == "__main__":
    import control_util
    import seaborn as sns
    import matplotlib.pyplot as plt

    omegas = (10, 10)
    zetas = (1, 1)

    ref_model = VelocityReferenceModel(omegas, zetas)

    v_ref = control_util.get_reference_trajectory_safe()
    v_d = np.zeros_like(v_ref)
    dt = 0.05
    xd = np.zeros(4)

    for i in range(v_ref.shape[1]):
        xd = ref_model.get_filtered_reference(xd, v_ref[:,i], dt)
        v_d[:,i] = xd[:2]

    time = np.linspace(0, int(v_ref.shape[1] * dt), v_ref.shape[1])

    sns.set()
    fig, ax = plt.subplots(2, 1, sharex=True)

    fig.suptitle("Raw and smoothed horizontal velocity references")

    # Vx
    ax[0].plot(time, v_ref[0,:], label="vx_ref")
    ax[0].plot(time, v_d[0,:], label="vx_d")
    ax[0].legend()
    ax[0].set_title("X-axis")

    # Vy
    ax[1].plot(time, v_ref[1,:], label="vy_ref")
    ax[1].plot(time, v_d[1,:], label="vy_d")
    ax[1].legend()
    ax[1].set_title("Y-axis")

    # fig.savefig("test.png", format="png", dpi=300)

    plt.show()