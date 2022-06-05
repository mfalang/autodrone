
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

def plot_reference_model_comparison_example():
    import control_util
    import seaborn as sns
    import matplotlib.pyplot as plt

    # LaTex must be installed for this to work
    # sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super

    plt.rcParams['text.usetex'] = True

    # plt.rcParams.update({
    #     'font.size' : 20,                   # Set font size to 20pt
    #     'legend.fontsize': 20,              # -> legends
    #     'font.family': 'lmodern',
    #     'text.usetex': True,
    #     'text.latex.preamble': (            # LaTeX preamble
    #         r'\usepackage{lmodern}'
    #         # ... more packages if needed
    #     )
    # })

    omegas_1 = (2, 2)
    zetas_1 = (1, 1)

    ref_model_1 = VelocityReferenceModel(omegas_1, zetas_1)

    omegas_2 = (4, 4)
    zetas_2 = (1, 1)

    ref_model_2 = VelocityReferenceModel(omegas_2, zetas_2)

    v_ref = np.zeros((2, 200))
    v_ref[0,:100] = 1
    v_d_1 = np.zeros((4, v_ref.shape[1]))
    xd_1 = np.zeros(4)
    v_d_2 = np.zeros((4, v_ref.shape[1]))
    xd_2 = np.zeros(4)

    dt = 0.05

    for i in range(v_ref.shape[1]):
        xd_1 = ref_model_1.get_filtered_reference(xd_1, v_ref[:,i], dt)
        v_d_1[:,i] = xd_1

        xd_2 = ref_model_2.get_filtered_reference(xd_2, v_ref[:,i], dt)
        v_d_2[:,i] = xd_2

    time = np.linspace(0, int(v_ref.shape[1] * dt), v_ref.shape[1])

    sns.set()
    fig, ax = plt.subplots(2, 1, sharex=True)

    # fig.suptitle("Raw and smoothed horizontal velocity references")

    # Example 1
    ax[0].plot(time, v_ref[0,:], label=r"$v_r$")
    ax[0].plot(time, v_d_1[0,:], label=r"$v_d$")
    ax[0].plot(time, v_d_1[2,:], label=r"$a_d$")
    ax[0].legend()
    # ax[0].set_title(f"$\omega_n = {omegas_1[0]}, \:\: \zeta = {zetas_1[0]}$")
    ax[0].set_title(f"Natural frequency: $\omega_n = {omegas_1[0]}$, relative damping: $\zeta = {zetas_1[0]}$")
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].set_ylim([-1.8, 1.8])


    # Example 2
    ax[1].plot(time, v_ref[0,:], label=r"$v_r$")
    ax[1].plot(time, v_d_2[0,:], label=r"$v_d$")
    ax[1].plot(time, v_d_2[2,:], label=r"$a_d$")
    ax[1].legend()
    ax[1].set_title(f"Natural frequency: $\omega_n = {omegas_2[0]}$, relative damping: $\zeta = {zetas_2[0]}$")
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_ylim([-1.8, 1.8])

    # fig.savefig("reference_model_example.png", format="png", dpi=300)
    plt.show()

def plot_reference_model_used_in_actual_system():
    import control_util
    import seaborn as sns
    import matplotlib.pyplot as plt

    # LaTex must be installed for this to work
    # sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
    plt.rcParams['text.usetex'] = True

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

    # fig.suptitle("Raw and smoothed horizontal velocity references")

    # Vx
    ax[0].plot(time, v_ref[0,:], label="$v_{r,x}$")
    ax[0].plot(time, v_d[0,:], label="$v_{d,x}$")
    ax[0].legend()
    ax[0].set_ylabel(r"X-axis velocity [m/s]")
    # ax[0].set_title("X-axis")

    # Vy
    ax[1].plot(time, v_ref[1,:], label="$v_{r,y}$")
    ax[1].plot(time, v_d[1,:], label="$v_{d,y}$")
    ax[1].legend()
    # ax[1].set_title("Y-axis")
    ax[1].set_xlabel(r"Time [sec]")
    ax[1].set_ylabel(r"Y-axis velocity [m/s]")


    fig.savefig("vel_refs_for_vel_controller_evaluation.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0)

    plt.show()

if __name__ == "__main__":
    # plot_reference_model_comparison_example()
    plot_reference_model_used_in_actual_system()