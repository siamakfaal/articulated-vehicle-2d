from icecream import ic
import numpy as np

from simulator import Simulator
import articulated_vehicle as av


def controller(t, x):
    """Control function
    Args:
        t: time
        x: state vector [p_1, p_2, theta_p, phi]

    Returns:
        u: input vector [v_p, d(phi)/dt]
    """
    return np.array([10, -20 * (x[3] - np.sin(t))])


if __name__ == "__main__":
    vehicle = av.Kinematics()

    sim = Simulator(vehicle)
    sim.solve([0, 5], controller)

    solution = av.Variables()
    solution.decompose(sim.solution)

    vis = av.Visual()

    vis.animate(solution)

    vis.plot(solution)
