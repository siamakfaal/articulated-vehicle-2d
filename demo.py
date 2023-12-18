"""
This example demonstrates the use of the Simulator class for vehicle kinematics simulation.
It defines a controller function, creates a vehicle instance, and then uses the Simulator
to solve the vehicle's motion over a specified time span. The results are then decomposed,
visualized, and animated using the vehicle's visualization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np

import vehicle.articulated_vehicle as av
from simulator.simulator import Simulator
from visualization.animation import Animate

# from icecream import ic


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
    vehicle = av.Vehicle()

    sim = Simulator(vehicle)
    sim.solve([0, 5], controller)

    # vehicle.plot(sim.solution)

    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", "box")

    vehicle.draw(ax)

    animator = Animate(ax)

    animator.animate(sim.solution.t, vehicle.update_function(sim.solution))
