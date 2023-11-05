from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


class Simulator:
    SIMULATION_TIME_STEP = 0.01  # IVP result steps in seconds
    DEFAULT_SIM_TIME = [0, 60]  # Simulation time = [t_0, t_f] in seconds

    def __init__(self, vehicle):
        self._vehicle = vehicle
        self.solution = None

    def solve(self, tspan: list, controller):
        dynamics = self._vehicle.close_loop_system(controller)

        if len(tspan) == 2:
            t_eval = np.arange(tspan[0], tspan[-1], self.SIMULATION_TIME_STEP)
        else:
            t_eval = tspan

        self.solution = solve_ivp(
            fun=dynamics,
            t_span=[tspan[0], tspan[-1]],
            y0=self._vehicle.x0,
            t_eval=t_eval,
        )
        return self.solution
