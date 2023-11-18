import json
import logging
from typing import Callable, List

import numpy as np
from scipy.integrate import solve_ivp

# Load configuration settings from JSON
with open("settings.json", "r") as config_file:
    config = json.load(config_file)
SIMULATION_TIME_STEP = config["simulation"]["time_step"]
DEFAULT_SIM_TIME = config["simulation"]["default_run_time"]


class Simulator:
    """
    Simulator class to solve vehicle kinematics over a given time span using a specified
    controller.
    """

    def __init__(self, vehicle):
        """
        Initialize the Simulator with a vehicle object.

        Args:
            vehicle: A vehicle object containing dynamics and initial conditions.
        """
        self._vehicle = vehicle
        self.solution = None

    def solve(self, tspan: List[float], controller: Callable):
        """
        Solves the vehicle kinematics over the specified time span using the given controller.

        Args:
            tspan: Time span for the simulation
                tspan = [start, end] => time points is generated automatically
                tspan = [t0, t1, t2, ..., tf] => tspan is also use for time points

            controller: Control function to be used in the simulation.
                controller(t: float, x: np.ndarray) -> u:np.ndarray
        Return:
            Solution of the simulation as a bunch object with the following fields
                t: ndarray -- time points
                y: ndarray -- solution at time points
                see: scipy.integrate.solve_ivp for details
        """
        try:
            dynamics = self._vehicle.close_loop_system(controller)

            if len(tspan) == 2:
                t_eval = np.arange(tspan[0], tspan[-1], SIMULATION_TIME_STEP)
            else:
                t_eval = tspan

            self.solution = solve_ivp(
                fun=dynamics,
                t_span=[tspan[0], tspan[-1]],
                y0=self._vehicle.x0,
                t_eval=t_eval,
            )

            # Check the status of the solution
            if self.solution.status == -1:
                logging.warning("Integration step failed.")
            elif self.solution.status == 0:
                logging.info("The solver successfully reached the end of tspan.")
            elif self.solution.status == 1:
                logging.info("A termination event occurred.")

            return self.solution

        except Exception as e:
            logging.error("Error in simulation: {}".format(e))
            raise
