"""Module for handling kinematics and visualization of an articulated vehicle."""

import json
import logging
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


class VehicleGeometry:
    def __init__(
        self,
        front_axle_to_pin: float = None,
        rear_axle_to_pin: float = None,
        front_tread_width: float = None,
        rear_tread_width: float = None,
        wheel_width: float = None,
        wheel_radius: float = None,
    ):
        defaults = (
            self.load_defaults()
            if any(
                param is None
                for param in [
                    front_axle_to_pin,
                    rear_axle_to_pin,
                    front_tread_width,
                    rear_tread_width,
                    wheel_width,
                    wheel_radius,
                ]
            )
            else None
        )

        self.front_axle_to_pin = (
            front_axle_to_pin
            if front_axle_to_pin is not None
            else defaults.front_axle_to_pin
        )
        self.rear_axle_to_pin = (
            rear_axle_to_pin
            if rear_axle_to_pin is not None
            else defaults.rear_axle_to_pin
        )
        self.front_tread_width = (
            front_tread_width
            if front_tread_width is not None
            else defaults.front_tread_width
        )
        self.rear_tread_width = (
            rear_tread_width
            if rear_tread_width is not None
            else defaults.rear_tread_width
        )
        self.wheel_width = (
            wheel_width if wheel_width is not None else defaults.wheel_width
        )
        self.wheel_radius = (
            wheel_radius if wheel_radius is not None else defaults.wheel_radius
        )

    @staticmethod
    def load_defaults():
        with open("settings.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return VehicleGeometry(
            front_axle_to_pin=config["vehicle_geometry"]["front_axle_to_pin"],
            rear_axle_to_pin=config["vehicle_geometry"]["rear_axle_to_pin"],
            front_tread_width=config["vehicle_geometry"]["front_tread_width"],
            rear_tread_width=config["vehicle_geometry"]["rear_tread_width"],
            wheel_width=config["vehicle_geometry"]["wheel_width"],
            wheel_radius=config["vehicle_geometry"]["wheel_radius"],
        )

    def lp(self):
        return self.front_axle_to_pin

    def wp(self):
        return self.front_tread_width

    def lq(self):
        return self.rear_axle_to_pin

    def wq(self):
        return self.rear_tread_width

    def sw(self):
        return self.wheel_width

    def rw(self):
        return self.wheel_radius

    def vehicle_body_polygons(self):
        f_head_ext = 1.3 * self.rw()
        f_artcl_ext = 1.5 * self.rw()
        f_side_ext = (self.wp() - self.sw()) / 2.2

        r_head_ext = 1.3 * self.rw()
        r_artcl_ext = 1.5 * self.rw()
        r_side_ext = (self.wq() - self.sw()) / 2.2

        front_body = np.array(
            [
                [f_head_ext, f_head_ext, -f_artcl_ext, -self.lp(), -f_artcl_ext],
                [-f_side_ext, f_side_ext, f_side_ext, 0, -f_side_ext],
            ]
        ).T

        rear_body = np.array(
            [
                [r_artcl_ext, self.lq(), r_artcl_ext, -r_head_ext, -r_head_ext],
                [-r_side_ext, 0, r_side_ext, r_side_ext, -r_side_ext],
            ]
        ).T

        return front_body, rear_body

    def front_wheel_polygons(self):
        w_side_ext = (self.wp() - self.sw()) / 2

        front_left_wheel = np.array(
            [
                [self.rw(), self.rw(), -self.rw(), -self.rw()],
                [
                    w_side_ext,
                    w_side_ext + self.sw(),
                    w_side_ext + self.sw(),
                    w_side_ext,
                ],
            ]
        ).T

        front_right_wheel = np.array(
            [
                [self.rw(), self.rw(), -self.rw(), -self.rw()],
                [
                    -w_side_ext,
                    -w_side_ext - self.sw(),
                    -w_side_ext - self.sw(),
                    -w_side_ext,
                ],
            ]
        ).T

        return front_left_wheel, front_right_wheel

    def rear_wheel_polygons(self):
        w_side_ext = (self.wq() - self.sw()) / 2

        rear_left_wheel = np.array(
            [
                [self.rw(), self.rw(), -self.rw(), -self.rw()],
                [
                    w_side_ext,
                    w_side_ext + self.sw(),
                    w_side_ext + self.sw(),
                    w_side_ext,
                ],
            ]
        ).T

        rear_right_wheel = np.array(
            [
                [self.rw(), self.rw(), -self.rw(), -self.rw()],
                [
                    -w_side_ext,
                    -w_side_ext - self.sw(),
                    -w_side_ext - self.sw(),
                    -w_side_ext,
                ],
            ]
        ).T

        return rear_left_wheel, rear_right_wheel


@dataclass
class VehicleKinematicVariables:
    t: np.ndarray
    p: np.ndarray
    theta_p: np.ndarray
    q: np.ndarray
    theta_q: np.ndarray
    phi: np.ndarray

    @classmethod
    def decompose_ivp_solution(cls, ivp_solution, vehicle_geometry: VehicleGeometry):
        p = np.c_[ivp_solution.y[0], ivp_solution.y[1]]
        theta_p = ivp_solution.y[2]
        phi = ivp_solution.y[3]
        q, theta_q = cls._compute_dependent_variables(p, theta_p, phi, vehicle_geometry)
        return cls(
            t=ivp_solution.t,
            p=p,
            theta_p=theta_p,
            q=q,
            theta_q=theta_q,
            phi=phi,
        )

    @classmethod
    def decompose_state_vector(
        cls, state_vector: np.ndarray, vehicle_geometry: VehicleGeometry, t: float = 0
    ):
        p = state_vector[0:1]
        theta_p = state_vector[2]
        phi = state_vector[3]
        q, theta_q = cls._compute_dependent_variables(p, theta_p, phi, vehicle_geometry)
        return cls(
            t=t,
            p=p,
            theta_p=theta_p,
            q=q,
            theta_q=theta_q,
            phi=phi,
        )

    @classmethod
    def get_instance(cls, existing_solution, index):
        return cls(
            t=existing_solution.t[index],
            p=existing_solution.p[index, :],
            theta_p=existing_solution.theta_p[index],
            q=existing_solution.q[index, :],
            theta_q=existing_solution.theta_q[index],
            phi=existing_solution.phi[index],
        )

    @staticmethod
    def _compute_dependent_variables(
        p: np.ndarray,
        theta_p: np.ndarray,
        phi: np.ndarray,
        vehicle_geometry: VehicleGeometry,
    ):
        theta_q = theta_p - phi
        q = (
            p
            - np.c_[
                vehicle_geometry.lp() * np.cos(theta_p)
                + vehicle_geometry.lq() * np.cos(theta_q),
                vehicle_geometry.lp() * np.sin(theta_p)
                + vehicle_geometry.lq() * np.sin(theta_q),
            ]
        )
        return q, theta_q


class Vehicle:
    def __init__(self, vehicle_geometry: VehicleGeometry = VehicleGeometry()):
        self._geometry = vehicle_geometry

        # Initial configuration
        self._position_0 = None
        self._orientation_0 = None
        self._articulation_0 = None

        # Visualizations
        self._patch_handles = {}
        self._vertex = {}

    def set_initial_condition(
        self,
        position: np.ndarray,
        orientation: float,
        articulation: float,
    ):
        self._position_0 = position
        self._orientation_0 = orientation
        self._articulation_0 = articulation

    def initial_condition(self):
        if any(
            param is None
            for param in [self._position_0, self._orientation_0, self._articulation_0]
        ):
            with open("settings.json", "r", encoding="utf-8") as config_file:
                config = json.load(config_file)

        position = (
            self._position_0
            if self._position_0 is not None
            else config["vehicle_initial_state"]["position"]
        )

        orientation = (
            self._orientation_0
            if self._orientation_0 is not None
            else config["vehicle_initial_state"]["orientation"]
        )

        articulation = (
            self._articulation_0
            if self._articulation_0 is not None
            else config["vehicle_initial_state"]["articulation"]
        )

        return np.concatenate((position, orientation, articulation), axis=None)

    def close_loop_system(self, controller):
        return lambda t, x: self._dem(t, x, controller(t, x))

    def _dem(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Differential Equations of Motion (DEM) of the Articulated Vehicle
        Args:
            t: time
            x: state vector [p_1, p_2, theta_p, phi]
            u: input vector [v_p, d(phi)/dt]

        Returns:
            dx/dt: derivative of the state vector in time
        """
        diff_p1 = u[0] * np.cos(x[2])
        diff_p2 = u[0] * np.sin(x[2])
        diff_theta_p = (u[0] * np.sin(x[3]) + self._geometry.lq() * u[1]) / (
            self._geometry.lp() * np.cos(x[3]) + self._geometry.lq()
        )
        diff_phi = u[1]

        return np.array([diff_p1, diff_p2, diff_theta_p, diff_phi])

    def draw(self, axis):
        front_body, rear_body = self._geometry.vehicle_body_polygons()
        front_left_wheel, front_right_wheel = self._geometry.front_wheel_polygons()
        rear_left_wheel, rear_right_wheel = self._geometry.rear_wheel_polygons()

        self._vertex = {
            "front_body": front_body,
            "rear_body": rear_body,
            "front_left_wheel": front_left_wheel,
            "front_right_wheel": front_right_wheel,
            "rear_left_wheel": rear_left_wheel,
            "rear_right_wheel": rear_right_wheel,
        }
        for key in self._vertex:
            self._patch_handles[key] = patches.Polygon(self._vertex[key], closed=True)
            axis.add_patch(self._patch_handles[key])

    def update_function(self, ivp_solution):
        if not self._patch_handles:
            raise ValueError(
                "Vehicle patches are not defined in the current axis.\n"
                "Call draw to add patches."
            )

        solution = VehicleKinematicVariables.decompose_ivp_solution(
            ivp_solution, self._geometry
        )
        return lambda frame: self._update(frame, solution)

    def _update(self, frame_number, solution: VehicleKinematicVariables):
        try:
            instance = solution.get_instance(solution, frame_number)
            for key in self._vertex.keys():
                if "front" in key:
                    self._patch_handles[key].set_xy(
                        self._p_to_w(instance, self._vertex[key])
                    )
                else:
                    self._patch_handles[key].set_xy(
                        self._q_to_w(instance, self._vertex[key])
                    )
        except Exception as e:
            logging.error("Error in updating vehicle animation: %s", e)
            raise

    def _p_to_w(self, instance, points):
        return instance.p + np.matmul(points, self.rotz_transpose(instance.theta_p))

    def _q_to_w(self, instance, points):
        return instance.q + np.matmul(points, self.rotz_transpose(instance.theta_q))

    def rotz_transpose(self, theta: float):
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    def plot(self, ivp_solution=None):
        if ivp_solution is None:
            logging.error("IVP solution is needed to plot.")

        solution = VehicleKinematicVariables.decompose_ivp_solution(
            ivp_solution, self._geometry
        )

        plt.figure()

        plt.subplot(321)
        plt.plot(solution.t, solution.p)
        plt.xlabel("t")
        plt.ylabel("p(t)")

        plt.subplot(322)
        plt.plot(solution.t, solution.theta_p)
        plt.xlabel("t")
        plt.ylabel("theta_p(t)")

        plt.subplot(323)
        plt.plot(solution.t, solution.q)
        plt.xlabel("t")
        plt.ylabel("q(t)")

        plt.subplot(324)
        plt.plot(solution.t, solution.theta_q)
        plt.xlabel("t")
        plt.ylabel("theta_q(t)")

        plt.subplot(325)
        plt.plot(solution.t, solution.phi)
        plt.xlabel("t")
        plt.ylabel("phi(t)")

        plt.subplot(326)
        plt.plot(solution.p[:, 0], solution.p[:, 1])
        plt.plot(solution.q[:, 0], solution.q[:, 1])
        plt.xlabel("Horizontal Axis")
        plt.ylabel("Vertical Axis")

        plt.show()
