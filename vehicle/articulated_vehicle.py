"""Module for handling kinematics and visualization of an articulated vehicle."""

import json
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


class VehicleGeometry:
    """
    This class represents the geometric properties of an articulated vehicle.
    It allows for the definition and manipulation of various geometrical parameters
    such as axle distances, tread widths, wheel width, and radius.

    Attributes:
        front_axle_to_pin (float): Distance from the front axle to the pin.
        rear_axle_to_pin (float): Distance from the rear axle to the pin.
        front_tread_width (float): Width of the front tread.
        rear_tread_width (float): Width of the rear tread.
        wheel_width (float): Width of the wheels.
        wheel_radius (float): Radius of the wheels.
    """

    def __init__(self, **kwargs):
        """
        Initialize the VehicleGeometry with specific geometric parameters. If any parameter is
        not provided, it is loaded from a default configuration file.

        Parameters:
            front_axle_to_pin (float, optional): Distance from the front axle to the pin.
            rear_axle_to_pin (float, optional): Distance from the rear axle to the pin.
            front_tread_width (float, optional): Width of the front tread.
            rear_tread_width (float, optional): Width of the rear tread.
            wheel_width (float, optional): Width of the wheels.
            wheel_radius (float, optional): Radius of the wheels.
        """
        self.front_axle_to_pin = kwargs.get("front_axle_to_pin", None)
        self.rear_axle_to_pin = kwargs.get("rear_axle_to_pin", None)
        self.front_tread_width = kwargs.get("front_tread_width", None)
        self.rear_tread_width = kwargs.get("rear_tread_width", None)
        self.wheel_width = kwargs.get("wheel_width", None)
        self.wheel_radius = kwargs.get("wheel_radius", None)

        defaults = (
            self.load_defaults()
            if any(
                param is None
                for param in [
                    self.front_axle_to_pin,
                    self.rear_axle_to_pin,
                    self.front_tread_width,
                    self.rear_tread_width,
                    self.wheel_width,
                    self.wheel_radius,
                ]
            )
            else None
        )

        self.front_axle_to_pin = (
            self.front_axle_to_pin
            if self.front_axle_to_pin is not None
            else defaults.front_axle_to_pin
        )
        self.rear_axle_to_pin = (
            self.rear_axle_to_pin
            if self.rear_axle_to_pin is not None
            else defaults.rear_axle_to_pin
        )
        self.front_tread_width = (
            self.front_tread_width
            if self.front_tread_width is not None
            else defaults.front_tread_width
        )
        self.rear_tread_width = (
            self.rear_tread_width
            if self.rear_tread_width is not None
            else defaults.rear_tread_width
        )
        self.wheel_width = (
            self.wheel_width if self.wheel_width is not None else defaults.wheel_width
        )
        self.wheel_radius = (
            self.wheel_radius
            if self.wheel_radius is not None
            else defaults.wheel_radius
        )

    @staticmethod
    def load_defaults():
        """
        Load default vehicle geometry settings from a configuration file.

        Returns:
            VehicleGeometry: An instance of VehicleGeometry with default settings.
        """
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

    def lp(self) -> float:
        """
        Get the distance from the front axle to the pin as float. Returns 0 if it is None
        """
        return self.front_axle_to_pin if self.front_axle_to_pin is not None else 0

    def wp(self) -> float:
        """
        Get the width of the front tread as float. Returns 0 if it is None
        """
        return self.front_tread_width if self.front_tread_width is not None else 0

    def lq(self) -> float:
        """
        Get the distance from the rear axle to the pin as float. Returns 0 if it is None
        """
        return self.rear_axle_to_pin if self.rear_axle_to_pin is not None else 0

    def wq(self) -> float:
        """
        Get the width of the rear tread as float. Returns 0 if it is None
        """
        return self.rear_tread_width if self.rear_tread_width is not None else 0

    def sw(self) -> float:
        """
        Get the width of the wheels as float. Returns 0 if it is None
        """
        return self.wheel_width if self.wheel_width is not None else 0

    def rw(self) -> float:
        """
        Get the radius of the wheels as float. Returns 0 if it is None
        """
        return self.wheel_radius if self.wheel_radius is not None else 0

    # pylint: disable=invalid-unary-operand-type
    def vehicle_body_polygons(self):
        """
        Calculate the polygonal representation of the vehicle's body,
        both for the front and rear sections.

        Returns:
            tuple: A tuple containing two NumPy arrays representing
                the front and rear body polygons.
        """

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

    # pylint: disable=invalid-unary-operand-type
    def front_wheel_polygons(self):
        """
        Calculate the polygonal representation of the front wheels,
        both for the front left and right wheels.

        Returns:
            tuple: A tuple containing two NumPy arrays representing
                the front left and right wheel polygons.
        """
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

    # pylint: disable=invalid-unary-operand-type
    def rear_wheel_polygons(self):
        """
        Calculate the polygonal representation of the rear wheels,
        both for the rear left and rear right wheels.

        Returns:
            tuple: A tuple containing two NumPy arrays representing
                the rear left and right wheel polygons.
        """
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
    """
    Represents the kinematic variables of a vehicle. This class stores arrays of time, positions,
    angles, and other kinematic variables relevant to the vehicle's motion.

    Attributes:
        t (np.ndarray): Array of time steps.
        p (np.ndarray): Array of positions (x, y) of the center of the front axis
            defined in the world reference frame.
        theta_p (np.ndarray): Array of orientation angles of the front articulation
            with respect to the world reference frame.
        q (np.ndarray): Array of positions (x, y) of the center of the rear axis
            defined in the world reference frame.
        theta_q (np.ndarray): Array of orientation anglesof the rear articulation
            with respect to the world reference frame.
        phi (np.ndarray): Array of articulation angles of the vehicle.
    """

    t: np.ndarray
    p: np.ndarray
    theta_p: np.ndarray
    q: np.ndarray
    theta_q: np.ndarray
    phi: np.ndarray

    @classmethod
    def decompose_ivp_solution(cls, ivp_solution, vehicle_geometry: VehicleGeometry):
        """
        Decomposes the solution of an initial value problem (IVP) into kinematic variables.

        Parameters:
            ivp_solution: The solution object obtained from an IVP solver, containing
                          arrays for the different kinematic variables.
            vehicle_geometry (VehicleGeometry): The vehicle geometry used to solve the IVP.

        Returns:
            VehicleKinematicVariables: An instance of the class populated with decomposed
                kinematic variables.
        """
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
        """
        Decomposes a state vector (in a fixed time) into kinematic variables of the vehicle.

        Parameters:
            state_vector (np.ndarray): The state vector containing:
                p (np.ndarray): Array of positions (x, y) of the center of the front axis
                    defined in the world reference frame.
                theta_p (np.ndarray): Array of orientation angles of the front articulation
                    with respect to the world reference frame.
                phi (np.ndarray): Array of articulation angles of the vehicle.
            vehicle_geometry (VehicleGeometry): The vehicle geometry used to compute
                dependent variables.
            t (float, optional): The time associated with the state vector.

        Returns:
            VehicleKinematicVariables: An instance of the class with kinematic variables derived
                from the state vector.
        """
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
    def get_instance(cls, existing_solution: "VehicleKinematicVariables", index: int):
        """
        Retrieves an instance of VehicleKinematicVariables at a specific index
            from an existing solution.

        Parameters:
            existing_solution (VehicleKinematicVariables): The existing solution
                object containing arrays of kinematic variables.
            index (int): The index at which to extract the kinematic variables.

        Returns:
            VehicleKinematicVariables: An instance of the class with kinematic variables
                from the specified index.
        """
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
        """
        Computes the dependent kinematic variables based on provided
        parameters and vehicle geometry.

        Parameters:
            p (np.ndarray): Array of positions (x, y) of the center of the front axis
                defined in the world reference frame.
            theta_p (np.ndarray): Array of orientation angles of the front articulation
                with respect to the world reference frame.
            phi (np.ndarray): Array of articulation angles of the vehicle.
            vehicle_geometry (VehicleGeometry): The vehicle geometry used for computation.

        Returns:
            tuple: A tuple containing the q and theta_q.
                q (np.ndarray): Array of positions (x, y) of the center of the rear axis
                    defined in the world refernce frame.
                theta_q (np.ndarray): Array of orientation angles of the
                    rear articulation with respect to the world frame.
        """
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
    """
    Represents a vehicle with methods to set initial conditions, simulate dynamics,
    and visualize the vehicle's motion.

    Attributes:
        _geometry (VehicleGeometry): The geometry of the vehicle.
        _position_0 (np.ndarray): The initial position of the vehicle.
        _orientation_0 (float): The initial orientation angle of the vehicle.
        _articulation_0 (float): The initial articulation angle of the vehicle.
        _patch_handles (dict): Handles for the matplotlib patches used in visualization.
        _vertex (dict): Coordinates for vertices of the vehicle's parts.
    """

    def __init__(self, vehicle_geometry: VehicleGeometry = VehicleGeometry()):
        """
        Initializes the Vehicle object with a given vehicle geometry.

        Parameters:
            vehicle_geometry (VehicleGeometry): The geometry of the vehicle.
        """
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
        """
        Sets the initial condition for the vehicle's position, orientation, and articulation.

        Parameters:
            position (np.ndarray): The initial position of the vehicle.
            orientation (float): The initial orientation angle of the vehicle.
            articulation (float): The initial articulation angle of the vehicle.
        """
        self._position_0 = position
        self._orientation_0 = orientation
        self._articulation_0 = articulation

    def initial_condition(self):
        """
        Returns the vehicle's initial condition. If any of the initial conditions
        are not set, it loads default values from a configuration file.

        Returns:
            np.ndarray: An array containing the initial position, orientation,
                        and articulation of the vehicle.
        """
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
        """
        Creates a closed-loop system function using the provided controller.

        Parameters:
            controller: A control function that takes time and state vector as inputs.

        Returns:
            function: A lambda function representing the closed-loop system.
        """
        return lambda t, x: self._dem(t, x, controller(t, x))

    # pylint: disable=unused-argument
    def _dem(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Differential Equations of Motion (DEM) of the articulated vehicle.

        Parameters:
            t (float): Time.
                t is not used currently, but reserved for future time-based operations
            x (np.ndarray): State vector [p_1, p_2, theta_p, phi].
            u (np.ndarray): Input vector [v_p, d(phi)/dt].

        Returns:
            np.ndarray: Derivative of the state vector with respect to time.
        """
        diff_p1 = u[0] * np.cos(x[2])
        diff_p2 = u[0] * np.sin(x[2])
        diff_theta_p = (u[0] * np.sin(x[3]) + self._geometry.lq() * u[1]) / (
            self._geometry.lp() * np.cos(x[3]) + self._geometry.lq()
        )
        diff_phi = u[1]

        return np.array([diff_p1, diff_p2, diff_theta_p, diff_phi])

    def draw(self, axis):
        """
        Draws the vehicle on the given matplotlib axis.

        Parameters:
            axis: The matplotlib axis on which to draw the vehicle.
        """
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
        for part, vertex in self._vertex.items():
            self._patch_handles[part] = patches.Polygon(vertex, closed=True)
            axis.add_patch(self._patch_handles[part])

    def update_function(self, ivp_solution):
        """
        Creates an update function for vehicle animation based on the solution of an IVP.

        Parameters:
            ivp_solution: The solution of the initial value problem (IVP).

        Returns:
            function: An update function for the vehicle animation.
        """
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
        """
        Updates the vehicle's animation for a given frame number.

        Parameters:
            frame_number (int): The current frame number in the animation.
            solution (VehicleKinematicVariables): Solution containing kinematic variables.

        Raises:
            Exception: If there is an error in updating the vehicle animation.
        """
        try:
            instance = solution.get_instance(solution, frame_number)
            for part, vertex in self._vertex.items():
                if "front" in part:
                    self._patch_handles[part].set_xy(self._p_to_w(instance, vertex))
                else:
                    self._patch_handles[part].set_xy(self._q_to_w(instance, vertex))
        except Exception as e:
            logging.error("Error in updating vehicle animation: %s", e)
            raise

    def _p_to_w(self, instance, points):
        """
        Transforms points from p-coordinate system to world coordinate system.

        Parameters:
            instance (VehicleKinematicVariables): Instance of kinematic variables.
            points (np.ndarray): Points in the p-coordinate system.

        Returns:
            np.ndarray: Transformed points in the world coordinate system.
        """
        return instance.p + np.matmul(points, self.rotz_transpose(instance.theta_p))

    def _q_to_w(self, instance, points):
        """
        Transforms points from q-coordinate system to world coordinate system.

        Parameters:
            instance (VehicleKinematicVariables): Instance of kinematic variables.
            points (np.ndarray): Points in the q-coordinate system.

        Returns:
            np.ndarray: Transformed points in the world coordinate system.
        """
        return instance.q + np.matmul(points, self.rotz_transpose(instance.theta_q))

    def rotz_transpose(self, theta: float):
        """
        Computes the transpose of the rotation matrix for a given angle.

        Parameters:
            theta (float): The rotation angle.

        Returns:
            np.ndarray: The transpose of the rotation matrix.
        """
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    def plot(self, ivp_solution=None):
        """
        Plots various kinematic variables of the vehicle using the solution of an IVP.

        Parameters:
            ivp_solution: The solution of the initial value problem (IVP).

        Raises:
            Exception: If the IVP solution is not provided.
        """
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
