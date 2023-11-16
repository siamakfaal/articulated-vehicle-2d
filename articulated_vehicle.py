import json
from icecream import ic
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# Load configuration settings from JSON
with open("settings.json", "r") as config_file:
    config = json.load(config_file)
DEFAULT_FRONT_AXLE_CENTER_TO_PIN = config["vehicle_geometry"]["front_axle_to_pin"]
DEFAULT_REAR_AXLE_CENTER_TO_PIN = config["vehicle_geometry"]["rear_axle_to_pin"]
DEFAULT_TREAD_WIDTH = config["vehicle_geometry"]["tread_width"]
DEFAULT_WHEEL_WIDTH = config["vehicle_geometry"]["wheel_width"]
DEFAULT_WHEEL_RADIUS = config["vehicle_geometry"]["wheel_radius"]


class Parameters:
    def __init__(
        self,
        front_axle_center_to_pin: float = DEFAULT_FRONT_AXLE_CENTER_TO_PIN,
        rear_axle_center_to_pin: float = DEFAULT_REAR_AXLE_CENTER_TO_PIN,
        tread_width: float = DEFAULT_TREAD_WIDTH,
        wheel_width: float = DEFAULT_WHEEL_WIDTH,
        wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    ):
        self.lp = front_axle_center_to_pin
        self.wp = tread_width

        self.lq = rear_axle_center_to_pin
        self.wq = tread_width

        self.sw = wheel_width
        self.rw = wheel_radius


class Variables:
    def __init__(self, params: Parameters = Parameters()):
        self._prms = params

    def _assign(
        self,
        t: np.ndarray,
        p: np.ndarray,
        theta_p: np.ndarray,
        q: np.ndarray,
        theta_q: np.ndarray,
        phi: np.ndarray,
    ):
        self.t = t
        self.p = p
        self.theta_p = theta_p
        self.q = q
        self.theta_q = theta_q
        self.phi = phi

    def decompose(self, ivp_solution):
        p = np.c_[ivp_solution.y[0], ivp_solution.y[1]]
        theta_p = ivp_solution.y[2]
        phi = ivp_solution.y[3]
        theta_q = theta_p - phi
        q = (
            p
            - np.c_[
                self._prms.lp * np.cos(theta_p) + self._prms.lq * np.cos(theta_q),
                self._prms.lp * np.sin(theta_p) + self._prms.lq * np.sin(theta_q),
            ]
        )

        self._assign(ivp_solution.t, p, theta_p, q, theta_q, phi)

    def get_instance(self, index):
        instance = Variables(self._prms)
        instance._assign(
            self.t[index],
            self.p[index, :],
            self.theta_p[index],
            self.q[index, :],
            self.theta_q[index],
            self.phi[index],
        )
        return instance


class Kinematics:
    def __init__(self, params: Parameters = Parameters()):
        self._prms = params
        self.initial_conditions()

    def initial_conditions(
        self,
        position: np.ndarray = [0, 0],
        orientation: float = 0,
        articulation: float = 0,
    ):
        self.x0 = np.concatenate((position, orientation, articulation), axis=None)

    def close_loop_system(self, controller):
        return lambda t, x: self._dem(t, x, controller(t, x))

    def _dem(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Differential Equations of Motion of the Articulated Vehicle
        Args:
            t: time
            x: state vector [p_1, p_2, theta_p, phi]
            u: input vector [v_p, d(phi)/dt]

        Returns:
            dx/dt: derivative of the state vector in time
        """
        diff_p1 = u[0] * np.cos(x[2])
        diff_p2 = u[0] * np.sin(x[2])
        diff_theta_p = (u[0] * np.sin(x[3]) + self._prms.lq * u[1]) / (
            self._prms.lp * np.cos(x[3]) + self._prms.lq
        )
        diff_phi = u[1]

        return np.array([diff_p1, diff_p2, diff_theta_p, diff_phi])

    def _rotation(theta: float):
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    def get_transforms(instance: Variables):
        p_T_w = lambda points: instance.p + np.matmul(
            points, Kinematics._rotation(instance.theta_p)
        )
        q_T_w = lambda points: instance.q + np.matmul(
            points, Kinematics._rotation(instance.theta_q)
        )
        return p_T_w, q_T_w


class Visual:
    DEFAULT_ANIMATION_WINDOW = [[-10, 10], [-10, 10]]

    def __init__(self, axis=None, params: Parameters = Parameters()):
        if axis is None:
            self._fig, self._ax = plt.subplots()
            self._ax.set_aspect("equal", "box")
            self._ax.set_xlim(
                self.DEFAULT_ANIMATION_WINDOW[0][0], self.DEFAULT_ANIMATION_WINDOW[0][1]
            )
            self._ax.set_ylim(
                self.DEFAULT_ANIMATION_WINDOW[1][0], self.DEFAULT_ANIMATION_WINDOW[1][1]
            )

        else:
            self._ax = axis

        self._build_ready = False
        self._patch_handles = {}
        self._prms = params
        self._vertex = {}

    def animate(self, solution: Variables = None):
        if not self._build_ready:
            self._build()

        animation_handle = FuncAnimation(
            self._fig,
            self._update,
            frames=len(solution.t),
            fargs=(solution,),
            blit=True,
            repeat=False,
        )
        plt.show()

    def _update(self, frame_number, solution):
        p_T_w, q_T_w = Kinematics.get_transforms(solution.get_instance(frame_number))
        for key in self._vertex.keys():
            if "front" in key:
                self._patch_handles[key].set_xy(p_T_w(self._vertex[key]))
            else:
                self._patch_handles[key].set_xy(q_T_w(self._vertex[key]))

        self._ax.autoscale_view()

    def _build(self):
        self._define_vertices(self._prms)
        self._build_patches()
        self._build_ready = True

    def _build_patches(self):
        self._patch_handles = {}
        for key in self._vertex.keys():
            self._patch_handles[key] = patches.Polygon(self._vertex[key], closed=True)
            self._ax.add_patch(self._patch_handles[key])

    def _define_vertices(self, prms):
        frnt_head_extension = 1.3 * prms.rw
        frnt_artcl_extension = 1.5 * prms.rw
        frnt_side_extension = (prms.wp - prms.sw) / 2.2

        rear_head_extension = 1.3 * prms.rw
        rear_artcl_extension = 1.5 * prms.rw
        rear_side_extension = (prms.wq - prms.sw) / 2.2

        self._vertex = {}
        self._vertex["front_body"] = np.array(
            [
                [
                    frnt_head_extension,
                    frnt_head_extension,
                    -frnt_artcl_extension,
                    -prms.lp,
                    -frnt_artcl_extension,
                ],
                [
                    -frnt_side_extension,
                    frnt_side_extension,
                    frnt_side_extension,
                    0,
                    -frnt_side_extension,
                ],
            ]
        ).transpose()
        self._vertex["rear_body"] = np.array(
            [
                [
                    rear_artcl_extension,
                    prms.lq,
                    rear_artcl_extension,
                    -rear_head_extension,
                    -rear_head_extension,
                ],
                [
                    -rear_side_extension,
                    0,
                    rear_side_extension,
                    rear_side_extension,
                    -rear_side_extension,
                ],
            ]
        ).transpose()
        self._vertex["front_left_wheel"] = np.array(
            [
                [prms.rw, prms.rw, -prms.rw, -prms.rw],
                [
                    (prms.wp - prms.sw) / 2,
                    (prms.wp + prms.sw) / 2,
                    (prms.wp + prms.sw) / 2,
                    (prms.wp - prms.sw) / 2,
                ],
            ]
        ).transpose()
        self._vertex["front_right_wheel"] = np.array(
            [
                [prms.rw, prms.rw, -prms.rw, -prms.rw],
                [
                    (-prms.wp - prms.sw) / 2,
                    (-prms.wp + prms.sw) / 2,
                    (-prms.wp + prms.sw) / 2,
                    (-prms.wp - prms.sw) / 2,
                ],
            ]
        ).transpose()
        self._vertex["rear_left_wheel"] = np.array(
            [
                [prms.rw, prms.rw, -prms.rw, -prms.rw],
                [
                    (prms.wq - prms.sw) / 2,
                    (prms.wq + prms.sw) / 2,
                    (prms.wq + prms.sw) / 2,
                    (prms.wq - prms.sw) / 2,
                ],
            ]
        ).transpose()
        self._vertex["rear_right_wheel"] = np.array(
            [
                [prms.rw, prms.rw, -prms.rw, -prms.rw],
                [
                    (-prms.wq - prms.sw) / 2,
                    (-prms.wq + prms.sw) / 2,
                    (-prms.wq + prms.sw) / 2,
                    (-prms.wq - prms.sw) / 2,
                ],
            ]
        ).transpose()

    def plot(self, solution: Variables = None):
        if solution is None:
            return

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
