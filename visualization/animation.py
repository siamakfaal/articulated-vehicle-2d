"""Module for creating and managing animations using Matplotlib.

This module provides a class 'Animate' that facilitates the creation
and management of animations using the Matplotlib library.
"""

import json
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes


class Animate:
    """
    A class for creating and managing animations using Matplotlib.

    Attributes:
        _fig (matplotlib.figure.Figure): The figure object for the animation.
        _ax (matplotlib.axes.Axes): The axes object for the animation.

    Methods:
        _wrapper_update(frame, *update_functions): Calls update functions for each frame.
        animate(time, *update_functions): Initiates the animation process.
    """

    def __init__(self, axis: Axes = None, animation_window: List = None):
        """
        Initializes the Animate class.

        Args:
            axis (matplotlib.axes.Axes, optional): Existing axes to draw the animation.
                Defaults to None.
            animation_window (List, optional): The window limits for the animation as
                [[xmin, xmax], [ymin, ymax]]. If None, tries to load from settings.json.
                Defaults to None.
        """
        if axis is None:
            self._fig, self._ax = plt.subplots()
            self._ax.set_aspect("equal", "box")
            if not animation_window:
                try:
                    with open("settings.json", "r", encoding="utf-8") as config_file:
                        config = json.load(config_file)
                    animation_window = config["animation"]["animation_window"]
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logging.error("Error loading settings: %s", e)
                    animation_window = [[-10, 10], [-10, 10]]  # Default value

            self._ax.set_xlim(animation_window[0][0], animation_window[0][1])
            self._ax.set_ylim(animation_window[1][0], animation_window[1][1])

        else:
            self._ax = axis
            self._fig = axis.figure

    def _wrapper_update(self, frame, *update_functions):
        """
        Wrapper function to update each frame of the animation.

        Args:
            frame (int): The current frame number.
            *update_functions: Variable length argument list of functions to be called
                for updating each frame.
        """
        for fcn in update_functions:
            fcn(frame)

    def animate(self, time: np.ndarray, *update_functions):
        """
        Initiates the animation process.

        Args:
            time (np.ndarray): An array of time steps for the animation.
            *update_functions: Variable length argument list of functions to be called
                for updating each frame.
        """
        animation_handle = FuncAnimation(  # pylint: disable=unused-variable
            self._fig,
            self._wrapper_update,
            frames=len(time),
            fargs=update_functions,
            blit=False,
            repeat=False,
        )
        plt.show()
