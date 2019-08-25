# cSpell: disable=invalid-name
"""
Unit tests for pose estimate generator.
"""
# pylint: disable=no-name-in-module

import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import gtsam.utils.plot as gtsam_plot
from gtsam import Point3, Pose3, Rot3
from gtsam.utils.test_case import GtsamTestCase
from utilities.pose_estimate_generator import pose_estimate_generator, pose_estimate_generator_rectangle


class TestMapping(GtsamTestCase):
    """Unit tests for pose estimate generator."""

    def test_pose_estimate_generator(self):
        """test pose estimate generator"""
        # Camera to world rotation
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
        theta = 45
        delta_x = [0,3.7592, 3.7592+5]
        delta_y = [0,1.75895,1.75895*2,1.75895*3]
        delta_z = 0.9652
        rows = 4
        cols = 3
        angles = 1
        degree = math.radians(theta)
        R = np.array([[math.cos(degree), -math.sin(degree), 0],
                       [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])

        prior1_delta = [0, 0, delta_z, 0]
        prior2_delta = [3.7592, 0, delta_z, 0]

        actual_pose_estimates = pose_estimate_generator_rectangle(
            theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles)

        # Declare an id for the figure
        figure_number = 0
        fig = plt.figure(figure_number)
        axes = fig.gca(projection='3d')
        plt.cla()
        # Plot cameras
        pose_origin = Pose3(Rot3(1, 0, 0, 0, 1, 0, 0, 0, 1), Point3(0, 0, 0))
        axis_length = 1
        gtsam_plot.plot_pose3(figure_number, pose_origin, axis_length)
        for pose in actual_pose_estimates:
            gtsam_plot.plot_pose3(figure_number, pose, axis_length)
        # Draw
        x_axe = y_axe = z_axe = 20
        axes.set_xlim3d(-x_axe, x_axe)
        axes.set_ylim3d(-y_axe, y_axe)
        axes.set_zlim3d(-z_axe, z_axe)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
