"""Plotting utilities."""
# cSpell: disable=invalid-name
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import gtsam.utils.plot as gtsam_plot
from gtsam import symbol  # pylint: disable=no-name-in-module


def X(i):  # pylint: disable=invalid-name
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):  # pylint: disable=invalid-name
    """Create key for landmark j."""
    return symbol(ord('p'), j)


def plot_sfm_result(result, pose_indices, point_indices, x_axe=20, y_axe=20, z_axe=20):
    """
    Plot mapping result.
    """
    # Declare an id for the figure
    figure_number = 0
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot points
    # gtsam_plot.plot_3d_points(figure_number, result, 'rx')
    for idx in point_indices:
        point_i = result.atPoint3(P(idx))
        gtsam_plot.plot_point3(figure_number, point_i, 'rx')
    # Plot cameras
    for idx in pose_indices:
        pose_i = result.atPose3(X(idx))
        gtsam_plot.plot_pose3(figure_number, pose_i, 1)
    # Draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)
    plt.legend()
    plt.show()


def plot_with_result(result, x_axe=30, y_axe=30, z_axe=30, figure_number=0):
    """plot the result of sfm"""
    # Declare an id for the figure
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot points
    gtsam_plot.plot_3d_points(figure_number, result, 'rx')
    # Plot cameras
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        gtsam_plot.plot_pose3(figure_number, pose_i, 2)
        i += 1

    # draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)
    plt.legend()
    plt.show()


def plot_with_results(result1, result2, x_axe=30, y_axe=30, z_axe=30, figure_number1=0, figure_number2=1):
    """plot the two results of sfm at the same time"""
    # Declare an id for the figure
    fig = plt.figure(figure_number1)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot points
    gtsam_plot.plot_3d_points(figure_number1, result1, 'rx')
    # Plot cameras
    i = 0
    while result1.exists(X(i)):
        pose_i = result1.atPose3(X(i))
        gtsam_plot.plot_pose3(figure_number1, pose_i, 2)
        i += 1

    # draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)

    # Declare an id for the figure
    fig = plt.figure(figure_number2)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot points
    gtsam_plot.plot_3d_points(figure_number2, result2, 'rx')
    # Plot cameras
    i = 0
    while result2.exists(X(i)):
        pose_i = result2.atPose3(X(i))
        gtsam_plot.plot_pose3(figure_number2, pose_i, 2)
        i += 1

    # draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)
    plt.legend()
    plt.show()
