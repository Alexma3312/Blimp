"""Plotting utilities."""
# cSpell: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import gtsam.utils.plot as gtsam_plot
from gtsam import (Point3, Pose3, Rot3,  # pylint: disable=no-name-in-module
                   symbol)


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


def plot_with_result(result, x_axe=30, y_axe=30, z_axe=30, axis_length=2, figure_number=0):
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
        axis_length = axis_length
        gtsam_plot.plot_pose3(figure_number, pose_i, axis_length)
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


def plot_map(landmarks, figure_number=0, x_axe=30, y_axe=30, z_axe=30, axis_length=2):
    """Plot the map and the generated trajectory.
    Parameters:
        landmarks - a list of [x,y,z]
        trajectory - a list of Pose3 object
    """
    # Declare an id for the figure
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot landmark points
    for landmark in landmarks:
        gtsam_plot.plot_point3(figure_number, Point3(
            landmark[0], landmark[1], landmark[2]), 'rx')
    # draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)
    plt.pause(1)
    plt.legend()
    # plt.show()


def plot_map_ax(landmarks, ax, x_axe=30, y_axe=30, z_axe=30):
    """Plot the map and the generated trajectory.
    Parameters:
        landmarks - a list of [x,y,z]
        trajectory - a list of Pose3 object
    """
    # Plot landmark points
    for landmark in landmarks:
        ax.plot([landmark[0]], [landmark[1]], [landmark[2]], 'rx')

    # draw
    ax.set_xlim3d(-x_axe, x_axe)
    ax.set_ylim3d(-y_axe, y_axe)
    ax.set_zlim3d(-z_axe, z_axe)
    plt.pause(1)
    plt.legend()
    # plt.show()


def plot_pose_on_axes(ax, pose, axis_length=0.1):
    """plot poase on axes"""
    # get rotation and translation (center)
    gRp = pose.rotation().matrix()  # rotation from pose to global
    t = pose.translation()
    origin = np.array([t.x(), t.y(), t.z()])

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')


def plot_trajectory_ax(trajectory, ax, x_axe=5, y_axe=5, z_axe=5, axis_length=0.3):
    """Plot the map and the generated trajectory.
    Parameters:
        landmarks - a list of [x,y,z]
        trajectory - a list of Pose3 object
    """
    # Plot cameras
    for pose in trajectory:
        plot_pose_on_axes(ax, pose, axis_length)
    # draw
    ax.set_xlim3d(-x_axe, x_axe)
    ax.set_ylim3d(-y_axe, y_axe)
    ax.set_zlim3d(-z_axe, z_axe)
    plt.pause(1)


def plot_trajectory(trajectory, figure_number=0, x_axe=5, y_axe=5, z_axe=5, axis_length=0.3):
    """Plot the map and the generated trajectory.
    Parameters:
        landmarks - a list of [x,y,z]
        trajectory - a list of Pose3 object
    """
    # Declare an id for the figure
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    # Plot cameras
    for pose in trajectory:
        gtsam_plot.plot_pose3(figure_number, pose, axis_length)
        # gRp = pose.rotation().matrix()  # rotation from pose to global
        # t = pose.translation()

        # axes.scatter([t.x()], [t.y()], [t.z()], s=20, color='red', alpha=1, marker="^")
    # draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)
    plt.pause(1)
    # plt.legend()
    # plt.show()


def plot_trajectory_verification(landmarks, poses, trajectory=[], x_axe=30, y_axe=30, z_axe=30, axis_length=2, figure_number=0):
    """Plot the map, mapping pose results, and the generated trajectory.
    Parameters:
        landmarks - a list of [x,y,z]
        poses - a list of [x,y,z,1*9 rotation matrix]
        trajectory - a list of Pose3 object
    """
    # Declare an id for the figure
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot landmark points
    for landmark in landmarks:
        gtsam_plot.plot_point3(figure_number, Point3(
            landmark[0], landmark[1], landmark[2]), 'rx')
    # Plot mapping pose results
    for pose in poses:
        rotation = Rot3(np.array(pose[3:]).reshape(3, 3))
        actual_pose_i = Pose3(rotation, Point3(np.array(pose[0:3])))
        gtsam_plot.plot_pose3(figure_number, actual_pose_i, axis_length)
        gRp = actual_pose_i.rotation().matrix()  # rotation from pose to global
        t = actual_pose_i.translation()
        axes.scatter([t.x()], [t.y()], [t.z()], s=20,
                     color='red', alpha=1, marker="^")

    # Plot cameras
    for pose in trajectory:
        gtsam_plot.plot_pose3(figure_number, pose, axis_length)
        gRp = pose.rotation().matrix()  # rotation from pose to global
        t = pose.translation()
        axes.scatter([t.x()], [t.y()], [t.z()], s=20, color='blue', alpha=1)

    # draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)
    plt.legend()
    plt.show()


def plot_poses(poses, x_axe=30, y_axe=30, z_axe=30, axis_length=2, figure_number=0):
    """plot the result of sfm"""
    # Declare an id for the figure
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot cameras
    for pose in poses:
        axis_length = axis_length
        gtsam_plot.plot_pose3(figure_number, pose, axis_length)

    # draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(-z_axe, z_axe)
    plt.legend()
    plt.show()


# >>>>>>>>>>>>>>>>>>Below is for color map>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def plot_pose3_on_axes(axes, pose, axis_length=0.1):
#     """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
#     # get rotation and translation (center)
#     gRp = pose.rotation().matrix()  # rotation from pose to global
#     t = pose.translation()
#     origin = np.array([t.x(), t.y(), t.z()])

#     # draw the camera axes
#     x_axis = origin + gRp[:, 0] * axis_length
#     line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
#     axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

#     y_axis = origin + gRp[:, 1] * axis_length
#     line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
#     axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

#     z_axis = origin + gRp[:, 2] * axis_length
#     line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
#     axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')

#     # plot the covariance
#     # TODO (dellaert): make this work
#     # if (nargin>2) && (~isempty(P))
#     #     pPp = P(4:6,4:6); % covariance matrix in pose coordinate frame
#     #     gPp = gRp*pPp*gRp'; % convert the covariance matrix to global coordinate frame
#     #     gtsam.covarianceEllipse3D(origin,gPp);
#     # end


# def plot_pose3(fignum, pose, axis_length=0.1):
#     """Plot a 3D pose on given figure with given 'axis_length'."""
#     # get figure object
#     fig = plt.figure(fignum)
#     axes = fig.gca(projection='3d')
#     plot_pose3_on_axes(axes, pose, axis_length)
