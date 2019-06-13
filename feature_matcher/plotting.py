"""Plotting utilities."""
# cSpell: disable=invalid-name
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import gtsam.utils.plot as gtsam_plot
from feature_matcher.mapping_back_end import P, X


def plot_sfm_result(result, pose_indices, point_indices):
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
    axes.set_xlim3d(-20, 20)
    axes.set_ylim3d(-20, 20)
    axes.set_zlim3d(-20, 20)
    plt.legend()
    plt.show()
