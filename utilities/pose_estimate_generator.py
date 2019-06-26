# cSpell: disable=invalid-name
"""Pose estimate generator."""
# pylint: disable=no-name-in-module
import math

import numpy as np

from gtsam import Point3, Pose3, Rot3


def pose_estimate_generator(theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles):
    """Generate pose estimates for mapping."""
    # Camera to world rotation
    wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0).matrix()
    theta = 360/angles
    # theta = 45

    def image_pose(i):
        y_idx = i // (cols*angles)
        x_idx = (i % (cols*angles))//angles
        angle_idx = (i % (cols*angles)) % angles

        rotation = Rot3(np.dot(wRc, np.array([[math.cos(theta*angle_idx), 0, -math.sin(
            theta*angle_idx)], [0, 1, 0], [math.sin(theta*angle_idx), 0, math.cos(theta*angle_idx)]])))
        translation = Point3(delta_x*x_idx, delta_y*y_idx, delta_z)
        image_pose = Pose3(rotation, translation)
        return image_pose

    image_poses = [image_pose(i) for i in range(rows*cols*angles)]

    # prior1_delta = [x_delta, y_delta, z_delta, theta_delta]
    prior_rot_1 = Rot3(np.dot(wRc, np.array([[math.cos(prior1_delta[3]), 0, -math.sin(
        prior1_delta[3])], [0, 1, 0], [math.sin(prior1_delta[3]), 0, math.cos(prior1_delta[3])]])))
    prior_1 = Pose3(prior_rot_1, Point3(
        prior1_delta[0], prior1_delta[1], prior1_delta[2]))
    # prior2_delta = [x_delta, y_delta, z_delta, theta_delta]
    prior_rot_2 = Rot3(np.dot(wRc, np.array([[math.cos(prior2_delta[3]), 0, -math.sin(
        prior2_delta[3])], [0, 1, 0], [math.sin(prior2_delta[3]), 0, math.cos(prior2_delta[3])]])))
    prior_2 = Pose3(prior_rot_2, Point3(
        prior2_delta[0], prior2_delta[1], prior2_delta[2]))

    image_poses[:0] = [prior_2]
    image_poses[:0] = [prior_1]

    return image_poses
