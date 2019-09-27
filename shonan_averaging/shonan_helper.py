"""Parser."""


import os

import numpy as np

from mapping.bundle_adjustment.parser import get_matches
from mapping.bundle_adjustment.parser import load_features as get_features
from shonan_averaging.myconfig import *
from utilities.pose_estimate_generator import pose_estimate_generator_rectangle
import cv2
from gtsam import Point3, Pose3, Point2, Rot3
import gtsam


def read_essential_data():
    """Read Essential Data from file."""
    dir_name = basedir+'matches/'
    file_name = dir_name+'essential_matrices.dat'

    with open(file_name) as file:
        lines = file.readlines()[2:]
    data = [list(map(np.float, match.split())) for match in lines]
    essential_dict = {(int(match[0]), int(match[1])): np.array(
        match[2:11]).reshape(3, 3) for match in data}

    return essential_dict


def back_projection(key_point=Point2(), pose=Pose3(), depth=20):
    """
    Back Projection Function.
    Input:
        key_point-gtsam.Point2, key point location within the image.
        pose-gtsam.Pose3, camera pose in world coordinate.
    Output:
        gtsam.Pose3, landmark pose in world coordinate.
    """
    # Normalize input key_point
    pn = calibration_matrix.calibrate(key_point)
    # Transfer normalized key_point into homogeneous coordinate and scale with depth
    ph = Point3(depth*pn.x(), depth*pn.y(), depth)
    # Transfer the point into the world coordinate
    return pose.transformFrom(ph)


def generate_g20_data_file():
    pass


def load_features(image_index):
    """ Load features from .key files
    """
    feat_file = os.path.join(
        basedir+'features/', "{0:07}.key".format(image_index))
    # feat_file = basedir+'features/{0:07}.key'.format(image_index)
    keypoints, descriptors = get_features(feat_file)
    return keypoints, descriptors


def load_matches(frame_1, frame_2):
    """ Load matches from .dat files
        matches - a list of [image 1 index, image 1 keypoint index, image 2 index, image 2 keypoint index]
    """
    matches_file = os.path.join(
        basedir+'matches/', "match_{0}_{1}.dat".format(frame_1, frame_2))
    if os.path.isfile(matches_file) is False:
        return []
    _, matches = get_matches(matches_file)
    return matches


def decompose_essential(depth):
    """Decompose essential matrices into rotation and transition."""
    # Create initial estimation.
    pose_estimates = pose_estimate_generator_rectangle(
        theta, delta_x, delta_y, delta_z, prior1_delta, prior2_delta, rows, cols, angles)
    # Get essential matrices
    essential_dict = read_essential_data()
    # Decompose essential matrices
    for value in essential_dict.keys():
        # Get image idices
        idx1 = value[0]
        idx2 = value[1]
        # Get essential matrix
        essential_mtx = essential_dict.get(value)
        R1, R2, T = cv2.decomposeEssentialMat(essential_mtx)
        # Current Pose
        current_pose = Pose3(Rot3(), pose_estimates[idx1].translation())
        # Estimate Pose
        estimate_pose_1 = Pose3(Rot3(R1), pose_estimates[idx2].translation())
        estimate_pose_2 = Pose3(Rot3(R2), pose_estimates[idx2].translation())
        # Get features
        keypoints, _ = load_features(idx1)
        # Get matches
        matches = load_matches(idx1, idx2)
        # Find the correct rotation matrix
        r1_good_project_points = 0
        r2_good_project_points = 0
        for match in matches:
            # back projection
            point_3d = back_projection(
                keypoints[match[1]], current_pose, depth)
            # check cheirality
            _, result1 = gtsam.PinholeCameraCal3_S2(
                estimate_pose_1, calibration_matrix).projectSafe(point_3d)
            if result1 is True:
                r1_good_project_points += 1
            _, result2 = gtsam.PinholeCameraCal3_S2(
                estimate_pose_2, calibration_matrix).projectSafe(point_3d)
            if result2 is True:
                r2_good_project_points += 1
        print(idx1, idx2, r1_good_project_points, r2_good_project_points)

        # Current Pose
        r_current_pose = Pose3(Rot3(), pose_estimates[idx2].translation())
        # Estimate Pose
        l_estimate_pose_1 = Pose3(Rot3(np.linalg.inv(R1)), pose_estimates[idx2].translation())
        l_estimate_pose_2 = Pose3(Rot3(np.linalg.inv(R2)), pose_estimates[idx2].translation())
        for match in matches:
            # back projection
            point_3d = back_projection(
                keypoints[match[3]], r_current_pose, depth)
            # check cheirality
            _, result1 = gtsam.PinholeCameraCal3_S2(
                l_estimate_pose_1, calibration_matrix).projectSafe(point_3d)
            if result1 is True:
                r1_good_project_points += 1
            _, result2 = gtsam.PinholeCameraCal3_S2(
                l_estimate_pose_2, calibration_matrix).projectSafe(point_3d)
            if result2 is True:
                r2_good_project_points += 1
        print(idx1, idx2, r1_good_project_points, r2_good_project_points)
        if(r1_good_project_points>=r2_good_project_points):
            rot = R1
        else:
            rot = R2


decompose_essential(15)
