"""Unit Test camera model and calibration matrix by calculating re-projection errors."""
# cSpell: disable=invalid-name
# pylint: disable=no-member, line-too-long

import cv2
import numpy as np

import gtsam
from gtsam import (Point2, Point3, Pose3,  # pylint: disable=no-name-in-module
                   Rot3, symbol)
from utilities.plotting import plot_with_result


def X(i):  # pylint:disable=invalid-name
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):  # pylint:disable=invalid-name
    """Create key for landmark j."""
    return symbol(ord('p'), j)


def back_projection(cal, key_point=Point2(), pose=Pose3(), depth=10, undist=0):
    """Back project a 2d key point to 3d base on depth."""
    # Normalize input key_point
    if undist == 0:
        point = cal.calibrate(key_point, tol=1e-5)
    if undist == 1:
        point = cal.calibrate(key_point)

    # Transfer normalized key_point into homogeneous coordinate and scale with depth
    point_3d = Point3(depth*point.x(), depth*point.y(), depth)

    # Transfer the point into the world coordinate
    return pose.transform_from(point_3d)


def bundle_adjustment(cal, kp1, kp2, undist=0):
    """ Use GTSAM to solve Structure from Motion.
        Parameters:
            cal - camera calibration matrix,gtsam.Cal3_S2/ gtsam.Cal3DS2
            kp1 - keypoints extracted at camera pose 1,gtsam.Point2
            kp2 - keypoints extracted at camera pose 2,gtsam.Point2
        Output:
            landmarks - a list includes landmark points, gtsam.Point3
            poses - a list includes camera poses, gtsam.Pose3
    """
    # Initialize factor Graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Add factors for all measurements
    measurement_noise_sigma = 1.0
    measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
        2, measurement_noise_sigma)
    for i in range(6):
        if undist == 0:
            graph.add(gtsam.GenericProjectionFactorCal3DS2(
                kp1[i], measurement_noise,
                X(0), P(i), cal))
            graph.add(gtsam.GenericProjectionFactorCal3DS2(
                kp2[i], measurement_noise,
                X(1), P(i), cal))
        if undist == 1:
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                kp1[i], measurement_noise,
                X(0), P(i), cal))
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                kp2[i], measurement_noise,
                X(1), P(i), cal))

    # Create priors and initial estimate
    s = np.radians(60)  # pylint: disable=invalid-name
    pose_noise_sigmas = np.array([s, s, s, 1, 1, 1])
    # pose_noise_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(pose_noise_sigmas)

    # Create Rot3
    wRc = Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]
                        ).T)  # pylint: disable=invalid-name
    for i, y in enumerate([0, 5*1.58]):
        # The approximate height measurement is 1.2
        initial_estimate.insert(X(i), Pose3(wRc, Point3(0, y, 1.2)))
        graph.add(gtsam.PriorFactorPose3(X(i), Pose3(
            wRc, Point3(0, y, 1.2)), pose_prior_noise))

    # Add initial estimates for Points.
    for j in range(6):
        # Generate initial estimates by back projecting feature points collected at the initial pose
        point_j = back_projection(cal, kp1[j], Pose3(
            wRc, Point3(0, 0, 1.2)), 30, undist)
        initial_estimate.insert(P(j), point_j)

    # Optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # Why this is not working?
    # tol_error = gtsam.reprojectionErrors(graph, result)
    # print("tol_error: ", tol_error)
    error = graph.error(result)
    print("Graph error:", error)

    # print(result)
    plot_with_result(result)

    landmarks = [result.atPoint3(P(i))for i in range(6)]
    poses = [result.atPose3(X(0)), result.atPose3(X(1))]

    return landmarks, poses


class CameraCalibration():
    """Camera calibration"""

    def __init__(self, cal_distort, cam1_features, cam2_features, cal_undistort, cam1_features_undistort, cam2_features_undistort):

        #"""Distorted image variables setting"""
        self.cal = cal_distort
        self.kp1 = cam1_features
        self.kp2 = cam2_features

        #"""Undistorted image variables setting"""
        self.cal_undist = cal_undistort
        self.kp1_undist = cam1_features_undistort
        self.kp2_undist = cam2_features_undistort

    def reprojection_error(self, landmarks, poses, undist=0):
        """
        Calculate the reprojection error of the result from a SfM factor graph
        """
        if undist == 0:
            kp1 = [[self.kp1[i].x(), self.kp1[i].y()]for i in range(6)]
            kp2 = [[self.kp2[i].x(), self.kp2[i].y()]for i in range(6)]
        if undist == 1:
            kp1 = [[self.kp1_undist[i].x(), self.kp1_undist[i].y()]
                   for i in range(6)]
            kp2 = [[self.kp2_undist[i].x(), self.kp2_undist[i].y()]
                   for i in range(6)]
        kp1 = np.expand_dims(np.array(kp1), axis=1)
        kp2 = np.expand_dims(np.array(kp2), axis=1)
        key_point = (kp1, kp2)

        total_error = 0

        if undist == 0:
            mtx = self.cal.K()
            dist = self.cal.k()
        if undist == 1:
            mtx = self.cal_undist.matrix()
            dist = np.array([0.0, 0.0, 0.0, 0.0])

        for i, pose in enumerate(poses):
            # Get wRc, wtc
            R = pose.rotation().matrix()
            t = pose.translation().vector()
            # Invert R,t to obtain cRw, ctw
            R = R.T
            t = -1*np.dot(R, t)

            point_list = [[point.x(), point.y(), point.z()]
                          for point in landmarks]
            point_list = np.expand_dims(np.array(point_list), axis=1)

            imgpoint, _ = cv2.projectPoints(point_list, R, t, mtx, dist)
            error = cv2.norm(
                imgpoint, key_point[i], cv2.NORM_L2) / len(imgpoint)
            total_error += error
        mean_error = total_error/len(poses)
        return mean_error
