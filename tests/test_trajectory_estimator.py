"""Test incremental visual SAM."""

import unittest

import math
import gtsam
from gtsam import Point2, Point3, Pose3, symbol
import numpy as np
from SFM import SFMdata

import cv2

from AtriumControl.TrajectoryEstimator import TrajectoryEstimator


class TestTrajectoryEstimator(unittest.TestCase):

    def setUp(self):

        # Import SFM map output as Trajectory Estimator map input
        self.atrium_map = SFMdata.createPoints()

        # Create an input image list
        self.Images = SFMdata.ReadImages()

        # Use the poses in SFM as the actual estimate trajectory
        self.past_trajectory_estimate = SFMdata.createPoses()

    def assertGtsamEquals(self, actual, expected, tol=2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Poses are not equal:\n{}!={}".format(actual, expected))

    def assertGtsamListEqual(self, actual_list, expect_list):
        assert len(actual_list) == len(expect_list)
        for i in range(len(actual_list)):
            self.assertGtsamEquals(actual_list[i], expect_list[i])

    def test_first_frame_process(self):

        actual_first_frame_feature_points = []
        actual_first_frame_landmarks = self.atrium_map

        # Initialize trajectory estimator
        trajectory_estimator = TrajectoryEstimator(self.atrium_map)

        # Initialize factor graph
        trajectory_estimator.initial_iSAM()

        for point in self.atrium_map:
            camera = gtsam.PinholeCameraCal3_S2(
                trajectory_estimator.estimate_trajectory[0], trajectory_estimator.calibration)
            actual_first_frame_feature_points.append(camera.project(point))

        current_estimate = trajectory_estimator.first_frame_process(
            actual_first_frame_landmarks, actual_first_frame_feature_points)

        # Compare output poses with ground truth poses
        pose_0 = current_estimate.atPose3(symbol(ord('x'), 0))
        self.assertGtsamEquals(
            pose_0, trajectory_estimator.estimate_trajectory[0])

        # Compare output points with ground truth points
        for j, point in enumerate(self.atrium_map):
            point_j = current_estimate.atPoint3(symbol(ord('p'), j))
            self.assertGtsamEquals(point_j, point[j])

    def test_feature_point_matching(self):
        return

    def test_trajectory_estimator(self):

        # Initialize trajectory estimator
        trajectory_estimator = TrajectoryEstimator(self.atrium_map)

        # Initialize factor graph
        trajectory_estimator.initial_iSAM()

        # Traverse through each image to generate trajectory poses
        for i, image in enumerate(self.Images):

            # Use Superpoint to generate all feature points
            # trajectory_estimator.superpoint_generator(image)

            # Find all manually selected feature points from all Superpoint features by matching descriptors
            # trajectory_estimator.feature_point_extraction(self, superpoints, descriptors)

            feature_data = []

            if(i == 0):
                for point in enumerate(self.atrium_map):
                    camera = gtsam.PinholeCameraCal3_S2(
                        trajectory_estimator.estimate_trajectory[0], trajectory_estimator.calibration)
                    feature_data.append(camera.project(point))
                trajectory_estimator.first_frame_process(
                    self.atrium_map, feature_data)
            else:
                for point in enumerate(self.atrium_map):
                    camera = gtsam.PinholeCameraCal3_S2(
                        trajectory_estimator.estimate_trajectory[i-1], trajectory_estimator.calibration)
                    feature_data.append(camera.project(point))
                trajectory_estimator.feature_point_matching(
                    self.atrium_map, feature_data)
                trajectory_estimator.update_iSAM(feature_data)

        self.assertGtsamListEqual([Point3(0, 0, 0), Point3(2, 2, 3)], [
                                  Point3(0, 0, 0), Point3(2, 2, 3)])


if __name__ == "__main__":
    unittest.main()
