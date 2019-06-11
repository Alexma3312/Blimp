from sfm import sfm, sfm_data, sim3
"""Unit Test to test Structure from Motion (SfM)."""
import math
import sys
import unittest

import gtsam
import gtsam.utils.visual_data_generator as generator
import numpy as np
from gtsam import Point2, Point3, Pose3, symbol

# Add parent directory into sys.
# This line has to be in front of sfm import.
sys.path.append('../')


class TestSfM(unittest.TestCase):

    def setUp(self):
        self.sfm = sfm.SfM(3, 5, 128, 640, 480)
        self.calibration = self.sfm.calibration
        self.nrCameras = self.sfm.nrCameras
        self.nrPoints = self.sfm.nrPoints
        self.rotation_error = 0.1
        self.translation_error = 0.5

    def assert_gtsam_equals(self, actual, expected, tol=1e-6):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Not equal:\n{}!={}".format(actual, expected))

    def test_back_project(self):
        """Test back project function. """
        actual_point = self.sfm.back_project(
            Point2(320, 240), self.calibration, 10)
        expected_point = gtsam.Point3(10, 0, 1.5)
        self.assert_gtsam_equals(actual_point, expected_point)

    def test_bundle_adjustment(self):
        """Test Structure from Motion bundle adjustment solution with accurate prior value."""

        # Create the set of ground-truth landmarks
        expected_points = sfm_data.create_points()

        # Create the set of ground-truth poses
        expected_poses = sfm_data.create_poses()

        # Create the nrCameras*nrPoints feature data input for  atrium_sfm()
        feature_data = sfm_data.Data(self.nrCameras, self.nrPoints)

        # Project points back to the camera to generate synthetic key points
        for i, pose in enumerate(expected_poses):
            for j, point in enumerate(expected_points):
                feature_data.J[i][j] = j
                camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
                feature_data.Z[i][j] = camera.project(point)

        result = self.sfm.bundle_adjustment(feature_data,2.5, 0, 0)

        # Compare output poses with ground truth poses
        for i in range(len(expected_poses)):
            pose_i = result.atPose3(symbol(ord('x'), i))
            self.assert_gtsam_equals(pose_i, expected_poses[i])

        # Compare output points with ground truth points
        for j in range(len(expected_points)):
            point_j = result.atPoint3(symbol(ord('p'), j))
            self.assert_gtsam_equals(point_j, expected_points[j])

    def test_bundle_adjustment_with_error(self):
        """Test Structure from Motion solution with ill estimated prior value."""

        # Create the set of actual points and poses
        actual_points = []
        actual_poses = []

        # Create the set of ground-truth landmarks
        points = sfm_data.create_points()

        # Create the set of ground-truth poses
        poses = sfm_data.create_poses()

        # Create the nrCameras*nrPoints feature point data input for  atrium_sfm()
        feature_data = sfm_data.Data(self.nrCameras, self.nrPoints)

        # Project points back to the camera to generate feature points
        for i, pose in enumerate(poses):
            for j, point in enumerate(points):
                feature_data.J[i][j] = j
                camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
                feature_data.Z[i][j] = camera.project(point)

        result = self.sfm.bundle_adjustment(
            feature_data, 2.5, self.rotation_error, self.translation_error)

        # Similarity transform
        s_poses = []
        s_points = []
        _sim3 = sim3.Similarity3()

        for i in range(len(poses)):
            pose_i = result.atPose3(symbol(ord('x'), i))
            s_poses.append(pose_i)

        for j in range(len(points)):
            point_j = result.atPoint3(symbol(ord('p'), j))
            s_points.append(point_j)

        pose_pairs = [[s_poses[2], poses[2]], [s_poses[1], poses[1]], [s_poses[0], poses[0]]]

        _sim3.sim3_pose(pose_pairs)

        s_map = (s_poses, s_points)
        actual_poses, actual_points = _sim3.map_transform(s_map)

        # Compare output poses with ground truth poses
        for i, pose in enumerate(actual_poses):
            self.assert_gtsam_equals(pose, poses[i],1e-2)

        # Compare output points with ground truth points
        for i, point in enumerate(actual_points):
            self.assert_gtsam_equals(point, points[i], 1e-2)


if __name__ == "__main__":
    unittest.main()
