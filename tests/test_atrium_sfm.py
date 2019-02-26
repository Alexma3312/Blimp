"""Unit Test to test Structure from Motion (SfM)."""
import math
import sys
import unittest

import gtsam
import gtsam.utils.visual_data_generator as generator
import numpy as np
from gtsam import Point2, Point3, Pose3, symbol

from sfm import atrium_sfm, sfm_data

sys.path.append('sfm')


class TestAtriumSFMEample(unittest.TestCase):

    def setUp(self):
        self.sfm = atrium_sfm.AtriumSfm(3, 5, 128, 640, 480)
        self.calibration = self.sfm.calibration
        self.nrCameras = self.sfm.nrCameras
        self.nrPoints = self.sfm.nrPoints

    def assertGtsamEquals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Not equal:\n{}!={}".format(actual, expected))

    def test_back_project(self):
        actual_point = self.sfm.back_project(
            Point2(320, 240), self.calibration, 10)
        expected_point = gtsam.Point3(10, 0, 1.5)
        self.assertGtsamEquals(actual_point, expected_point)

    def test_Atrium_Sfm(self):

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

        result = self.sfm.atrium_sfm(feature_data, 0, 2.5)

        print(result)

        # Compare output poses with ground truth poses
        for i in range(len(poses)):
            pose_i = result.atPose3(symbol(ord('x'), i))
            self.assertGtsamEquals(pose_i, poses[i])

        # Compare output points with ground truth points
        for j in range(len(points)):
            point_j = result.atPoint3(symbol(ord('p'), j))
            self.assertGtsamEquals(point_j, points[j])


if __name__ == "__main__":
    unittest.main()
