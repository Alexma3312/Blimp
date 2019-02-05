"""Test incremental visual SAM."""

import unittest

import math
import gtsam
from gtsam import Point2, Point3, Pose3
import numpy as np

import cv2

import TrajectoryEstimator


class TestVisualISAMExample(unittest.TestCase):

    def assertGtsamEquals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Poses are not equal:\n{}!={}".format(actual, expected))

    def test_process_first_frame(self):
        # Horizontal FOV = 160
        fov_in_degrees, w, h = 160, 160, 120
        calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)

        sfm = TrajectoryEstimator.SFM(calibration)
        features = TrajectoryEstimator.Features(np.array([[80, 60]], dtype=np.float))
        sfm.process_first_frame(features)
        self.assertGtsamEquals(sfm.pose(0), Pose3())

        # calculate expected point
        pn = calibration.calibrate(Point2(80.0, 60.0))  # normalized
        u, v = pn.x(), pn.y()
        depth = 10
        expected_point = Point3(depth*u, depth*v, depth)
        self.assertGtsamEquals(sfm.point(0), expected_point)


if __name__ == "__main__":
    unittest.main()
