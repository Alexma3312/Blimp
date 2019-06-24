# cSpell: disable=invalid-name
"""
Unit tests for back projection
"""
# pylint: disable=no-name-in-module

import unittest

from gtsam import Cal3_S2, Point2, Point3, Pose3, Rot3
from gtsam.utils.test_case import GtsamTestCase
from utilities.back_projection import back_projection


class TestBackProjection(GtsamTestCase):
    """Unit tests for back projection."""

    def test_back_projection(self):
        """Test back projection"""
        fov, width, height = 60, 1280, 720
        calibration = Cal3_S2(fov, width, height)
        actual = back_projection(calibration, Point2(640, 360), Pose3(
            Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0), Point3()), 20)
        expected = Point3(0, 20, 0)
        self.gtsamAssertEquals(actual, expected)


if __name__ == "__main__":
    unittest.main()
