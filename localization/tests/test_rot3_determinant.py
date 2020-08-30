"""Test Rot3 Determinant"""
import math
import unittest

import numpy as np
from gtsam import Rot3
from gtsam.utils.test_case import GtsamTestCase


class Test(GtsamTestCase):
    """Test methods."""

    @unittest.skip("test_rot3_determinant_compose")
    def test_rot3_determinant_compose(self):
        """Test rot3 determinant."""
        degree = 1
        R_w0 = Rot3()
        R = Rot3.Ry(math.radians(degree))
        for i in range(1,360):
            R_w1 = R_w0.compose(R)
            R_w0 = R_w1
            R_det = np.linalg.det(R_w1.matrix())
            expected_rotation = Rot3.Ry(math.radians(i*degree))
            expected_determinant = 1.0
            print(i, ' determinant error =',math.sqrt(R_det-expected_determinant))
            # self.assertAlmostEqual(R_det, expected_determinant, delta=1e-5)
            self.gtsamAssertEquals(R_w1, expected_rotation, 1e-5)

    def test_rot3_determinant_between(self):
        """Test rot3 determinant."""
        degree = 1
        R_w0 = Rot3()
        R_w1 = Rot3.Ry(math.radians(degree))
        for i in range(2,360):
            # R_01 = R_w0.between(R_w1)
            # R_w2 = R_w1.compose(R_01)
            # R_w2 = R_w1 * R_w0.inverse() * R_w1
            R_w2 = R_w1.compose(R_w0.inverse().compose(R_w1))
            R_w0 = R_w1
            R_w1 = R_w2
            R_det = np.linalg.det(R_w2.matrix())
            expected_rotation = Rot3.Ry(math.radians(i*degree))
            expected_determinant = 1.0
            print(i-1, ' determinant error =',math.sqrt(R_det-expected_determinant))
            self.assertAlmostEqual(R_det, expected_determinant, delta=1e-5)
            # self.gtsamAssertEquals(R_w2, expected_rotation, 1e-5)

if __name__ == "__main__":
    unittest.main()
