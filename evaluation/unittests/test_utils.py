"""Test utils"""
import math
import unittest

from gtsam import Point3, Pose3, Rot3

from evaluation.utils import *


class testUtils(unittest.TestCase):
    def test_angle(self):
        """Test angle"""
        R1 = Rot3()
        R2 = Rot3()
        expected_angle = 0
        actual_angle = angle(R1, R2)
        self.assertEqual(expected_angle, actual_angle)
        R1 = Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        R2 = Rot3(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
        expected_angle = 90
        actual_angle = angle(R1, R2)
        self.assertEqual(expected_angle, actual_angle)
        # Get expected angle from online conversion website: https://www.andre-gaschler.com/rotationconverter/
        R1 = Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        R2 = Rot3(np.array([[0.7071068,  0.0000000,  0.7071068], [
                  0.5000000,  0.7071068, -0.5000000], [-0.5000000,  0.7071068,  0.5000000]]))
        expected_angle = 62.799
        actual_angle = angle(R1, R2)
        self.assertAlmostEqual(expected_angle, actual_angle, delta=0.1)
        # Most commonly used method to calculate the rotation difference
        # diff_r = r1*r2';
        # acos((trace(diff_r)-1)/2)*180/pi
        different_R = R1.between(R2).matrix()
        trace_R = different_R[0][0]+different_R[1][1]+different_R[2][2]
        actual_angle = math.acos((trace_R-1)/2)*180/math.pi
        self.assertAlmostEqual(expected_angle, actual_angle, delta=0.1)

    def test_difference(self):
        R1 = Rot3()
        R2 = Rot3()
        expected_angle = 0
        t1 = Point3(4, 5, 0)
        t2 = Point3(1, 1, 0)
        expected_distance = 5.0
        actual_distance, actual_angle = difference(
            Pose3(R1, t1), Pose3(R2, t2))
        self.assertEqual(expected_distance, actual_distance)
        self.assertEqual(expected_angle, actual_angle)
        R1 = Rot3()
        R2 = Rot3.Ry(math.radians(90))
        expected_angle = 90
        t1 = Point3(4, 1, 1)
        t2 = Point3(1, 1, 0)
        expected_distance = 4.0
        actual_distance, actual_angle = difference(
            Pose3(R1, t1), Pose3(R2, t2))
        self.assertEqual(expected_distance, actual_distance)
        self.assertEqual(expected_angle, actual_angle)

    def test_ate(self):
        diffs = [1,2,3]
        actual_ate = ate(diffs)
        expected_ate = (14/3)**(1/2)
        self.assertAlmostEqual(expected_ate, actual_ate, delta=0.001)


if __name__ == "__main__":
    unittest.main()
