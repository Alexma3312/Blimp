"""Test Constant Speed"""
import math
import unittest

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Rot2, Rot3
from gtsam.utils.test_case import GtsamTestCase
from scipy.spatial.transform import Rotation as R


def inverse(rot):
    return rot.inverse()


def constant_speed(trajectory):
    """Use constant speed model to estimate current pose."""
    assert len(
        trajectory) >= 2, "Trajectory has to have at least 2 elements to use constant speed model."
    T_w2 = trajectory[-1]
    T_w1 = trajectory[-2]
    T_12 = T_w1.between(T_w2)
    # T_12 = T_w1.inverse().compose(T_w2)
    T_w3 = T_w2.compose(T_12)
    R_w3 = T_w3.rotation()
    R_w3 = Rot3.ClosestTo(R_w3.matrix())
    T_w3 = Pose3(R_w3, T_w3.translation())
    return T_w3


class Test(GtsamTestCase):
    """Test methods."""

    def test_constant_speed_Pose3(self):
        """Test constant speed."""
        S0 = Pose3()
        degree = 1
        S1 = Pose3(Rot3.Ry(math.radians(degree)),Point3(0,0,0))
        trajectory = [S0,S1]
        trajectory_degree = []
        for i in range(1,2000):
            new_pose = constant_speed(trajectory)
            trajectory.append(new_pose)
            expected_new_pose = Pose3(Rot3.Ry(math.radians(degree+i*degree)),Point3(0,0,0))
            print(i)
            self.gtsamAssertEquals(new_pose, expected_new_pose, 1e-11)
            # self.gtsamAssertEquals(new_pose.rotation().inverse().compose(new_pose.rotation()), Rot3(), 1e-12)

if __name__ == "__main__":
    unittest.main()
