"""
SO(n): We no longer use SO3 and SO4 fixed types, *only* SOn
Authors: Frank Dellaert
"""
# pylint: disable=import-error, no-name-in-module, no-member, invalid-name

import time
from typing import Callable

import gtsam
import numpy as np
from gtsam import Pose3, Rot3, Values, Point3


def angle(R1, R2):
    """Calculate angle of two given rotations, in degrees.
    R1, R2: gtsam.Rot3
    """
    return np.degrees(np.linalg.norm(Rot3.Logmap(R1.between(R2))))


def distance(t1, t2):
    dist_sqaure = (t1.x() - t2.x())**(2) + (t1.y() -
                                            t2.y())**(2) + (t1.z() - t2.z())**(2)
    dist = dist_sqaure**(1/2)
    return dist


expected_r1 = Rot3(np.array([[-0.343845, 0.0768681, 0.935875],
                             [-0.0719647, 0.991556, -0.107882], [-0.936265, -0.104444, -0.335409]]))
actual_r1 = Rot3(np.array([[-0.320338, 0.091935,   0.942832], [-0.0712105,
                                                               0.990127,  -0.120741], [-0.944623,  -0.105818,  -0.310628]]))
expected_t1 = Point3(-2.18452, 0.365463, 4.46998)
actual_t1 = Point3(-2.1872, 0.362101, 4.3731)

angle1 = angle(expected_r1, actual_r1)
dist1 = distance(expected_t1, actual_t1)
print(angle1, dist1)

expected_r2 = Rot3(np.array([[0.308414, 0.0991402,   0.946072],
                             [-0.063629,  0.994477, -0.0834699], [-0.949122, -0.0344543,   0.313019]]))
actual_r2 = Rot3(np.array([[0.29929, 0.10453, 0.948419], [-0.077006,
                                                          0.993385, -0.0851854], [-0.95105, -0.0475388,    0.30536]]))
expected_t2 = Point3(-3.30885, 0.245563, 3.36012)
actual_t2 = Point3(-3.36179, 0.255783, 3.35059)

angle2 = angle(expected_r2, actual_r2)
dist2 = distance(expected_t2, actual_t2)
print(angle2, dist2)

expected_r3 = Rot3(np.array([[0.598184,   0.025366,   0.800957],
                             [-0.0299209,   0.999509, -0.00930806], [-0.8008,  -0.0183974,    0.598649]]))
actual_r3 = Rot3(np.array([[0.600257,  0.0333181,   0.799113], [-0.0137419,
                                                                0.999414, -0.0313472], [-0.799689, 0.00783498,   0.600363]]))
expected_t3 = Point3(-2.36902, 0.180154, 1.94036)
actual_t3 = Point3(-2.28832, 0.178058, 1.97821)

angle3 = angle(expected_r3, actual_r3)
dist3 = distance(expected_t3, actual_t3)
print(angle3, dist3)
