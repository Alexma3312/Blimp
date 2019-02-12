"""
A structure-from-motion example with landmarks

"""
# pylint: disable=invalid-name, E1101

import numpy as np
import math

import gtsam


def createPoints():
    # Create the set of ground-truth landmarks
    points = [gtsam.Point3(10.0, 0.0, 0.0),
              gtsam.Point3(10.0, 5.0, 0.0),
              gtsam.Point3(10.0, 2.5, 2.5),
              gtsam.Point3(10.0, 0.0, 5.0),
              gtsam.Point3(10.0, 5.0, 5.0)]
    return points


def createPoses():
    # Create the set of ground-truth poses
    poses = []
    angle = 0

    for i, y in enumerate([-1, 0, 1]):
        theta = np.radians(-y*angle)
        wRc = gtsam.Rot3(np.array([[0, math.cos(
            theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)
        poses.append(gtsam.Pose3(wRc, gtsam.Point3(0, (y+1)*2.5, 1.5)))

    return poses
