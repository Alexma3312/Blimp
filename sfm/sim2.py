"""This is a two dimensional Similarity Transformation module"""
import unittest

import gtsam
import numpy as np
from gtsam import Point2, Pose2, Rot2
import math


class Similarity2(object):
    """
    Similarity transform in 2D.
    """

    def __init__(self, rotation=Rot2(), translation=Point2(0, 0), scale=1):
        """Construct from rotation, translation, and scale."""
        self._R = rotation
        self._t = translation
        self._s = scale

    def point(self, point):
        """ Calculate the similarity transform of a Point2
        Parameters:
            point - Point2 object
        Returns:
            Point2 object
        """
        return Point2(self._s*np.dot(self._R.matrix(), point.vector())+self._t.vector())

    def pose(self, pose):
        """ Calculate the similarity transform of a Pose2
        Parameters:
            pose - Pose2 object
        Returns:
            Pose2 object
        """
        R = self._R.compose(pose.rotation())
        translation = self._s * \
            self._R.rotate(pose.translation()).vector() + self._t.vector()
        return Pose2(R.theta(), Point2(translation))


    def map_transform(self):
        return


class TestSimilarity2(unittest.TestCase):

    def setUp(self):
        # Create poses for the source map
        s_pose1 = Pose2(0, Point2(0, 0))
        s_pose2 = Pose2(math.radians(-90), Point2(0, 2.5))
        s_poses = [s_pose1, s_pose2]
        # Create points for the source map
        s_point1 = Point2(0.5, 0.5)
        s_point2 = Point2(2, 0.5)
        s_point3 = Point2(2, 2)
        s_point4 = Point2(0.5, 2)
        s_points = [s_point1, s_point2, s_point3, s_point4]
        # Create the source map
        self.s_map = (s_poses, s_points)

        # Create poses for the destination map
        d_pose1 = Pose2(math.radians(90), Point2(10, 10))
        d_pose2 = Pose2(0, Point2(5, 10))
        d_poses = [d_pose1, d_pose2]
        # Create points for the destination map
        d_point1 = Point2(9, 11)
        d_point2 = Point2(9, 14)
        d_point3 = Point2(6, 14)
        d_point4 = Point2(6, 11)
        d_points = [d_point1, d_point2, d_point3, d_point4]
        d_map = (d_poses, d_points)
        # Create the destination map
        self.d_map = (d_poses, d_points)

    def assert_gtsam_equals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Not equal:\n{}!={}".format(actual, expected))

    def test_map_transform(self):
        """Test similarity transform on a map."""
        expected_poses, expected_points = self.d_map

        sim2 = Similarity2(Rot2(math.radians(90)), Point2(10, 10), 2)
        poses, points = self.s_map
        for i, point in enumerate(points):
            point = sim2.point(point)
            print(point)
            self.assert_gtsam_equals(expected_points[i], point)

        for i, pose in enumerate(poses):
            pose = sim2.pose(pose)
            print(pose)
            self.assert_gtsam_equals(expected_poses[i], pose)


if __name__ == "__main__":
    unittest.main()
