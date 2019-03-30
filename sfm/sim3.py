"""This is a three dimensional Similarity Transformation module"""
import math
import unittest

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Rot3


class Similarity3(object):
    """
    Similarity transform in 3D.
    """

    def __init__(self, rotation=Rot3(), translation=Point3(0, 0, 0), scale=1):
        """Construct from rotation, translation, and scale."""
        self._R = rotation
        self._t = translation
        self._s = scale

    def point(self, point):
        """ Calculate the similarity transform of a Point3
        Parameters:
            point - Point3 object
        Returns:
            Point3 object
        """
        return Point3(self._s*np.dot(self._R.matrix(), point.vector())+self._t.vector())

    def pose(self, pose):
        """ Calculate the similarity transform of a Pose3
        Parameters:(
            pose - Pose3 object
        Returns:
            Pose3 object
        """
        R = self._R.compose(pose.rotation())
        translation = self._s * \
            self._R.rotate(pose.translation()).vector() + self._t.vector()
        return Pose3(R, Point3(translation))

    def align_pose(self, pose_pairs):
        """
        Generate similarity transform with Pose3 pairs.

        R:  Rs.T * Rd = R

        s,t:
          t = td_1 - s*R*ts_1
          t = td_2 - s*R*ts_2
          s*(R*d_ts) = d_td 
        """
        n = len(pose_pairs)
        assert n >= 2  # we need at least two pairs

        # calculate rotation matrix Rs.T * Rd = R
        R = np.dot(pose_pairs[0][0].rotation().matrix().T,
                   pose_pairs[0][1].rotation().matrix())

        # calculate scale
        d_ts = pose_pairs[0][0].translation().vector(
        ) - pose_pairs[1][0].translation().vector()
        d_td = pose_pairs[0][1].translation().vector(
        ) - pose_pairs[1][1].translation().vector()
        s = d_td[0]/np.dot(R, d_ts)[0]

        # calculate translation
        t = pose_pairs[0][1].translation().vector(
        ) - s*np.dot(R, pose_pairs[0][0].translation().vector())

        self._R = Rot3(R)
        self._t = Point3(t)
        self._s = s

    def align_point(self, point_pairs):
        """
        Generate similarity transform with Point3 pairs.
        """
        n = len(point_pairs)
        assert n >= 3  # we need at least three pairs


class TestSimilarity2(unittest.TestCase):

    def setUp(self):
        # Create poses for the source map
        s_pose1 = Pose3(
            Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), Point3(0, 0, 0))
        s_pose2 = Pose3(
            Rot3(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])), Point3(4, 0, 0))
        s_poses = [s_pose1, s_pose2]
        # Create points for the source map
        s_point1 = Point3(1, 1, 0)
        s_point2 = Point3(1, 3, 0)
        s_point3 = Point3(3, 3, 0)
        s_point4 = Point3(3, 1, 0)
        s_point5 = Point3(2, 2, 1)
        s_points = [s_point1, s_point2, s_point3, s_point4, s_point5]
        # Create the source map
        self.s_map = (s_poses, s_points)

        # Create poses for the destination map
        d_pose1 = Pose3(
            Rot3(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])), Point3(4, 6, 10))
        d_pose2 = Pose3(
            Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])), Point3(-4, 6, 10))
        d_poses = [d_pose1, d_pose2]
        # Create points for the destination map
        d_point1 = Point3(2, 8, 10)
        d_point2 = Point3(2, 12, 10)
        d_point3 = Point3(-2, 12, 10)
        d_point4 = Point3(-2, 8, 10)
        d_point5 = Point3(0, 10, 8)
        d_points = [d_point1, d_point2, d_point3, d_point4, d_point5]
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

        sim3 = Similarity3(Rot3.Ry(math.radians(180)), Point3(4, 6, 10), 2)
        poses, points = self.s_map

        for i, point in enumerate(points):
            point = sim3.point(point)
            self.assert_gtsam_equals(expected_points[i], point)

        for i, pose in enumerate(poses):
            pose = sim3.pose(pose)
            self.assert_gtsam_equals(expected_poses[i], pose)

    def test_align_pose(self):
        """Test generating similarity transform with Pose3 pairs."""
        s_pose1 = Pose3(
            Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), Point3(0, 0, 0))
        s_pose2 = Pose3(
            Rot3(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])), Point3(4, 0, 0))

        d_pose1 = Pose3(
            Rot3(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])), Point3(4, 6, 10))
        d_pose2 = Pose3(
            Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])), Point3(-4, 6, 10))

        pose_pairs = [[s_pose1, d_pose1], [s_pose2, d_pose2]]

        sim3 = Similarity3()
        sim3.align_pose(pose_pairs)

        expected_R = Rot3.Ry(math.radians(180))
        expected_s = 2
        expected_t = Point3(4, 6, 10)

        expected_R.equals(sim3._R, 0.01)
        self.assertEqual(sim3._s, expected_s)
        self.assert_gtsam_equals(sim3._t, expected_t)

    def test_align_point(self):
        """Test generating similarity transform with Point3 pairs."""
        s_point1 = Point3(1, 1, 0)
        s_point2 = Point3(1, 3, 0)
        s_point3 = Point3(3, 3, 0)

        d_point1 = Point3(2, 8, 10)
        d_point2 = Point3(2, 12, 10)
        d_point3 = Point3(-2, 12, 10)

        point_pairs = [[s_point1, d_point1], [
            s_point2, d_point2], [s_point3, d_point3]]


if __name__ == "__main__":
    unittest.main()
