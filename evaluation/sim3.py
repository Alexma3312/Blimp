"""This is a three dimensional Similarity Transformation module"""
import math
import sys
import unittest

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Rot3
from gtsam.utils.test_case import GtsamTestCase

sys.path.append("../")


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
        """ Calculate the similarity transform of a Point3 object
        Parameters:
            point - Point3 object
        Returns:
            Point3 object
        """
        return Point3(self._s*np.dot(self._R.matrix(), point.vector())+self._t.vector())

    def pose(self, pose):
        """ Calculate the similarity transform of a Pose3 object
        Parameters:
            pose - Pose3 object
        Returns:
            Pose3 object
        """
        R = self._R.compose(pose.rotation())
        translation = self._s * \
            self._R.rotate(pose.translation()).vector() + self._t.vector()
        return Pose3(R, Point3(translation))

    def rotation_averaging(self, rotations, error=1e-10):
        """ Use the geodesic L2 mean to solve the mean of rotations,
            Ref: http://users.cecs.anu.edu.au/~hongdong/rotationaveraging.pdf (on page 18)
        Parameters:
            rotations - a list of rotations
        Returns:
            the mean rotation matrix
        """
        R = rotations[0]
        n = len(rotations)
        r = np.array([0, 0, 0]).astype(np.float64)
        while(True):
            for R_i in rotations:
                r += Rot3.Logmap(R.inverse().compose(R_i))
            r = r/n
            if(np.linalg.norm(r) < error):
                return R
            R = R.compose(Rot3.Expmap(r))

    def sim3_pose(self, pose_pairs):
        """
        Generate similarity transform with Pose3 pairs.
        Parameters:
            pose_pairs: N*2 list of Pose3 objects, [pose_s, pose_d]

        Step 1. Calculate rotation matrix
            Rd = R*Rs => R_i = Rd_i*Rs_i.T
            R = mean(R_1,R_2,...,R_i)

        Step 2. Calculate scale 
            2.1 calculate centroid
            cs = (/sum_i t_s)/n 
            cd = (/sum_i t_d)/n 

            2.2 calculate scale
            t = td_i - s*R*ts_i ,  t = cd - s*R*cs
            d_ts_i = ts_i - cs, d_td_i = td_i - cd
            Hence, s*(R*d_ts) = d_td

        Step 3. Calculate translation
            t = cd - s*R*cs
        """
        n = len(pose_pairs)
        assert n >= 2, "The input should have at least two pairs of poses"
        R = np.zeros((3, 3))

        # Calculate rotation matrix and centers
        s_center = np.zeros(3)
        d_center = np.zeros(3)
        rotation_list = []
        for pose_pair in pose_pairs:
            rotation_list.append(pose_pair[1].rotation().compose(
                pose_pair[0].rotation().inverse()))
            s_center += pose_pair[0].translation().vector()
            d_center += pose_pair[1].translation().vector()
        s_center /= n
        d_center /= n
        R = self.rotation_averaging(rotation_list).matrix()

        # Calculate scale
        x = 0
        y = 0
        for pose_pair in pose_pairs:
            d_ts = pose_pair[0].translation().vector() - s_center
            R_ts = np.dot(R, d_ts)
            d_td = pose_pair[1].translation().vector() - d_center
            y += np.dot(d_td.T, R_ts)
            x += np.dot(R_ts.T, R_ts)
        s = y/x

        # Calculate translation
        t = d_center - np.dot(s*R, s_center)

        self._R = Rot3.ClosestTo(R)
        self._t = Point3(t)
        self._s = s

    def sim3_point(self, point_pairs):
        """
        Generate similarity transform with Point3 pairs.
        refer to: http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Zinsser05-PSR.pdf
        """
        n = len(point_pairs)
        assert n >= 3, "The input should have at least three pairs of points"

        # calculate centroids
        s_center = np.zeros(3)
        d_center = np.zeros(3)
        for point_pair in point_pairs:
            s_center += point_pair[0].vector()
            d_center += point_pair[1].vector()
        s_center /= n
        d_center /= n

        # Add to form H matrix
        H = np.zeros((3, 3))
        for point_pair in point_pairs:
            s_d = (point_pair[0].vector() - s_center).reshape(3,1)
            d_d = (point_pair[1].vector() - d_center).reshape(3,1)
            H += np.dot(d_d, s_d.T)
        # calculate angle, scale, and translation
        R = Rot3.ClosestTo(H).matrix()

        # Calculate scale
        x = 0
        y = 0
        for point_pair in point_pairs:
            d_ts = point_pair[0].vector() - s_center
            R_ts = np.dot(R, d_ts)
            d_td = point_pair[1].vector() - d_center
            y += np.dot(d_td.T, R_ts)
            x += np.dot(R_ts.T, R_ts)
        s = y/x

        # Calculate translation
        t = d_center - np.dot(s*R, s_center)

        self._R = Rot3.ClosestTo(R)
        self._t = Point3(t)
        self._s = s

    def map_transform(self, s_map):
        """
        Calculate the similarity transform of a Map, included Pose3 and Point3
        Parameters:
            s_map: source map, a two demension tuple include a list of Pose3 and a list of Point3 
        Returns:
            d_map: destination map, a two demension tuple include a list of Pose3 and a list of Point3
        """
        d_poses = []
        d_points = []

        # Transform all the poses in the map
        for pose in s_map[0]:
            pose = self.pose(pose)
            d_poses.append(pose)

        # Transform all the points in the map
        for point in s_map[1]:
            point = self.point(point)
            d_points.append(point)

        d_map = (d_poses, d_points)

        return d_map


class TestSimilarity3(GtsamTestCase):

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

    def test_transform(self):
        """Test similarity transform on poses and points."""
        expected_poses, expected_points = self.d_map

        sim3 = Similarity3(Rot3.Ry(math.radians(180)), Point3(4, 6, 10), 2)
        poses, points = self.s_map

        for i, point in enumerate(points):
            point = sim3.point(point)
            self.gtsamAssertEquals(expected_points[i], point)

        for i, pose in enumerate(poses):
            pose = sim3.pose(pose)
            self.gtsamAssertEquals(expected_poses[i], pose)

    def test_map_transform(self):
        """Test similarity transform on a map."""
        expected_poses, expected_points = self.d_map

        sim3 = Similarity3(Rot3.Ry(math.radians(180)), Point3(4, 6, 10), 2)
        actual_poses, actual_points = sim3.map_transform(self.s_map)

        for i, point in enumerate(actual_points):
            self.gtsamAssertEquals(expected_points[i], point)

        for i, pose in enumerate(actual_poses):
            self.gtsamAssertEquals(expected_poses[i], pose)

    def test_sim3_point(self):
        """Test generating similarity transform with Point3 pairs."""
        # Create expected sim3
        expected_R = Rot3.Rz(math.radians(-90))
        expected_s = 2
        expected_t = Point3(6, 8, 10)

        # Create source points
        s_point1 = Point3(0, 0, 0)
        s_point2 = Point3(3, 0, 0)
        s_point3 = Point3(3, 0, 4)

        # Create destination points
        sim3 = Similarity3()
        sim3._R = expected_R
        sim3._t = expected_t
        sim3._s = expected_s
        d_point1 = sim3.point(s_point1)
        d_point2 = sim3.point(s_point2)
        d_point3 = sim3.point(s_point3)

        # Align
        point_pairs = [[s_point1, d_point1], [
            s_point2, d_point2], [s_point3, d_point3]]
        sim3 = Similarity3()
        sim3.sim3_point(point_pairs)

        # Check actual sim3 equals to expected sim3
        self.gtsamAssertEquals(sim3._R, expected_R)
        self.assertAlmostEqual(sim3._s, expected_s, delta=1e-6)
        self.gtsamAssertEquals(sim3._t, expected_t)

    def test_sim3_point_case2(self):
        """Test generating similarity transform with Point3 pairs."""
        # Create expected sim3
        expected_R = Rot3()
        expected_s = 1
        expected_t = Point3(10, 10, 0)

        # Create source points
        s_point1 = Point3(0, 0, 0)
        s_point2 = Point3(20, 10, 0)
        s_point3 = Point3(10, 20, 0)

        # Create destination points
        d_point1 = Point3(10, 10, 0)
        d_point2 = Point3(30, 20, 0)
        d_point3 = Point3(20, 30, 0)

        # Align
        point_pairs = [[s_point1, d_point1], [
            s_point2, d_point2], [s_point3, d_point3]]
        sim3 = Similarity3()
        sim3.sim3_point(point_pairs)

        # Check actual sim3 equals to expected sim3
        self.gtsamAssertEquals(sim3._R, expected_R)
        self.assertAlmostEqual(sim3._s, expected_s, delta=1e-6)
        self.gtsamAssertEquals(sim3._t, expected_t)

    def test_sim3_point_case3(self):
        """Test generating similarity transform with Point3 pairs."""
        # Create expected sim3
        expected_R = Rot3.RzRyRx(0.3, 0.2, 0.1)
        expected_s = 1
        expected_t = Point3(20, 10, 5)

        # Create source points
        s_point1 = Point3(0, 0, 1)
        s_point2 = Point3(10, 0, 2)
        s_point3 = Point3(20, -10, 30)

        # Create destination points
        sim3 = Similarity3()
        sim3._R = expected_R
        sim3._t = expected_t
        sim3._s = expected_s
        d_point1 = sim3.point(s_point1)
        d_point2 = sim3.point(s_point2)
        d_point3 = sim3.point(s_point3)

        # Align
        point_pairs = [[s_point1, d_point1], [
            s_point2, d_point2], [s_point3, d_point3]]
        sim3 = Similarity3()
        sim3.sim3_point(point_pairs)

        # Check actual sim3 equals to expected sim3
        self.gtsamAssertEquals(sim3._R, expected_R)
        self.assertAlmostEqual(sim3._s, expected_s, delta=1e-6)
        self.gtsamAssertEquals(sim3._t, expected_t)

    def test_sim3_pose(self):
        """Test generating similarity transform with Pose3 pairs."""
        # Create expected sim3
        expected_R = Rot3.Ry(math.radians(180))
        expected_s = 2
        expected_t = Point3(4, 6, 10)

        # Create source poses
        s_pose1 = Pose3(
            Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), Point3(0, 0, 0))
        s_pose2 = Pose3(
            Rot3(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])), Point3(4, 0, 0))

        # Create destination poses
        d_pose1 = Pose3(
            Rot3(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])), Point3(4, 6, 10))
        d_pose2 = Pose3(
            Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])), Point3(-4, 6, 10))

        # Align
        pose_pairs = [[s_pose1, d_pose1], [s_pose2, d_pose2]]
        sim3 = Similarity3()
        sim3.sim3_pose(pose_pairs)

        # Check actual sim3 equals to expected sim3
        self.gtsamAssertEquals(sim3._R, expected_R)
        self.assertAlmostEqual(sim3._s, expected_s, delta=1e-6)
        self.gtsamAssertEquals(sim3._t, expected_t)

    def test_rotation_averaging(self):
        """test rotation averaging."""
        rotations = [Rot3(), Rot3.Rx(math.radians(90)),
                     Rot3.Rx(math.radians(180))]
        expected_rotation = Rot3.Rx(math.radians(90))
        sim3 = Similarity3()
        actual_rotation = sim3.rotation_averaging(rotations)
        self.gtsamAssertEquals(expected_rotation, actual_rotation)


if __name__ == "__main__":
    unittest.main()
