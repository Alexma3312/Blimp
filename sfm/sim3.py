"""This is a three dimensional Similarity Transformation module"""
import math
import unittest
import sys

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Rot3

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

    def sim3_pose(self, pose_pairs):
        """
        Generate similarity transform with Pose3 pairs.
        Parameters:
            pose_pairs: N*2 list of Pose3 objects

        Step 1. Calculate rotation matrix
        R_d = R*R_s
        J = /sum_i norm(R_d_i - R_i*R_s_i), J=0
        /sum_i R_i = /sum_i R_d_i*R_s_i.T
        R:  R = (/sum_i Rd * Rs.T) / n

        Step 2. Calculate scale 
        2.1 calculate centroid
        2.2 calculate scale
        s:
          t = td_i - s*R*ts_i ,  t = cd_i - s*R*cs_i
          d_ts_i = ts_i - cs_i, d_td_i = td_i - cd_i
          Hence, s*(R*d_ts) = d_td
          s = (/sum_i d_td_i) / (/sum_j (R*d_ts)_j) 

        Step 3. Calculate translation
            t = cq - sR*cp
        """
        n = len(pose_pairs)
        assert n >= 2  # we need at least two pairs
        R = np.zeros((3, 3))

        s_center = np.zeros(3)
        d_center = np.zeros(3)
        for pose_pair in pose_pairs:
            R += np.dot(pose_pair[1].rotation().matrix(),
                        pose_pair[0].rotation().matrix().T)
            s_center += np.array(pose_pair[0].translation().vector())
            d_center += np.array(pose_pair[1].translation().vector())
        s_center /= n
        d_center /= n
        R /= n

        x = np.zeros((1,1))
        y = np.zeros((1,1))
        for pose_pair in pose_pairs:
            d_ts = pose_pair[0].translation().vector() - s_center
            R_ts = np.dot(R,d_ts)
            d_td = pose_pair[1].translation().vector() - d_center

            x = np.vstack((x,R_ts.reshape(3,1)))
            y = np.vstack((y,d_td.reshape(3,1)))

        # # Cited from https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
        # # We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]].
        # # Now use lstsq to solve for p:
        A = np.hstack([x, np.ones((len(x),1))])
        m, c = np.linalg.lstsq(A, y, rcond=-1)[0]
        s = m

        # calculate translation
        t = d_center - np.dot(s*R,s_center)

        self._R = Rot3(R)
        self._t = Point3(t)
        self._s = s
        print(self._R)
        print(self._t)
        print(self._s)

# unfinished
    def sim3_point(self, point_pairs):
        """
        Generate similarity transform with Point3 pairs.
        """
        n = len(point_pairs)
        assert n >= 3  # we need at least three pairs

        # calculate centroids
        s_center = Point3(0, 0, 0)
        d_center = Point3(0, 0, 0)
        for point_pair in point_pairs:
            s_center = Point3(
                s_center.x() + point_pair[0].x(), s_center.y() + point_pair[0].y(), s_center.z() + point_pair[0].z())
            d_center = Point3(
                d_center.x() + point_pair[1].x(), d_center.y() + point_pair[1].y(), d_center.z() + point_pair[1].z())
        f = 1.0/n
        s_center = Point3(s_center.x()*f, s_center.y()*f, s_center.z()*f)
        d_center = Point3(d_center.x()*f, d_center.y()*f, d_center.z()*f)

        # Add to form H matrix
        H = np.zeros([3, 3])
        for point_pair in point_pairs:
            s_d = Point3(point_pair[0].x()-s_center.x(),
                         point_pair[0].y()-s_center.y(),
                         point_pair[0].z()-s_center.z())
            d_d = Point3(point_pair[1].x()-d_center.x(),
                         point_pair[1].y()-d_center.y(),
                         point_pair[1].z()-d_center.z())
            c += s_d.x() * d_d.x() + s_d.y() * d_d.y()
            s += -s_d.y() * d_d.x() + s_d.x() * d_d.y()
            H += s_d.x()*s_d.x() + s_d.y()*s_d.y()

        # calculate angle, scale, and translation
        theta = math.atan2(s, c)
        R = Rot2.fromAngle(theta)
        determinant = np.linalg.det(np.array([[c, -s], [s, c]]))
        if(determinant >= 0):
            scale = determinant**(1/2)
        elif(determinant < 0):
            scale = -(-determinant)**(1/2)

        t = R.rotate(s_center)
        t = Point2(d_center.x() - scale*t.x(), d_center.y() - scale*t.y())

        self._R = R
        self._t = t
        self._s = scale

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

    def assert_gtsam_equals(self, actual, expected, tol=1e-6):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Not equal:\n{}!={}".format(actual, expected))

    def test_transform(self):
        """Test similarity transform on poses and points."""
        expected_poses, expected_points = self.d_map

        sim3 = Similarity3(Rot3.Ry(math.radians(180)), Point3(4, 6, 10), 2)
        poses, points = self.s_map

        for i, point in enumerate(points):
            point = sim3.point(point)
            self.assert_gtsam_equals(expected_points[i], point)

        for i, pose in enumerate(poses):
            pose = sim3.pose(pose)
            self.assert_gtsam_equals(expected_poses[i], pose)

    def test_map_transform(self):
        """Test similarity transform on a map."""
        expected_poses, expected_points = self.d_map

        sim3 = Similarity3(Rot3.Ry(math.radians(180)), Point3(4, 6, 10), 2)
        actual_poses, actual_points = sim3.map_transform(self.s_map)

        for i, point in enumerate(actual_points):
            self.assert_gtsam_equals(expected_points[i], point)

        for i, pose in enumerate(actual_poses):
            self.assert_gtsam_equals(expected_poses[i], pose)

    def test_align_pose(self):
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
        sim3.align_pose(pose_pairs)

        # Check actual sim3 equals to expected sim3
        expected_R.equals(sim3._R, 0.01)
        self.assertAlmostEqual(sim3._s, expected_s, delta=0.01)
        self.assert_gtsam_equals(sim3._t, expected_t)

    # Unfinished
    def test_sim3_point(self):
        """Test generating similarity transform with Point3 pairs."""
        s_point1 = Point3(1, 1, 0)
        s_point2 = Point3(1, 3, 0)
        s_point3 = Point3(3, 3, 0)

        d_point1 = Point3(2, 8, 10)
        d_point2 = Point3(2, 12, 10)
        d_point3 = Point3(-2, 12, 10)

        point_pairs = [[s_point1, d_point1], [
            s_point2, d_point2], [s_point3, d_point3]]
    
    def test_sim3_pose(self):
        """Test generating similarity transform with Pose3 pairs."""
        # Create expected sim3
        expected_R = Rot3.Ry(math.radians(180))
        expected_s = 2
        expected_t = Point3(4, 6, 10)
        print(expected_R)

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
        assert(expected_R.equals(sim3._R, 1e-6))
        self.assertAlmostEqual(sim3._s, expected_s, delta=1e-6)
        self.assert_gtsam_equals(sim3._t, expected_t)


if __name__ == "__main__":
    unittest.main()
