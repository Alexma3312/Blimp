"""This is a two dimensional Similarity Transformation module"""
import math
import unittest

import gtsam
import numpy as np
from gtsam import Point2, Pose2, Rot2


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

    def align(self, point_pairs):
        """
        Generate similarity transform with Point2 pairs.
        This function is based on gtsam::Pose2::align() 
        *************************************************************************
        It finds the angle using a linear method:
        q = Pose2::transform_from(p) = t + R*p
        Subtract the centroid from each point

        q1 = R*p1+t     -> (q1-c_q) = R * (p1-c_p) 
        c_q = R*c_p+t

        using dp=[dpx;dpy] and q=[dqx;dqy] we have
         |dqx|   |c  -s|     |dpx|     |dpx -dpy|     |c|
         |   | = |     |  *  |   |  =  |        |  *  | | = H_i*cs
         |dqy|   |s   c|     |dpy|     |dpy  dpx|     |s|
        where the Hi are the 2*2 matrices. Then we will minimize the criterion
        J = \sum_i norm(q_i - H_i * cs)
        Taking the derivative with respect to cs and setting to zero we have
        cs = (\sum_i H_i' * q_i)/(\sum H_i'*H_i)

        H_i'*H_i = (\sum_i dpx_i^2+dpy_i^2) |1  0|
                                            |    |
                                            |0  1|

        The hessian is diagonal and just divides by a constant.
        i.e., cos ~ sum(dpx*dqx + dpy*dqy) and sin ~ sum(-dpy*dqx + dpx*dqy)

        The scale is the square root of the determinant of the rotation matrix because the rotation matrix is an orthogonal matrix.

        The translation is then found from the centroids
        as they also satisfy cq = t + sR*cp, hence t = cq - sR*cp
        """

        n = len(point_pairs)
        assert n >= 2  # we need at least two pairs

        # calculate centroids
        s_center = Point2(0, 0)
        d_center = Point2(0, 0)
        for point_pair in point_pairs:
            s_center = Point2(
                s_center.x() + point_pair[0].x(), s_center.y() + point_pair[0].y())
            d_center = Point2(
                d_center.x() + point_pair[1].x(), d_center.y() + point_pair[1].y())
        f = 1.0/n
        s_center = Point2(s_center.x()*f, s_center.y()*f)
        d_center = Point2(d_center.x()*f, d_center.y()*f)

        # calculate cos and sin
        c = 0
        s = 0
        constant = 0
        for point_pair in point_pairs:
            s_d = Point2(point_pair[0].x()-s_center.x(),
                         point_pair[0].y()-s_center.y())
            d_d = Point2(point_pair[1].x()-d_center.x(),
                         point_pair[1].y()-d_center.y())
            c += s_d.x() * d_d.x() + s_d.y() * d_d.y()
            s += -s_d.y() * d_d.x() + s_d.x() * d_d.y()
            constant += s_d.x()*s_d.x() + s_d.y()*s_d.y()

        c /= constant
        s /= constant

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

    def test_transform(self):
        """Test similarity transform on poses and points."""
        expected_poses, expected_points = self.d_map

        sim2 = Similarity2(Rot2(math.radians(90)), Point2(10, 10), 2)
        poses, points = self.s_map
        for i, point in enumerate(points):
            point = sim2.point(point)
            self.assert_gtsam_equals(expected_points[i], point)

        for i, pose in enumerate(poses):
            pose = sim2.pose(pose)
            self.assert_gtsam_equals(expected_poses[i], pose)

    def test_align(self):
        """Test generating similarity transform with Point2 pairs."""
        # Create expected sim2
        expected_R = Rot2(math.radians(90))
        expected_s = 2
        expected_t = Point2(10, 10)

        # Create source points
        s_point1 = Point2(0.5, 0.5)
        s_point2 = Point2(2, 0.5)

        # Create destination points
        d_point1 = Point2(9, 11)
        d_point2 = Point2(9, 14)

        # Align
        point_pairs = [[s_point1, d_point1], [s_point2, d_point2]]
        sim2 = Similarity2()
        sim2.align(point_pairs)

        # Check actual sim2 equals to expected sim2
        expected_R.equals(sim2._R, 0.01)
        self.assertEqual(sim2._s, expected_s)
        self.assert_gtsam_equals(sim2._t, expected_t)

    def test_align_case_1(self):
        """Test generating similarity transform with gtsam pose2 align test case 1 - translation only."""
        # Create expected sim2
        expected_R = Rot2(math.radians(0))
        expected_s = 1
        expected_t = Point2(10, 10)

        # Create source points
        s_point1 = Point2(0, 0)
        s_point2 = Point2(20, 10)

        # Create destination points
        d_point1 = Point2(10, 10)
        d_point2 = Point2(30, 20)

        # Align
        point_pairs = [[s_point1, d_point1], [s_point2, d_point2]]
        sim2 = Similarity2()
        sim2.align(point_pairs)

        # Check actual sim2 equals to expected sim2
        expected_R.equals(sim2._R, 0.01)
        self.assert_gtsam_equals(sim2._t, expected_t)
        self.assertEqual(sim2._s, expected_s)

    def test_align_case_2(self):
        """Test generating similarity transform with gtsam pose2 align test case 2."""
        # Create expected sim2
        expected_R = Rot2(math.radians(90))
        expected_s = 1
        expected_t = Point2(20, 10)

        expected = Pose2(math.radians(90),expected_t)

        # Create source points
        s_point1 = Point2(0, 0)
        s_point2 = Point2(10, 10)

        # Create destination points
        d_point1 = expected.transform_from(s_point1)
        d_point2 = expected.transform_from(s_point2) 
        self.assert_gtsam_equals(d_point1,Point2(20, 10))
        self.assert_gtsam_equals(d_point2,Point2(10, 20))

        # Align
        point_pairs = [[s_point1, d_point1], [s_point2, d_point2]]
        sim2 = Similarity2()
        sim2.align(point_pairs)

        # Check actual sim2 equals to expected sim2
        expected_R.equals(sim2._R, 0.01)
        self.assert_gtsam_equals(sim2._t, expected_t)
        self.assertEqual(sim2._s, expected_s)

    def test_align_case_3(self):
        """Test generating similarity transform with gtsam pose2 align test case 3 - transformation of a triangle."""
        # Create expected sim2
        expected_R = Rot2(math.radians(60))
        expected_s = 1
        expected_t = Point2(10, 10)

        expected = Pose2(math.radians(60),expected_t)

        # Create source points
        s_point1 = Point2(0, 0)
        s_point2 = Point2(10, 0)
        s_point3 = Point2(10,10)

        # Create destination points
        d_point1 = expected.transform_from(s_point1)
        d_point2 = expected.transform_from(s_point2)
        d_point3 = expected.transform_from(s_point3)

        # Align
        point_pairs = [[s_point1, d_point1], [s_point2, d_point2]]
        sim2 = Similarity2()
        sim2.align(point_pairs)

        # Check actual sim2 equals to expected sim2
        expected_R.equals(sim2._R, 0.01)
        self.assert_gtsam_equals(sim2._t, expected_t)
        self.assertAlmostEqual(sim2._s, expected_s,delta = 0.1)

if __name__ == "__main__":
    unittest.main()
