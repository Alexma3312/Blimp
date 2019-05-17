import unittest
import gtsam
from gtsam import Point2, Point3, Pose3, Rot3, symbol
from feature_matcher.mapping import *


def load_points():
    """load landmark data"""
    pts3d = []
    with open("feature_matcher/sim_match_data/points.txt") as f:
        pts = f.readlines()
        num_pts = int(pts[0].strip())
        for i in range(num_pts):
            pt = [float(x) for x in pts[i+1].strip().split()]
            pts3d.append(gtsam.Point3(*pt))
    return pts3d


class TestMapping(unittest.TestCase):
    def setUp(self):
        fe = MappingFrontEnd()
        fe.get_all_image_features()
        fe.get_feature_matches()
        fe.initial_estimation()

        self.sfm_result, _, _ = fe.bundle_adjustment()
        # fe.plot_sfm_result(self.sfm_result)

    def assert_gtsam_equals(self, actual, expected, tol=1e-6):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Not equal:\n{}!={}".format(actual, expected))

    def test_poses(self):
        """Test pose output"""
        delta_z = 1
        num_frames = 3
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
        for i in range(num_frames):
            # Create ground truth poses
            expected_pose_i = Pose3(wRc, Point3(0, delta_z*i, 2))
            # Get actual poses
            actual_pose_i = self.sfm_result.atPose3(X(i))
            self.assert_gtsam_equals(actual_pose_i, expected_pose_i)

    def test_points(self):
        """Test landmark output"""
        expected_points = load_points()
        for i, expected_point_i in enumerate(expected_points):
            actual_point_i = self.sfm_result.atPoint3(P(i))
            # print(actual_point_i,expected_point_i)
            self.assert_gtsam_equals(actual_point_i, expected_point_i,1e-4)


if __name__ == "__main__":
    unittest.main()
