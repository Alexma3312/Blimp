"""
Test to recover the initial pose.
"""
import numpy as np

# cSpell: disable
# pylint: disable=no-name-in-module,wrong-import-order, no-member,ungrouped-imports, invalid-name
from gtsam import Cal3_S2, Point3, Pose3, Rot3
from localization.trajectory_estimator import TrajectoryEstimator
import unittest
from gtsam.utils.test_case import GtsamTestCase
from utilities.plotting import plot_trajectory_verification
from feature_matcher.mapping_result_helper import load_poses_from_file


class TestInitialPoseRecover(GtsamTestCase):
    """Unit tests for initial pose recover."""

    def setUp(self):
        # Camera to world rotation
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
        self.initial_pose = Pose3(wRc, Point3(0, 0, 1.2))
        self.directory_name = "localization/data_sets/Klaus_1/"

        calibration = Cal3_S2(
            fx=232.0542, fy=252.8620, s=0, u0=325.3452, v0=240.2912)

        image_size = (640, 480)
        l2_thresh = 0.7
        distance_thresh = [5, 5]
        self.trajectory_estimator = TrajectoryEstimator(
            self.initial_pose, self.directory_name, calibration, image_size, l2_thresh, distance_thresh)

    def test_trajectory_generator(self):
        """Test initial pose recover"""
        camid = 1
        skip = 1
        start_index = 0
        img_glob = "*.jpg"
        distort_calibration = Cal3_S2(
            fx=347.820593, fy=329.096945, s=0, u0=295.717950, v0=222.964889).matrix()
        distortion = np.array(
            [-0.284322, 0.055723, 0.006772, 0.005264, 0.000000])
        trajectory = self.trajectory_estimator.trajectory_generator(
            distort_calibration, distortion, self.directory_name, camid, skip, img_glob, start_index)
        for pose in trajectory:
            print(pose)
        actual_poses = load_poses_from_file(self.directory_name+"poses.dat")
        plot_trajectory_verification(self.trajectory_estimator.map.landmarks, actual_poses, trajectory)

        self.gtsamAssertEquals(trajectory[-1], self.initial_pose, tol=0.1)


if __name__ == "__main__":
    unittest.main()
