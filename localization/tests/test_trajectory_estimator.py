# cSpell: disable=invalid-name
"""
Unit tests for trajectory estimator.
"""
# pylint: disable=no-name-in-module

import unittest
import gtsam
from gtsam import Cal3_S2, PinholeCameraCal3_S2, Point2, Point3, Pose3, Rot3
from gtsam.utils.test_case import GtsamTestCase
from localization.camera import Camera
from localization.features import Features
from localization.observed_landmarks import ObservedLandmarks
from localization.trajectory_estimator import TrajectoryEstimator
from utilities.back_projection import back_projection
from mock import patch


def create_dummy_map(directory):
    """Create a dummy map to test trajectory estimator."""
    pass


class TestTrajectoryEstimator(GtsamTestCase):
    """Unit tests for trajectory estimator."""

    def setUp(self):
        """Create noise model for bundle adjustment."""
        pass

    # def test_trajectory_estimator_initialization(self):
    #     """"""
    #     measurement_noise_sigma = 1.0
    #     measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
    #         2, measurement_noise_sigma)
    #     point_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.01)

    #     # Create calibration matrix
    #     fx = 1406.5/3
    #     fy = 1317/3
    #     u0 = 775.2312/3
    #     v0 = 953.3863/3
    #     image_size = (640, 480)
    #     camera = Camera(fx, fy, u0, v0, image_size)

    #     # Camera to world rotation
    #     wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # pylint: disable=invalid-name
    #     initial_pose = Pose3(wRc, Point3(0, 0, 1.5))
    #     directory_name = "localization/datasets/Klaus_14x4_phone/"

    #     l2_thresh = 0.7
    #     distance_thresh = [5, 5]
    #     trajectory_estimator = TrajectoryEstimator(
    #         initial_pose, directory_name, camera, l2_thresh, distance_thresh, measurement_noise, point_prior_noise)
    #     assert trajectory_estimator.l2_thresh = 0.7

    def test_init(self):
        """Mocking complicated __init__ ."""
        # https://medium.com/@george.shuklin/mocking-complicated-init-in-python-6ef9850dd202
        with patch.object(TrajectoryEstimator, "__init__", lambda x1, x2, x3, x4, x5, x6, x7, x8: None):
            trajectory_estimator = TrajectoryEstimator(
                None, None, None, None, None, None, None)
            trajectory_estimator.l2_thresh = 0.7
            assert trajectory_estimator.l2_thresh is 0.7
    
    def test_superpoint_generator(self):
        pass

    def test_landmark_projection(self):
        pass

    def test_find_find_keypoints_within_boundingbox(self):
        pass

    def test_find_smallest_l2_distance_keypoint(self):
        pass

    def test_landmark_association(self):
        pass

    def test_DLT_ransac(self):
        pass


if __name__ == "__main__":
    unittest.main()
