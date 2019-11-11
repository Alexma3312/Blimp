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
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier
import cv2


def create_dummy_map(directory):
    """Create a dummy map to test trajectory estimator."""
    pass


# https://medium.com/@george.shuklin/mocking-complicated-init-in-python-6ef9850dd202
with patch.object(TrajectoryEstimator, "__init__", lambda x1, x2, x3, x4, x5, x6, x7, x8: None):
    trajectory_estimator = TrajectoryEstimator(
        None, None, None, None, None, None, None)


class TestTrajectoryEstimator(GtsamTestCase):
    """Unit tests for trajectory estimator."""

    def setUp(self):
        """Create noise model for bundle adjustment."""
        pass

    def test_init(self):
        """Mocking complicated __init__ ."""
        trajectory_estimator.l2_thresh = 0.7
        assert trajectory_estimator.l2_thresh is 0.7

    @unittest.skip("test_superpoint_generator")
    def test_superpoint_generator(self):
        image = np.zeros((480, 640)).astype(np.float32)
        image[20:50, 90:400] = 1
        features = trajectory_estimator.superpoint_generator(image)
        np.testing.assert_equal(features.descriptors.shape, np.array([4, 256]))
        np.testing.assert_equal(features.keypoints.shape, np.array([4, 2]))
        np.testing.assert_equal(features.get_length(), 4)
        np.testing.assert_equal(
            features.descriptor(1).shape, np.array([256, ]))

    def test_landmark_projection(self):
        # Create a map with three points - a point within view, a point out of view and a point back of view

        # create pose
        pose = Pose3(Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0), Point3(0, 0, 0))

        # create camera

        # create estimate observations

        pass

    def test_find_smallest_l2_distance_keypoint(self):

        # Vectorization

        # Looping
        pass

    def test_landmark_association(self):
        pass

    def test_pnp_ransac(self):
        # observations - a list, [(Point2(), Point3())]
        object_points = np.array([])
        image_points = np.array([])

        # 12 set of points
        point_2d_1 = [2, 1]
        point_3d_1 = [10, 10, 10]

        # 6 set of points
        # 5 set of points

        expected_pose = np.array([])

        # initial_estimation
        useExtrinsicGuess = None

        # method - CV_EPNP
        flags = cv2.CV_EPNP

        # Flag method for solving a PnP problem
        rvec, tvec, actual_inlier = trajectory_estimator.pnp_ransac()

        pass

    def test_BA_pose_estimation(self):
        pass

    def test_BA_pose_estimation_with_pnp_ransac_initial(self):
        pass

    @unittest.skip("test_sklearn_kd_tree_knn")
    def test_sklearn_kd_tree_knn(self):
        # projected_points = np.array([[1,2],[1,1],[1,0.5],[1,0.8],[10,10],[20,20],[20,21],[20,19]])
        projected_points = np.array([[1, 2], [1, 1], [1, 0.5], [1, 0.8], [
                                    10, 10], [20, 20], [20, 21], [20, 19]])
        extracted_points = np.array([[1, 1], [20, 20]])
        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(projected_points)
        NearestNeighbors(algorithm='auto', leaf_size=30)
        test = np.array([[1, 1]])
        _, indices = neigh.kneighbors(test)
        indices = indices[0].tolist()
        np.testing.assert_equal(indices, [1, 3, 2])
        # print(neigh.kneighbors([[1,1]]))
        # print(projected_points[indices])

    @unittest.skip("test_sklearn_knn_radius_classifier")
    def test_sklearn_knn_radius_classifier(self):
        projected_points = np.array([[1, 2], [1, 1], [1, 0.5], [1, 0.8], [
                                    10, 10], [20, 20], [20, 21], [20, 19]])
        extracted_points = np.array([[1, 1], [20, 20]])
        neigh = NearestNeighbors(n_neighbors=3)
        NearestNeighbors(algorithm='auto', leaf_size=30)
        neigh.fit(projected_points)

        test = np.array([[1, 1]])
        _, indices = neigh.radius_neighbors(test, radius=0.5)

        indices = indices[0].tolist()
        np.testing.assert_equal(indices, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
