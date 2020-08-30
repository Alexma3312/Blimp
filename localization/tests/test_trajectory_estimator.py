# cSpell: disable=invalid-name
"""
Unit tests for trajectory estimator.
"""
# pylint: disable=no-name-in-module

import math
import unittest

import cv2
import gtsam
import numpy as np
from gtsam import Cal3_S2, PinholeCameraCal3_S2, Point2, Point3, Pose3, Rot3
from gtsam.utils.test_case import GtsamTestCase
from mock import patch
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier

from localization.camera import Camera
from localization.features import Features
from localization.landmark_map import LandmarkMap
from localization.observed_landmarks import ObservedLandmarks
from localization.trajectory_estimator import TrajectoryEstimator
from utilities.back_projection import back_projection


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

    @unittest.skip("test_landmark_projection")
    def test_landmark_projection(self):
        # Create a map with three points - a point within view, a point out of view and a point back of view

        # create pose input
        pose = Pose3(Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0), Point3(0, 0, 0))
        # Create member variables
        # Three points, one good point, one bad cheirality point, and four out of field point
        landmark_points = np.array(
            [[0, 10, 0], [0, -10, 0], [320*15, 3000, 10]])
        descriptors = np.ones((3, 256))
        trajectory_estimator.map = LandmarkMap(landmark_points, descriptors)
        fx = 200
        fy = 200
        u0 = 320
        v0 = 240
        #  width,height
        image_size = (640, 480)
        trajectory_estimator._camera = Camera(fx, fy, u0, v0, image_size)

        # create estimate observations
        expected_landmarks = np.array([[0, 10, 0]])
        expected_keypoints = np.array([[320, 240]])

        actual_observations = trajectory_estimator.landmark_projection(pose)
        self.assertIsInstance(actual_observations, ObservedLandmarks)
        np.testing.assert_equal(
            actual_observations.landmarks, expected_landmarks)
        np.testing.assert_equal(
            actual_observations.keypoints, expected_keypoints)

    # @unittest.skip("test_landmark_projection")
    def test_landmark_projection_matrix(self):
        # create pose input
        pose = Pose3(Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0), Point3(0, 0, 0))
        # Create member variables
        # Three points, one good point, one bad cheirality point, and four out of field point
        landmark_points = np.array(
            [[0, 10, 0], [0, -10, 0], [320*15, 3000, 10]])
        descriptors = np.ones((3, 256))
        trajectory_estimator.map = LandmarkMap(landmark_points, descriptors)
        fx = 200
        fy = 200
        u0 = 320
        v0 = 240
        #  width,height
        image_size = (640, 480)
        trajectory_estimator._camera = Camera(fx, fy, u0, v0, image_size)

        # create estimate observations
        expected_landmarks = np.array([[0, 10, 0]])
        expected_keypoints = np.array([[320, 240]])

        actual_observations = trajectory_estimator.landmark_projection_matrix(pose)
        self.assertIsInstance(actual_observations, ObservedLandmarks)
        np.testing.assert_equal(
            actual_observations.landmarks, expected_landmarks)
        np.testing.assert_equal(
            actual_observations.keypoints, expected_keypoints)

    def find_keypoints_within_boundingbox(self):
        pass

    @unittest.skip("test_find_smallest_l2_distance_keypoint")
    def test_find_smallest_l2_distance_keypoint(self):
        # Create features
        descriptors = np.zeros((6, 256))
        descriptors[1][0] = 1
        descriptors[2][1] = 1
        descriptors[3][0] = 0.4
        # Closest l2
        descriptors[4][0] = 0.9
        descriptors[5][0] = 0.9
        # descriptors = descriptors.reshape(6,256)
        keypoints = np.arange(12).reshape(6, 2)
        features = Features(keypoints, descriptors)
        # Other paramters
        trajectory_estimator.l2_threshold = 0.6
        landmark_desc = np.zeros((1, 256))
        landmark_desc[0][0] = 1
        landmark_desc = landmark_desc.reshape(256,)
        landmark = np.arange(3)
        feature_indices = np.array([2, 3, 4, 5]).reshape(4,)
        # expected result
        expected_point2 = Point2(8, 9)
        expected_point3 = Point3(0, 1, 2)

        observation = trajectory_estimator.find_smallest_l2_distance_keypoint(
            feature_indices, features, landmark, landmark_desc)
        self.gtsamAssertEquals(expected_point2, observation[0])
        self.gtsamAssertEquals(expected_point3, observation[1])

    @unittest.skip("test_landmark_association")
    def test_landmark_association(self):
        # Create observed landmarks
        obersved_keypoints = np.array([[10, 10], [100, 10], [500, 50]])
        obersved_descriptors = np.zeros((3, 256))
        obersved_descriptors[:3, 0] = 1
        obersved_landmarks = np.arange(9).reshape(3, 3)
        observed_landmarks = ObservedLandmarks()
        observed_landmarks.landmarks = obersved_landmarks
        observed_landmarks.descriptors = obersved_descriptors
        observed_landmarks.keypoints = obersved_keypoints
        # Create Superpoint Features
        superpoint_keypoints = np.array(
            [[12, 14], [30, 30], [100, 200], [400, 200], [100, 70]])
        superpoint_descriptors = np.zeros((5, 256))
        superpoint_descriptors[0][1] = 1
        superpoint_descriptors[1][0] = 0.9
        superpoint_descriptors[2:4, 0] = 1
        superpoint_descriptors[4][0] = 0.9
        superpoint_features = Features(
            superpoint_keypoints, superpoint_descriptors)

        trajectory_estimator.x_distance_thresh = 65
        trajectory_estimator.l2_threshold = 0.6
        # Create expected observations
        expected_observations = [(Point2(30, 30), Point3(
            0, 1, 2)), (Point2(100, 70), Point3(3, 4, 5))]

        actual_observations = trajectory_estimator.landmark_association(
            superpoint_features, observed_landmarks)
        self.gtsamAssertEquals(
            actual_observations[0][0], expected_observations[0][0])
        self.gtsamAssertEquals(
            actual_observations[0][1], expected_observations[0][1])
        self.gtsamAssertEquals(
            actual_observations[1][0], expected_observations[1][0])
        self.gtsamAssertEquals(
            actual_observations[1][1], expected_observations[1][1])

    def test_cv2_pnp_ransac(self):
        # Create 6 points associations
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
        pose = Pose3(wRc, Point3(0.5, 0.2, 1))
        fx = 200
        fy = 200
        u0 = 320
        v0 = 240
        #  width,height
        K = Cal3_S2(fx, fy, 0, u0, v0)
        simple_camera = gtsam.SimpleCamera(pose, K)
        point3d_0 = Point3(5, 5, 1)
        point2d_0 = simple_camera.project(point3d_0)
        point3d_1 = Point3(7, 8, 2)
        point2d_1 = simple_camera.project(point3d_1)
        point3d_2 = Point3(11, 2, 4)
        point2d_2 = simple_camera.project(point3d_2)
        point3d_3 = Point3(17, 15, 13)
        point2d_3 = simple_camera.project(point3d_3)
        point3d_4 = Point3(4, 3, 6)
        point2d_4 = simple_camera.project(point3d_4)
        point3d_5 = Point3(3, 7, 12)
        point2d_5 = simple_camera.project(point3d_5)
        observations = [(point2d_0, point3d_0),
                        (point2d_1, point3d_1),
                        (point2d_2, point3d_2),
                        (point2d_3, point3d_3),
                        (point2d_4, point3d_4),
                        (point2d_5, point3d_5)]
        image_points = np.empty((2,))
        object_points = np.empty((3,))
        for observation in observations:
            image_points = np.vstack(
                (image_points, [observation[0].x(), observation[0].y()]))
            object_points = np.vstack(
                (object_points, [observation[1].x(), observation[1].y(), observation[1].z()]))

        #https://stackoverflow.com/questions/35650105/what-are-python-constants-for-cv2-solvepnp-method-flag
        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            object_points, image_points, K.matrix(), None, None, None, False, cv2.SOLVEPNP_DLS)
        rotation, _ = cv2.Rodrigues(rvecs)

        # R.T
        np.testing.assert_almost_equal(rotation, wRc.matrix().T, decimal=4)
        np.testing.assert_almost_equal(rotation.T, wRc.matrix(), decimal=4)
        # -R.T*t
        np.testing.assert_almost_equal(
            tvecs, -np.dot(wRc.matrix().T, np.array([[0.5], [0.2], [1]])), decimal=4)
        np.testing.assert_almost_equal(
            np.dot(rotation.T,-tvecs), np.array([[0.5], [0.2], [1]]), decimal=4)

    # @unittest.skip("test_pnp_ransac")

    def test_pnp_ransac(self):

        # Create 12 points associations

        # Create 12 points with noise

        # Create 18 points with mismatches

        # # observations - a list, [(Point2(), Point3())]
        # object_points = np.array([])
        # image_points = np.array([])

        # # 12 set of points
        # point_2d_1 = [2, 1]
        # point_3d_1 = [10, 10, 10]

        # # 6 set of points
        # # 5 set of points

        # expected_pose = np.array([])

        # # initial_estimation
        # useExtrinsicGuess = None

        # # method - CV_EPNP
        # # flags = cv2.CV_EPNP

        # # Flag method for solving a PnP problem
        # rvec, tvec, actual_inlier = trajectory_estimator.pnp_ransac()

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

    def test_constant_speed(self):
        """test constant speed"""
        pose1 = Pose3()
        pose2 = Pose3(Rot3.Ry(math.radians(180)), Point3(1,2,3))
        trajectory = [pose1, pose2]
        counter = 0
        expected_pose = Pose3(Rot3(),Point3(0,4,0))
        estiamte_pose = trajectory_estimator.constant_speed(trajectory, counter)
        self.gtsamAssertEquals(estiamte_pose,expected_pose)
        counter = 2
        expected_pose = Pose3(Rot3(),Point3(-2,8,-6))
        estiamte_pose = trajectory_estimator.constant_speed(trajectory, counter)
        pass

    def test_check_current_pose(self):
        """test check current pose"""
        current_pose = Pose3()
        pre_pose = Pose3(Rot3.Ry(math.radians(30)), Point3(1,0,0))
        status = True 
        dift_threshold = [2,30]
        actual_status = trajectory_estimator.check_current_pose(current_pose, pre_pose, status, dift_threshold)
        expected_status = True
        self.assertEqual(actual_status,expected_status)

        current_pose = Pose3()
        pre_pose = Pose3(Rot3.Ry(math.radians(40)), Point3(1,0,0))
        status = True 
        dift_threshold = [2,30]
        actual_status = trajectory_estimator.check_current_pose(current_pose, pre_pose, status, dift_threshold)
        expected_status = False
        self.assertEqual(actual_status,expected_status)

        current_pose = Pose3()
        pre_pose = Pose3(Rot3.Ry(math.radians(30)), Point3(3,0,0))
        status = True 
        dift_threshold = [2,30]
        actual_status = trajectory_estimator.check_current_pose(current_pose, pre_pose, status, dift_threshold)
        expected_status = False
        self.assertEqual(actual_status,expected_status)

if __name__ == "__main__":
    unittest.main()
