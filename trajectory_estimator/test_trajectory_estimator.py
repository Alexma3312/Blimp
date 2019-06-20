"""Test trajectory estimator."""

import math
import sys
import unittest

import cv2
import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3, symbol

from atrium_control.trajectory_estimator import TrajectoryEstimator, read_image
from sfm import sfm_data
from tests.trajectory_estimator_data import Trajectory_Data, create_atrium_map

# Place the following line before import if the code is ran in a terminal (not VScode terminal).
sys.path.append('../')


def read_images(width, height):
    """
    Read a list of images from the dataset folder then resize and transfer each image to grey scale image 
    """

    # Remove `../` if the code is ran in VScode terminal. Add `../` in path if the code is ran in a terminal (not VScode terminal)
    image_list = []
    image_1 = read_image(
        'datasets/wall_corresponding_feature_data/raw_frame_left.jpg', [width, height])
    image_2 = read_image(
        'datasets/wall_corresponding_feature_data/raw_frame_middle.jpg', [width, height])
    image_3 = read_image(
        'datasets/wall_corresponding_feature_data/raw_frame_right.jpg', [width, height])
    image_list.append(image_1)
    image_list.append(image_2)
    image_list.append(image_3)
    return image_list


class TestTrajectoryEstimator(unittest.TestCase):
    """ Test Trajectory Estimator."""

    def setUp(self):

        # Import SFM map output as Trajectory Estimator map input
        self.atrium_map = create_atrium_map()

        # Create an input image list
        self.image_list = read_images(640, 480)

        # Use the poses in SFM as the actual estimate trajectory
        #self.past_trajectory_estimate = SFMdata.createPoses()
        self.trajectory_estimator = TrajectoryEstimator(
            self.atrium_map, 4, 4, 0.7, 5)
        self.trajectory_data = Trajectory_Data(128, 640, 480, 5)
        self.trajectory_data.create_poses()
        self.trajectory_data.create_past_poses(0.01)
        self.trajectory_data.create_project_features()
        self.trajectory_data.create_map_indices()
        self.trajectory_data.create_superpoint_features()
        self.trajectory_data.create_matched_features()
        self.trajectory_data.create_visible_map()

    def assertGtsamEquals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Poses are not equal:\n{}!={}".format(actual, expected))

    def assertGtsamListEqual(self, actual_list, expected_list):
        """Test if two list of gtsam object are the same."""
        assert len(actual_list) == len(
            expected_list), "The two lists have different lengths. "
        for i in range(len(actual_list)):
            self.assertGtsamEquals(actual_list[i], expected_list[i])

    def assertFeaturesEqual(self, actual_features, expected_features):
        """Test if two Feature Objects are the same."""
        assert actual_features.get_length() == expected_features.get_length(
        ), "The two feature objects have different number of features."
        # Check the descriptors
        self.assertEqual(actual_features.descriptor_list,
                         expected_features.descriptor_list)
        # Check the keypoints
        self.assertGtsamListEqual(
            actual_features.key_point_list, expected_features.key_point_list)

    def assertMapEqual(self, actual_map, expected_map):
        """Test if two Map Objects are the same."""
        assert actual_map.get_length() == expected_map.get_length(
        ), "The two Map objects have different number of landmarks."
        # Check the descriptors
        self.assertEqual(actual_map.descriptor_list,
                         expected_map.descriptor_list)
        # Check the trajectory
        self.assertGtsamListEqual(
            actual_map.trajectory, expected_map.trajectory)
        # Check the landmarks
        self.assertGtsamListEqual(
            actual_map.landmark_list, expected_map.landmark_list)

    def test_landmarks_projection(self):
        """Test landmarks projection."""
        for i, pose in enumerate(self.trajectory_data.past_poses):
            projected_features, map_indices = self.trajectory_estimator.landmarks_projection(
                pose)
            self.assertEqual(map_indices, self.trajectory_data.map_indices[i])
            self.assertFeaturesEqual(
                projected_features, self.trajectory_data.projected_features[i])

    def test_descriptor_match(self):
        """Test descriptor match."""
        # Calculate distances of unit normalized vectors
        desc1 = np.array([0, 0, 1])
        desc2 = np.array([0, 0, 1])
        dmat = np.dot(desc1, desc2.T)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        self.assertEqual(dmat, 0)

    def test_data_association(self):
        """Test data association."""
        for i, pose in enumerate(self.trajectory_data.past_poses):
            matched_features, visible_map = self.trajectory_estimator.data_association(
                self.trajectory_data.superpoint_features[i], self.trajectory_data.projected_features[i], self.trajectory_data.map_indices[i])

            self.assertFeaturesEqual(
                matched_features, self.trajectory_data.matched_features[i])
            self.assertMapEqual(
                visible_map, self.trajectory_data.visible_map[i])

    def test_trajectory_estimator(self):
        """Test trajectory estimator."""
        for i, pose in enumerate(self.trajectory_data.past_poses):
            self.trajectory_estimator.atrium_map.trajectory = [pose]
            actual_pose = self.trajectory_estimator.trajectory_estimator(
                self.trajectory_data.matched_features[i], self.trajectory_data.visible_map[i])
            self.assertGtsamEquals(
                actual_pose, self.trajectory_data.poses[i], 0.5)


if __name__ == "__main__":
    unittest.main()
