"""
Unit tests for observed landmarks 
"""
# pylint: disable=

import unittest
from localization.observed_landmarks import ObservedLandmarks
import numpy as np


observed_landmarks = ObservedLandmarks()


class TestObservedLandmarks(unittest.TestCase):
    """Unit tests for features."""

    def test_append(self):
        for i in range(100):
            descriptor = np.zeros((1, 256))
            keypoint = np.array([[1, 1]])
            landmark = np.array([[1, 1, 1]])
            observed_landmarks.append(landmark, descriptor, keypoint)
        expected_keypoints = np.ones((100, 2))
        expected_landmarks = np.ones((100, 3))
        np.testing.assert_equal(
            observed_landmarks.descriptors, np.zeros((100, 256)))
        np.testing.assert_equal(
            observed_landmarks.keypoints, expected_keypoints)
        np.testing.assert_equal(
            observed_landmarks.landmarks, expected_landmarks)

        for i in range(100):
            descriptor = np.zeros((1, 256))
            keypoint = np.array([1, 1])
            landmark = np.array([1, 1, 1])
            observed_landmarks.append(landmark, descriptor, keypoint)
        expected_keypoints = np.ones((200, 2))
        expected_landmarks = np.ones((200, 3))
        np.testing.assert_equal(
            observed_landmarks.descriptors, np.zeros((200, 256)))
        np.testing.assert_equal(
            observed_landmarks.keypoints, expected_keypoints)
        np.testing.assert_equal(
            observed_landmarks.landmarks, expected_landmarks)


    def test_keypoint(self):
        keypoint = observed_landmarks.keypoint(10)
        np.testing.assert_equal(keypoint.shape, np.array([2, ]))

    def test_descriptor(self):
        descriptor = observed_landmarks.descriptor(10)
        np.testing.assert_equal(descriptor.shape, np.array([256, ]))

    def test_landmark(self):
        landmark = observed_landmarks.landmark(10)
        np.testing.assert_equal(landmark.shape, np.array([3, ]))

    def test_get_length(self):
        length = observed_landmarks.get_length()
        np.testing.assert_equal(length, 200)

        observed_landmarks2 = ObservedLandmarks()
        length = observed_landmarks2.get_length()
        np.testing.assert_equal(length, 0)


if __name__ == "__main__":
    unittest.main()
