"""
Unit tests for landmark map 
"""
# pylint: disable=

import unittest
from localization.landmark_map import LandmarkMap
import numpy as np

landmarks = np.ones((300, 3))
descriptors = np.ones((300, 256))
landmark_map = LandmarkMap(landmarks, descriptors)


class TestFeatures(unittest.TestCase):
    """Unit tests for features."""

    def test_keypoint(self):
        landmark = landmark_map.landmark(10)
        np.testing.assert_equal(landmark.shape, np.array([3, ]))

    def test_descriptor(self):
        descriptor = landmark_map.descriptor(10)
        np.testing.assert_equal(descriptor.shape, np.array([256, ]))

    def test_get_length(self):
        length = landmark_map.get_length()
        np.testing.assert_equal(length, 300)

        landmarks = np.ones((0, 0))
        descriptors = np.ones((0, 0))
        landmarks_2 = LandmarkMap(landmarks, descriptors)
        length = landmarks_2.get_length()
        np.testing.assert_equal(length, 0)

        landmarks = np.array([])
        descriptors = np.array([])
        landmarks_2 = LandmarkMap(landmarks, descriptors)
        length = landmarks_2.get_length()
        np.testing.assert_equal(length, 0)


if __name__ == "__main__":
    unittest.main()
