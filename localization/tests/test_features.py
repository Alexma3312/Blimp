"""
Unit tests for features 
"""
# pylint: disable=

import unittest
from localization.features import Features
import numpy as np

keypoints = np.ones((300, 2))
descriptors = np.ones((300, 256))
features = Features(keypoints, descriptors)


class TestFeatures(unittest.TestCase):
    """Unit tests for features."""

    def test_keypoint(self):
        keypoint = features.keypoint(10)
        np.testing.assert_equal(keypoint.shape, np.array([2, ]))

    def test_descriptor(self):
        descriptor = features.descriptor(10)
        np.testing.assert_equal(descriptor.shape, np.array([256, ]))

    def test_get_length(self):
        length = features.get_length()
        np.testing.assert_equal(length, 300)

        keypoints = np.ones((0, 0))
        descriptors = np.ones((0, 0))
        features_2 = Features(keypoints, descriptors)
        length = features_2.get_length()
        np.testing.assert_equal(length, 0)

        keypoints = np.array([])
        descriptors = np.array([])
        features_2 = Features(keypoints, descriptors)
        length = features_2.get_length()
        np.testing.assert_equal(length, 0)


if __name__ == "__main__":
    unittest.main()
