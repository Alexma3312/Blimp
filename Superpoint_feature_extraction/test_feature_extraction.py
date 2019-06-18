# cSpell: disable=invalid-name
"""Unit Test for Front End."""
# pylint: disable = wrong-import-order,no-name-in-module
import unittest

from Superpoint_feature_extraction.feature_extraction import FeatureExtraction
from gtsam import Cal3_S2


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction"""

    def setUp(self):
        image_directory_path = 'Superpoint_feature_extraction/undistort_images/'
        image_extension = '*.jpg'
        image_size = (640, 480)
        nn_thresh = 0.7
        # Input images(undistorted) calibration
        self.calibration = Cal3_S2(fx=232.0542, fy=252.8620,
                                   s=0, u0=325.3452, v0=240.2912).matrix()
        self.front_end = FeatureExtraction(
            image_directory_path, image_extension, image_size, nn_thresh)

    def test_leading_zero(self):
        """test leading zero"""
        actual = self.front_end.leading_zero(145)
        expected = '0000145'
        self.assertEqual(actual, expected)

    def test_get_image_paths(self):
        """test get image paths"""
        self.front_end.get_image_paths()
        self.assertEqual(len(self.front_end.img_paths), 6)

    def test_get_all_feature_matches(self):
        """test get all feature matches"""
        self.front_end.get_all_feature_matches(self.calibration, 0.5)

    # def test_ransac_filter(self):
    #     """test ransac filter"""
    #     pass


if __name__ == "__main__":
    unittest.main()
