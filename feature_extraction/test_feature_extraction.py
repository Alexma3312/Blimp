# cSpell: disable=invalid-name
"""Unit Test for Front End."""
import unittest

from feature_extraction import FeatureExtraction


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction"""
    def setUp(self):
        image_directory_path = 'feature_extraction/undistort_images/'
        image_extension = '*.jpg'
        image_size = (640, 480)
        nn_thresh = 0.7
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


if __name__ == "__main__":
    unittest.main()
