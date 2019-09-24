# cSpell: disable=invalid-name
"""
Unit tests for plot feature matches.
"""

import unittest
from utilities.visualize_matched_features import visualize_matched_features


class TestVisualizeMatchedFeatures(unittest.TestCase):
    """Unit tests for ."""

    def test_visualize_matched_features(self):
        """Test visualize matched features."""
        basedir = "utilities/tests/match_data/"
        number_images = 20
        visualize_matched_features(basedir, number_images)


if __name__ == "__main__":
    unittest.main()
