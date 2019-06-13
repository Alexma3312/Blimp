# cSpell: disable=invalid-name
"""
Unit tests for Parser
"""
# pylint: disable=unused-import

import unittest

from feature_matcher.parser import (get_matches, load_features,
                                    load_features_list, parse_matches)


class TestMapping(unittest.TestCase):
    """Unit tests for parser."""
    # def test_parse_matches(self):
    #     """test parse matches"""
    #     pass

    def test_get_matches(self):
        """test get matches"""
        frames, matches = get_matches(
            'feature_matcher/sim_match_data/match_0_1.dat')
        self.assertEqual(frames, [0, 1])
        self.assertEqual(matches[0][0], 0)
        self.assertEqual(matches[0][2], 1)

    def test_load_features(self):
        """test load features"""
        keypoints, descriptors = load_features(
            'feature_matcher/sim_match_data/0000000.key')
        self.assertEqual(keypoints.shape[1], 2)
        self.assertEqual(descriptors.shape[0], keypoints.shape[0])

    def test_load_features_list(self):
        """test load features list"""
        keypoints, descriptors = load_features_list(
            'feature_matcher/sim_match_data/0000000.key')
        self.assertIsInstance(keypoints, list)
        self.assertIsInstance(descriptors, list)


if __name__ == "__main__":
    unittest.main()
