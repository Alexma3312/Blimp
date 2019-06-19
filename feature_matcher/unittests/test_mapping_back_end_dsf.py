# cSpell: disable=invalid-name
"""
Unit tests for mapping back end dsf.
"""
# pylint: disable=no-name-in-module, wrong-import-order, no-member, line-too-long

import unittest

import numpy as np

import gtsam
from feature_matcher.mapping_back_end_dsf import MappingBackEnd, P, X
from gtsam import Cal3_S2, Point2, Point3, Pose3, Rot3 # pylint: disable=ungrouped-imports
from gtsam.utils.test_case import GtsamTestCase


def load_points():
    """load landmark data"""
    pts3d = []
    with open("feature_matcher/sim_match_data/points.txt") as f:
        pts = f.readlines()
        num_pts = int(pts[0].strip())
        for i in range(num_pts):
            pt = [float(x) for x in pts[i+1].strip().split()]
            pts3d.append(Point3(*pt))
    return pts3d


class TestMappingBackEnd(GtsamTestCase):
    """Unit tests for mapping back end."""

    def setUp(self):
        """Create mapping Back-end and read csv file."""
        # Input images(undistorted) calibration
        fov, w, h = 60, 1280, 720
        calibration = Cal3_S2(fov, w, h)
        # Create pose priors
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # camera to world rotation
        self.pose_estimates = [Pose3(wRc, Point3(0, i, 2)) for i in range(3)]
        # Create measurement noise for bundle adjustment
        sigma = 1.0
        measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, sigma)
        # Create pose prior noise
        rotation_sigma = np.radians(60)
        translation_sigma = 1
        pose_noise_sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
                                      translation_sigma, translation_sigma, translation_sigma])
        pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(pose_noise_sigmas)
        # Create MappingBackEnd instance
        data_directory = 'feature_matcher/sim_match_data/'
        min_obersvation_number = 2
        filter_bad_landmarks_enable = True
        self.num_images = 3
        self.back_end = MappingBackEnd(
            data_directory, self.num_images, calibration, self.pose_estimates, measurement_noise, pose_prior_noise, filter_bad_landmarks_enable, min_obersvation_number)  # pylint: disable=line-too-long

    def assert_landmark_map_equal(self, actual_landmark_map, expected_landmark_map):
        """Helper function to test landmark map."""
        for item in expected_landmark_map.items():
            assert (
                item[0] in actual_landmark_map), "FAIL: Test create landmark map."
            for i, observation in enumerate(expected_landmark_map[item[0]]):
                self.assertEqual(
                    actual_landmark_map[item[0]][i][0], observation[0])
                self.gtsamAssertEquals(
                    actual_landmark_map[item[0]][i][1], observation[1])

    def test_load_points(self):
        """Test load points."""
        pts3d = load_points()
        self.assertEqual(len(pts3d), 90)

    def test_load_features(self):
        """Test load image features"""
        image_features, _ = self.back_end.load_features(0)
        self.assertIsInstance(image_features, list)

    def test_load_matches(self):
        """Test get matches"""
        matches = self.back_end.load_matches(0, 1)
        self.assertEqual(matches[0][0], 0)
        self.assertEqual(matches[0][2], 1)
        self.assertIsInstance(matches, list)

    def test_find_dsf(self):
        """Test functions related to DSF"""
        # """Test find bad matches."""
        data_directory = 'feature_matcher/dsf_test_data/'
        num_images = 3
        filter_bad_landmarks_enable = False
        min_obersvation_number = 3
        back_end = MappingBackEnd(
            data_directory, num_images, gtsam.Cal3_S2(), [], [], [], filter_bad_landmarks_enable, min_obersvation_number)
        bad_matches = back_end.find_bad_matches()
        self.assertEqual(bad_matches, {(2, 6), (0, 4)})

        # """Test create landmark map."""
        actual_landmark_map, dsf = back_end.create_landmark_map(False)
        expected_landmark_map = {(0, 1): [(0, Point2(1, 1)), (1, Point2(2, 1)), (2, Point2(3, 1))], (2, 0): [(2, Point2(0, 0))], (0, 0): [(0, Point2(0, 0))], (0, 5): [(0, Point2(1, 5)), (0, Point2(1, 6)), (2, Point2(3, 6))], (0, 4): [
            (0, Point2(1, 4)), (2, Point2(3, 3)), (2, Point2(3, 5))], (1, 0): [(1, Point2(0, 0))], (0, 3): [(0, Point2(1, 3)), (1, Point2(2, 2)), (2, Point2(3, 2))], (0, 2): [(0, Point2(1, 2)), (2, Point2(3, 4))]}
        self.assert_landmark_map_equal(
            actual_landmark_map, expected_landmark_map)

        # """Test generate dsf."""
        landmark_representative = dsf.find(gtsam.IndexPair(2, 1))
        key = (landmark_representative.i(), landmark_representative.j())
        self.assertEqual(key, (0, 1))

        # """Test filter bad landmarks."""
        actual_landmark_map = back_end.filter_bad_landmarks(
            expected_landmark_map, dsf, True)
        expected_landmark_map = [[(0, Point2(1, 1)), (1, Point2(2, 1)), (2, Point2(3, 1))], [
            (0, Point2(1, 3)), (1, Point2(2, 2)), (2, Point2(3, 2))]]
        # self.assert_landmark_map_equal(actual_landmark_map, expected_landmark_map)
        for i, observations in enumerate(expected_landmark_map):
            for j, observation in enumerate(observations):
                self.assertEqual(actual_landmark_map[i][j][0], observation[0])
                self.gtsamAssertEquals(
                    actual_landmark_map[i][j][1], observation[1])

    def test_create_initial_estimate(self):
        """test create initial estimate"""
        initial_estimate = self.back_end.create_initial_estimate()
        for i, pose in enumerate(self.pose_estimates):
            self.gtsamAssertEquals(initial_estimate.atPose3((X(i))), pose)

    def test_bundle_adjustment(self):
        """Test bundle adjustment"""
        result = self.back_end.bundle_adjustment()
        # """Test pose output"""
        delta_z = 1
        num_frames = 3
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
        for i in range(num_frames):
            # Create ground truth poses
            expected_pose_i = Pose3(wRc, Point3(0, delta_z*i, 2))
            # Get actual poses
            actual_pose_i = result.atPose3(X(i))
            self.gtsamAssertEquals(actual_pose_i, expected_pose_i, 1e-6)

        # """Test landmark output"""
        expected_points = load_points()
        for i, expected_point_i in enumerate(expected_points):
            actual_point_i = result.atPoint3(P(i))
            self.gtsamAssertEquals(actual_point_i, expected_point_i, 1e-4)


if __name__ == "__main__":
    unittest.main()
