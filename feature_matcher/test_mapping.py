# cSpell: disable=invalid-name
"""
Unit tests for MappingBackEnd
"""
# pylint: disable=invalid-name, no-name-in-module, no-member

import unittest

import numpy as np

from feature_matcher.Shicong_01_mapping_4d_agri_feature_match import (
    ImageFeature, MappingBackEnd, P, X, transform_from)
from gtsam import Cal3_S2, Point2, Point3, Pose3, Rot3, Values
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


class TestMapping(GtsamTestCase):
    """Unit tests for mapping."""

    def setUp(self):
        """Create mapping Back-end and read csv file."""
        # Input images(undistorted) calibration
        fov, w, h = 60, 1280, 720
        calibration = Cal3_S2(fov, w, h)
        # Create pose priors
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # camera to world rotation
        pose_priors = [Pose3(wRc, Point3(0, i, 2)) for i in range(3)]
        # Create MappingBackEnd instance
        data_directory = 'feature_matcher/sim_match_data/'
        self.back_end = MappingBackEnd(
            data_directory, 3, pose_priors, calibration)

    def test_transform_from(self):
        """Test transform form"""
        T = Pose3(Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0), Point3(1, 2, 3))
        pose = Pose3(Rot3(), Point3(1, 2, 3))
        actual = transform_from(T, pose)
        expected = Pose3(Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0), Point3(2, 5, 1))
        self.gtsamAssertEquals(actual, expected)

    def test_load_image_features(self):
        """Test load image features"""
        image_feature = self.back_end.load_image_features(0)
        self.assertIsInstance(image_feature, ImageFeature)
        self.assertEqual(np.array(image_feature.keypoints).shape[1], 2)

    def test_get_matches_from_file(self):
        """Test get matches"""
        matches = self.back_end.get_matches_from_file(0, 1)
        self.assertEqual(matches[0][0], 0)
        self.assertEqual(matches[0][2], 1)
        self.assertIsInstance(matches, list)

    def test_load_image_matches(self):
        """Test load image matches"""
        image_matches = self.back_end.load_image_matches()
        self.assertEqual(len(image_matches), self.back_end.nrimages)

    def test_initial_estimate(self):
        """Test initial estimate"""
        pass

    def test_data_associate_single_image(self):
        """Test data associate single image"""
        pass

    def test_data_associate_all_images(self):
        """Test data associate all images"""
        pass

    def test_back_projection(self):
        """Test back projection"""
        actual = self.back_end.back_projection(Point2(640, 360), Pose3(
            Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0), Point3()), 20)
        expected = Point3(0, 20, 0)
        self.gtsamAssertEquals(actual, expected)

    def test_create_index_sets(self):
        """Test initial estimate."""
        pose_indices, point_indices = self.back_end.create_index_sets()
        self.assertIsInstance(pose_indices, set)
        self.assertIsInstance(point_indices, set)

    def test_create_initial_estimate(self):
        """Test initial estimate."""
        pose_indices, _ = self.back_end.create_index_sets()
        initial_estimate = self.back_end.create_initial_estimate(pose_indices)
        self.assertIsInstance(initial_estimate, Values)

    def test_bundle_adjustment(self):
        """
        Test Bundle Adjustment method.
        """
        sfm_result, _, _ = self.back_end.bundle_adjustment()
        # self.back_end.plot_sfm_result(sfm_result)

        # """Test pose output"""
        delta_z = 1
        num_frames = 3
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
        for i in range(num_frames):
            # Create ground truth poses
            expected_pose_i = Pose3(wRc, Point3(0, delta_z*i, 2))
            # Get actual poses
            actual_pose_i = sfm_result.atPose3(X(i))
            self.gtsamAssertEquals(actual_pose_i, expected_pose_i, 1e-6)

        # """Test landmark output"""
        expected_points = load_points()
        for i, expected_point_i in enumerate(expected_points):
            actual_point_i = sfm_result.atPoint3(P(i+1))
            # print(actual_point_i,expected_point_i)
            self.gtsamAssertEquals(actual_point_i, expected_point_i, 1e-4)


if __name__ == "__main__":
    unittest.main()
