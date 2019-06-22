# cSpell: disable=invalid-name
"""
Unit tests for mapping back end dsf.
"""
# pylint: disable=no-name-in-module, wrong-import-order, no-member, line-too-long

import unittest
from os import path

import numpy as np

import gtsam
from feature_matcher.mapping_back_end_dsf import MappingBackEnd, P, X
from feature_matcher.mapping_result_helper import (load_map_from_file,
                                                   load_poses_from_file)
from gtsam import (Cal3_S2, Point2,  # pylint: disable=ungrouped-imports
                   Point3, Pose3, Rot3)
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
        self.data_directory = 'feature_matcher/sim_match_data/'
        min_obersvation_number = 2
        filter_bad_landmarks_enable = True
        self.num_images = 3
        self.back_end = MappingBackEnd(
            self.data_directory, self.num_images, calibration, self.pose_estimates, measurement_noise, pose_prior_noise, filter_bad_landmarks_enable, min_obersvation_number)  # pylint: disable=line-too-long
        self.result = self.back_end.bundle_adjustment()

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

        # """Test pose output"""
        delta_z = 1
        num_frames = 3
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
        for i in range(num_frames):
            # Create ground truth poses
            expected_pose_i = Pose3(wRc, Point3(0, delta_z*i, 2))
            # Get actual poses
            actual_pose_i = self.result.atPose3(X(i))
            self.gtsamAssertEquals(actual_pose_i, expected_pose_i, 1e-6)

        # """Test landmark output"""
        expected_points = load_points()
        for i, expected_point_i in enumerate(expected_points):
            actual_point_i = self.result.atPoint3(P(i))
            self.gtsamAssertEquals(actual_point_i, expected_point_i, 1e-4)

    def test_get_landmark_descriptor(self):
        """Test get landmark descriptor"""
        landmark_map = self.back_end.get_landmark_map()
        actual_desc = self.back_end.get_landmark_descriptor(landmark_map[0])
        expected_desc = [-0.0228, 0.0062, -0.0301, 0.036, -0.0042, 0.0606, -0.0554, 0.0104, 0.0241, -0.035, -0.0211, -0.0809, -0.0347, -0.0648, -0.057, 0.0412, 0.0136, -0.054, -0.1114, 0.0901, -0.0057, -0.0057, 0.1536, 0.0046, 0.0693, 0.014, 0.0236, 0.0228, -0.0224, 0.1264, -0.0837, -0.0236, -0.0631, -0.0292, -0.079, 0.033, 0.0204, -0.0499, -0.021, 0.0188, 0.0961, -0.0149, 0.1638, 0.0308, 0.0588, -0.0109, 0.0628, -0.0412, -0.0603, 0.1131, -0.0276, -0.1137, -0.0436, 0.0407, 0.074, -0.0027, 0.1288, 0.0492, 0.015, -0.038, -0.0786, 0.071, -0.0096, -0.0719, 0.0424, -0.0402, -0.03, 0.0149, -0.02, 0.0198, -0.0398, 0.0264, -0.0646, 0.0226, -0.1117, 0.0384, -0.0536, 0.0803, 0.0263, -0.0287, 0.0501, -0.084, 0.0477, 0.0513, -0.0853, 0.069, 0.0149, 0.0205, -0.0866, -0.0932, 0.0007, 0.0516, -0.0772, 0.0433, -0.0196, 0.0588, -0.1192, 0.0434, -0.1432, -0.0494, -0.0591, -0.0849, -0.0109, 0.0259, -0.0516, 0.0327, 0.0501, 0.0841, 0.1331, -0.0449, -0.1234, -0.1025, -0.0441, -0.0664, -0.0375, 0.0967, 0.0053, -0.0094, -0.0837, -0.0522, -0.0298, 0.0471, -0.0443, -0.0133, -0.0289, 0.1113, -0.0001,
                         0.017, -0.0362, -0.0649, 0.0312, 0.0098, -0.056, -0.0203, -0.1457, 0.0829, -0.089, -0.0392, -0.0005, 0.0221, 0.1029, 0.0557, -0.0487, -0.0617, 0.0521, 0.0139, -0.0726, 0.0166, 0.0683, -0.074, -0.1101, 0.0131, 0.0443, 0.0267, 0.092, -0.0104, -0.0051, 0.0572, 0.1161, 0.0068, -0.0206, 0.0114, -0.0884, -0.0064, 0.0044, 0.017, 0.0119, 0.0685, -0.0887, 0.0374, 0.0613, 0.0047, 0.0824, 0.0412, -0.0018, -0.0425, 0.0399, -0.0303, -0.0843, 0.0117, 0.0592, 0.0628, 0.0256, 0.0666, -0.021, -0.0388, 0.1168, -0.0384, -0.087, -0.0219, 0.0143, -0.018, 0.0789, 0.0609, -0.0085, -0.1109, 0.0361, 0.0202, -0.0423, 0.0313, -0.0234, -0.1378, -0.0772, 0.129, -0.0511, 0.0378, 0.1046, 0.0658, -0.1097, -0.0898, -0.1398, -0.0301, -0.0137, -0.0905, -0.03, -0.0905, -0.0365, -0.0711, 0.0941, 0.0047, 0.0003, 0.022, 0.0143, -0.0393, -0.0196, -0.024, 0.0045, -0.0532, -0.0117, 0.0089, -0.0144, -0.0027, -0.0225, 0.0857, 0.0125, 0.1007, 0.0446, 0.0132, -0.055, -0.154, -0.0789, 0.0045, -0.0907, 0.0156, 0.0707, 0.0667, -0.1578, 0.0206, -0.0072, 0.0477, -0.0862, 0.0817, -0.0221, -0.0816, 0.0652, 0.0355]
        np.testing.assert_almost_equal(actual_desc, expected_desc, 5)

    def test_save_map_to_file(self):
        """Test save map to file."""
        self.back_end.save_map_to_file(self.result)
        file_exist = path.exists(self.data_directory+"result/map.dat")
        self.assertEqual(file_exist, True)

    def test_load_map_from_file(self):
        """Test load map from file."""
        file_name = self.data_directory+"result/map.dat"
        landmark_points, descriptors = load_map_from_file(file_name)
        actual_landmark = landmark_points[0]
        actual_desc = descriptors[0]
        expected_landmark = [-4.999999944544284425e+00,
                             4.999999945095506604e+01, 1.000000011114795528e+00]
        expected_desc = [-0.0228, 0.0062, -0.0301, 0.036, -0.0042, 0.0606, -0.0554, 0.0104, 0.0241, -0.035, -0.0211, -0.0809, -0.0347, -0.0648, -0.057, 0.0412, 0.0136, -0.054, -0.1114, 0.0901, -0.0057, -0.0057, 0.1536, 0.0046, 0.0693, 0.014, 0.0236, 0.0228, -0.0224, 0.1264, -0.0837, -0.0236, -0.0631, -0.0292, -0.079, 0.033, 0.0204, -0.0499, -0.021, 0.0188, 0.0961, -0.0149, 0.1638, 0.0308, 0.0588, -0.0109, 0.0628, -0.0412, -0.0603, 0.1131, -0.0276, -0.1137, -0.0436, 0.0407, 0.074, -0.0027, 0.1288, 0.0492, 0.015, -0.038, -0.0786, 0.071, -0.0096, -0.0719, 0.0424, -0.0402, -0.03, 0.0149, -0.02, 0.0198, -0.0398, 0.0264, -0.0646, 0.0226, -0.1117, 0.0384, -0.0536, 0.0803, 0.0263, -0.0287, 0.0501, -0.084, 0.0477, 0.0513, -0.0853, 0.069, 0.0149, 0.0205, -0.0866, -0.0932, 0.0007, 0.0516, -0.0772, 0.0433, -0.0196, 0.0588, -0.1192, 0.0434, -0.1432, -0.0494, -0.0591, -0.0849, -0.0109, 0.0259, -0.0516, 0.0327, 0.0501, 0.0841, 0.1331, -0.0449, -0.1234, -0.1025, -0.0441, -0.0664, -0.0375, 0.0967, 0.0053, -0.0094, -0.0837, -0.0522, -0.0298, 0.0471, -0.0443, -0.0133, -0.0289, 0.1113, -0.0001,
                         0.017, -0.0362, -0.0649, 0.0312, 0.0098, -0.056, -0.0203, -0.1457, 0.0829, -0.089, -0.0392, -0.0005, 0.0221, 0.1029, 0.0557, -0.0487, -0.0617, 0.0521, 0.0139, -0.0726, 0.0166, 0.0683, -0.074, -0.1101, 0.0131, 0.0443, 0.0267, 0.092, -0.0104, -0.0051, 0.0572, 0.1161, 0.0068, -0.0206, 0.0114, -0.0884, -0.0064, 0.0044, 0.017, 0.0119, 0.0685, -0.0887, 0.0374, 0.0613, 0.0047, 0.0824, 0.0412, -0.0018, -0.0425, 0.0399, -0.0303, -0.0843, 0.0117, 0.0592, 0.0628, 0.0256, 0.0666, -0.021, -0.0388, 0.1168, -0.0384, -0.087, -0.0219, 0.0143, -0.018, 0.0789, 0.0609, -0.0085, -0.1109, 0.0361, 0.0202, -0.0423, 0.0313, -0.0234, -0.1378, -0.0772, 0.129, -0.0511, 0.0378, 0.1046, 0.0658, -0.1097, -0.0898, -0.1398, -0.0301, -0.0137, -0.0905, -0.03, -0.0905, -0.0365, -0.0711, 0.0941, 0.0047, 0.0003, 0.022, 0.0143, -0.0393, -0.0196, -0.024, 0.0045, -0.0532, -0.0117, 0.0089, -0.0144, -0.0027, -0.0225, 0.0857, 0.0125, 0.1007, 0.0446, 0.0132, -0.055, -0.154, -0.0789, 0.0045, -0.0907, 0.0156, 0.0707, 0.0667, -0.1578, 0.0206, -0.0072, 0.0477, -0.0862, 0.0817, -0.0221, -0.0816, 0.0652, 0.0355]
        np.testing.assert_almost_equal(actual_landmark, expected_landmark, 5)
        np.testing.assert_almost_equal(actual_desc, expected_desc, 5)

    def test_save_poses_to_file(self):
        """Test save poses to file."""
        self.back_end.save_poses_to_file(self.result)
        file_exist = path.exists(self.data_directory+"result/poses.dat")
        self.assertEqual(file_exist, True)

    def test_load_poses_from_file(self):
        """Test load psoes from file."""
        file_name = self.data_directory+"result/poses.dat"
        poses = load_poses_from_file(file_name)
        # """Test pose output"""
        delta_z = 1
        num_frames = 3
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)
        for i in range(num_frames):
            # Create ground truth poses
            expected_pose_i = Pose3(wRc, Point3(0, delta_z*i, 2))
            # Get actual poses
            rotation = Rot3(np.array(poses[i][3:]).reshape(3, 3))
            actual_pose_i = Pose3(rotation, Point3(np.array(poses[i][0:3])))
            self.gtsamAssertEquals(actual_pose_i, expected_pose_i, 1e-6)


if __name__ == "__main__":
    unittest.main()
