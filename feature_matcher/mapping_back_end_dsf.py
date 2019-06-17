"""Mapping back end based on DSF (disjoint set forests)"""
# cSpell: disable=
# pylint: disable=no-member, no-name-in-module

import copy
import os
from collections import defaultdict

import numpy as np

import gtsam
from feature_matcher.parser import get_matches, load_features
from gtsam import (  # pylint: disable=wrong-import-order,ungrouped-imports
    Point2, Point3, Pose3, symbol)


def X(i):  # pylint: disable=invalid-name
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):  # pylint: disable=invalid-name
    """Create key for landmark j."""
    return symbol(ord('p'), j)


class MappingBackEnd():
    """
    Mapping Back End.
        data_directory - input files(feature and feature match data) directory path
        num_images - number of input images
        pose_prior - pose prior data, used in initial estimation
        calibration - camera calibration, gtsam.Cal3_S2
        backprojection_depth - the estimated depth used in back projection
    """

    def __init__(self, data_directory, num_images, calibration, pose_estimates, measurement_noise, pose_prior_noise, min_obersvation_number=4, backprojection_depth=20):
        """Construct by reading from a data directory."""
        self._basedir = data_directory
        self._nrimages = num_images
        self._calibration = calibration
        self._min_landmark_seen = 3
        self._seen = min_obersvation_number
        self._depth = backprojection_depth
        self._pose_estimates = pose_estimates
        self._measurement_noise = measurement_noise
        self._pose_prior_noise = pose_prior_noise
        self._image_features = [self.load_features(
            image_index) for image_index in range(self._nrimages)]
        landmark_map, dsf = self.create_landmark_map()
        self._landmark_map = self.filter_bad_landmarks(landmark_map, dsf)

    def load_features(self, image_index):
        """ Load features from .key files
            features - keypoints:A N length list of gtsam.Point2(x,y). Descriptors: A Nx256 list of descriptors.
        """
        feat_file = os.path.join(
            self._basedir, "{0:07}.key".format(image_index))
        keypoints, _ = load_features(feat_file)
        return keypoints

    def load_matches(self, frame_1, frame_2):
        """ Load matches from .dat files
            matches - a list of [image 1 index, image 1 keypoint index, image 2 index, image 2 keypoint index]
        """
        matches_file = os.path.join(
            self._basedir, "match_{0}_{1}.dat".format(frame_1, frame_2))
        _, matches = get_matches(matches_file)
        return matches

    def generate_dsf(self):
        """Use dsf to find data association between landmark and landmark observation(features)"""
        dsf = gtsam.DSFMapIndexPair()

        for i in range(0, self._nrimages-1):
            for j in range(i+1, self._nrimages):
                matches = self.load_matches(i, j)
                for frame_1, keypt_1, frame_2, keypt_2 in matches:
                    dsf.merge(gtsam.IndexPair(frame_1, keypt_1),
                              gtsam.IndexPair(frame_2, keypt_2))

        return dsf

    def find_bad_matches(self):
        """Find bad matches:
           1. landmarks with more than one observations in an image.
           2. Features with more than one landmark correspondences"""
        bad_matches = set()
        for i in range(0, self._nrimages-1):
            for j in range(i+1, self._nrimages):
                matches = self.load_matches(i, j)
                matches_array = np.array(matches)
                keypt_1 = list(matches_array[:, 1])
                keypt_2 = list(matches_array[:, 3])
                for key in keypt_1:
                    # landmarks with more than one observations in an image
                    if keypt_1.count(key) > 1:
                        bad_matches.add((i, key))
                for key in keypt_2:
                    # Features with more than one landmark correspondences
                    if keypt_2.count(key) > 1:
                        bad_matches.add((j, key))
        return bad_matches

    def filter_bad_landmarks(self, landmark_map, dsf):
        """Filter bad landmarks:
            1. landmark observations<3
            2. landmarks with more than one observations in an image.
            3. Features with more than one landmark correspondences"""

        # filter bad matches
        bad_matches = self.find_bad_matches()
        bad_key_list = set()
        for bad_match in bad_matches:
            landmark_representative = dsf.find(
                gtsam.IndexPair(bad_match[0], bad_match[1]))
            key = (landmark_representative.i(),
                   landmark_representative.j())
            bad_key_list.add(key)
        for key in bad_key_list:
            del landmark_map[key]
        landmark_map_values = [landmark_map[key]
                               for key in sorted(landmark_map.keys())]
        landmark_map_new = copy.copy(landmark_map_values)

        for observation_list in landmark_map_values:
            if len(observation_list) < self._seen:
                landmark_map_new.remove(observation_list)

        return landmark_map_new

    def create_landmark_map(self):
        """Create a list to map landmarks and their correspondences.
            [Landmark_i:[(i,Point2()), (j,Point2())...]...]"""
        dsf = self.generate_dsf()
        landmark_map = defaultdict(list)
        for img_index, feature_list in enumerate(self._image_features):
            for feature_index, feature in enumerate(feature_list):
                landmark_representative = dsf.find(
                    gtsam.IndexPair(img_index, feature_index))
                key = (landmark_representative.i(),
                       landmark_representative.j())
                landmark_map[key].append((img_index, feature))
        return landmark_map, dsf

    def back_projection(self, key_point=Point2(), pose=Pose3(), depth=20):
        """
        Back Projection Function.
        Input:
            key_point-gtsam.Point2, key point location within the image.
            pose-gtsam.Pose3, camera pose in world coordinate.
        Output:
            gtsam.Pose3, landmark pose in world coordinate.
        """
        # Normalize input key_point
        pn = self._calibration.calibrate(key_point)
        # Transfer normalized key_point into homogeneous coordinate and scale with depth
        ph = Point3(depth*pn.x(), depth*pn.y(), depth)
        # Transfer the point into the world coordinate
        return pose.transform_from(ph)

    # def create_index_sets(self):
    #     """Create two sets with valid pose and point indices."""
    #     return pose_indices, point_indices

    def create_initial_estimate(self):
        """Create initial estimate with landmark map.
            Parameters:
                pose_estimates - list, pose estimates by measurements
                landmark_map - list, A map of landmarks and their correspondence 
        """
        initial_estimate = gtsam.Values()
        # Initial estimate for landmarks
        for landmark_idx, observation_list in enumerate(self._landmark_map):
            key_point = observation_list[0][1]
            pose_idx = observation_list[0][0]
            pose = self._pose_estimates[pose_idx]
            landmark_3d_point = self.back_projection(
                key_point, pose, self._depth)
            initial_estimate.insert(P(landmark_idx), landmark_3d_point)
        # Initial estimate for poses
        for pose_idx, pose_i in enumerate(self._pose_estimates):
            initial_estimate.insert(X(pose_idx), pose_i)
        return initial_estimate

    def bundle_adjustment(self):
        """
        Parameters:
            calibration - gtsam.Cal3_S2, camera calibration
            landmark_map - list, A map of landmarks and their correspondence 
        """
        # Initialize factor Graph
        graph = gtsam.NonlinearFactorGraph()

        initial_estimate = self.create_initial_estimate()

        # Create Projection Factors
        # """
        #   Measurement noise for bundle adjustment:
        #   sigma = 1.0
        #   measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, sigma)
        # """
        for landmark_idx, observation_list in enumerate(self._landmark_map):
            for obersvation in observation_list:
                pose_idx = obersvation[0]
                key_point = obersvation[1]
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    key_point, self._measurement_noise,
                    X(pose_idx), P(landmark_idx), self._calibration))

        # Create priors for first two poses
        # """
        #   Pose prior noise:
        #   rotation_sigma = np.radians(60)
        #   translation_sigma = 1
        #   pose_noise_sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
        #                             translation_sigma, translation_sigma, translation_sigma])
        # """
        for idx in (0, 1):
            pose_i = initial_estimate.atPose3(X(idx))
            graph.add(gtsam.PriorFactorPose3(
                X(idx), pose_i, self._pose_prior_noise))

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        sfm_result = optimizer.optimize()
        return sfm_result
