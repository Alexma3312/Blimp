"""Mapping back end based on DSF (disjoint set forests)"""
# cSpell: disable=
# pylint: disable=no-member, no-name-in-module, line-too-long

import copy
import os
from collections import defaultdict

import cv2
import numpy as np

import gtsam
from feature_matcher.mapping_result_helper import (save_map_to_file,
                                                   save_poses_to_file)
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

    def __init__(self, data_directory, num_images, calibration, pose_estimates, measurement_noise, pose_prior_noise, filter_bad_landmarks_enable=True, min_obersvation_number=4, prob=0.9, threshold=3, backprojection_depth=20):
        """Construct by reading from a data directory."""
        # Parameters for CV2 find Essential matrix
        self._cv_prob = prob
        self._cv_threshold = threshold
        # Mapping result tunning parameters
        self._seen = min_obersvation_number
        self._depth = backprojection_depth
        # Basic variables
        self._basedir = data_directory
        self._nrimages = num_images
        self._calibration = calibration
        # GTSAM graph optimization parameters
        self._pose_estimates = pose_estimates
        self._measurement_noise = measurement_noise
        self._pose_prior_noise = pose_prior_noise
        # Store all features and descriptors
        self._image_features = [self.load_features(
            image_index)[0] for image_index in range(self._nrimages)]
        self._image_descriptors = [self.load_features(
            image_index)[1] for image_index in range(self._nrimages)]
        landmark_map, dsf = self.create_landmark_map()
        self._landmark_map = self.filter_bad_landmarks(
            landmark_map, dsf, filter_bad_landmarks_enable)

    def load_features(self, image_index):
        """ Load features from .key files
        """
        feat_file = os.path.join(
            self._basedir, "{0:07}.key".format(image_index))
        keypoints, descriptors = load_features(feat_file)
        return keypoints, descriptors

    def load_matches(self, frame_1, frame_2):
        """ Load matches from .dat files
            matches - a list of [image 1 index, image 1 keypoint index, image 2 index, image 2 keypoint index]
        """
        matches_file = os.path.join(
            self._basedir, "match_{0}_{1}.dat".format(frame_1, frame_2))
        if os.path.isfile(matches_file) is False:
            return []
        _, matches = get_matches(matches_file)
        return matches

    def ransac_filter_keypoints(self, matches, idx1, idx2):
        """Use opencv ransac to filter matches."""
        src = np.array([], dtype=np.float).reshape(0, 2)
        dst = np.array([], dtype=np.float).reshape(0, 2)
        for match in matches:
            kp_src = self._image_features[idx1][int(match[1])]
            kp_dst = self._image_features[idx2][int(match[3])]
            src = np.vstack((src, [float(kp_src.x()), float(kp_src.y())]))
            dst = np.vstack((dst, [float(kp_dst.x()), float(kp_dst.y())]))

        if src.shape[0] < 6:
            return True, []

        src = np.expand_dims(src, axis=1)
        dst = np.expand_dims(dst, axis=1)
        # _, mask = cv2.findEssentialMat(
        #     dst, src, cameraMatrix=self._calibration.matrix(), method=cv2.RANSAC, prob=self._cv_prob, threshold=self._cv_threshold)
        _, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 0.01, 0.999)
        if mask is None:
            return True, np.array([])
        
        good_matches = [matches[i]
                        for i, score in enumerate(mask) if score == 1]
        return False, good_matches

    def generate_dsf(self, enable=True):
        """Use dsf to find data association between landmark and landmark observation(features)"""
        dsf = gtsam.DSFMapIndexPair()

        for i in range(0, self._nrimages-1):
            for j in range(i+1, self._nrimages):
                matches = self.load_matches(i, j)
                if enable:
                    bad_essential, matches = self.ransac_filter_keypoints(
                        matches, i, j)
                    if bad_essential:
                        print(
                            "Not enough points to generate essential matrix for image_", i, " and image_", j)
                        continue
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
                # If no matches exist return empty set.
                if matches == []:
                    return {}
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

    def filter_bad_landmarks(self, landmark_map, dsf, enable):
        """Filter bad landmarks:
            1. landmark observations<3
            2. landmarks with more than one observations in an image.
            3. Features with more than one landmark correspondences"""

        # filter bad matches
        if enable is True:
            bad_matches = self.find_bad_matches()
        else:
            bad_matches = {}
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

    def create_landmark_map(self, enable=True):
        """Create a list to map landmarks and their correspondences.
            [Landmark_i:[(i,Point2()), (j,Point2())...]...]"""
        dsf = self.generate_dsf(enable)
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
            # To test indeterminate system
            # if(landmark_idx == 477 or landmark_idx == 197 or landmark_idx == 204 or landmark_idx == 458 or landmark_idx == 627 or landmark_idx == 198):
            #     continue
            initial_estimate.insert(P(landmark_idx), landmark_3d_point)
        # Filter valid poses
        valid_pose_indices = set()
        for observation_list in self._landmark_map:
            for observation in observation_list:
                pose_idx = observation[0]
                valid_pose_indices.add(pose_idx)
        # Initial estimate for poses
        for pose_idx in valid_pose_indices:
            initial_estimate.insert(
                X(pose_idx), self._pose_estimates[pose_idx])

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
                # To test indeterminate system
                # if(landmark_idx == 477 or landmark_idx == 197 or landmark_idx == 204 or landmark_idx == 458 or landmark_idx == 627 or landmark_idx == 198):
                #     continue
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
        # Using QR rather than Cholesky decomposition
        # params = gtsam.LevenbergMarquardtParams()
        # params.setLinearSolverType("MULTIFRONTAL_QR")
        # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)

        sfm_result = optimizer.optimize()
        # Check if factor covariances are under constrain
        marginals = gtsam.Marginals(  # pylint: disable=unused-variable
            graph, sfm_result)
        return sfm_result

    def get_landmark_descriptor(self, observation_list):
        """Calculate the normalized average descriptor."""
        camera_number = len(observation_list)
        desc_sum = np.zeros((1, 256))
        for observation in observation_list:
            pose_idx = observation[0]
            for i, pose in enumerate(self._image_features[pose_idx]):
                if pose == observation[1]:
                    desc_idx = i
            desc = self._image_descriptors[pose_idx][desc_idx]
            desc_sum += np.array(desc)
        desc_average = desc_sum/camera_number
        desc_normalize = desc_average/np.linalg.norm(desc_average)
        return desc_normalize[0]

    def get_landmark_map(self):
        """Get landmark map."""
        return self._landmark_map

    def save_map_to_file(self, sfm_result):
        """Save the map result to a file."""
        sfm_map = []
        for i, observation_list in enumerate(self._landmark_map):
            landmark_pt = sfm_result.atPoint3(P(i))
            descriptor = self.get_landmark_descriptor(
                observation_list).tolist()
            landmark = [landmark_pt.x(), landmark_pt.y(), landmark_pt.z()]
            landmark.extend(descriptor)
            sfm_map.append(landmark)
        save_map_to_file(sfm_map, self._basedir)

    def save_poses_to_file(self, sfm_result):
        """Save poses to file."""
        valid_pose_indices = set()
        for observation_list in self._landmark_map:
            for observation in observation_list:
                pose_idx = observation[0]
                valid_pose_indices.add(pose_idx)
        camera_poses = []
        for idx in valid_pose_indices:
            r = sfm_result.atPose3(X(idx)).rotation().matrix()
            t = sfm_result.atPose3(X(idx)).translation().vector()
            camera_poses .append([t[0], t[1], t[2], r[0][0], r[0][1], r[0]
                                  [2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]])
        save_poses_to_file(camera_poses, self._basedir)
