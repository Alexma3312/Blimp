# cSpell: disable=invalid-name
"""
A Mapping Pipeline: feature match info parser, data association, and Bundle Adjustment(gtsam).
"""
# pylint: disable=invalid-name, no-name-in-module, no-member, assignment-from-no-return,line-too-long


import copy
import os

import cv2
import numpy as np

import gtsam
from feature_matcher.parser import get_matches, load_features_list
from gtsam import (  # pylint: disable=wrong-import-order,ungrouped-imports
    Point2, Point3, Pose3, Rot3, symbol)


def X(i):
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return symbol(ord('p'), j)


def transform_from(T, pose):
    """ Calculate the Euclidean transform of a Pose3
    Parameters:
        pose - Pose3 object, SE3
    Returns:
        Pose3 object, SE3
    """
    R = T.rotation().compose(pose.rotation())
    translation = T.rotation().rotate(pose.translation()).vector() + \
        T.translation().vector()
    return Pose3(R, Point3(translation))


class ImageFeature():
    """
    Store feature key points and descriptors information.
        self._descriptors - Nx256 list
        self.keypoints - Nx2 list
    """

    def __init__(self, keypoints, descriptors):
        # Initialize image information including key points and feature descriptors
        self.keypoints = keypoints
        self._descriptors = descriptors


class ImageMatches():
    """
    Store both image and pose information.
        self.kp_matches - a dictionary, {current image key point index: {matched image index: matched image key point index}}
    """

    def __init__(self):
        # Initialize a dictionary to manage matching information between
        # current image keypoints indices and keypoints in other images
        self.kp_matches = {}

    def kp_match_idx(self, kp_idx, img_idx):
        """
        Use key point idx and an image index to return the matched key point in the img_idx image
        """
        return self.kp_matches.get(kp_idx).get(img_idx)

    def kp_match_exist(self, kp_idx, img_idx):
        """
        Check if kp_idx keypoint has matched keypoints within the img_index image features
        """
        if self.kp_matches.get(kp_idx) is None:
            return False
        # Return True if img_idx is in self.kp_matches[kp_idx] else False
        return self.kp_matches.get(kp_idx).get(img_idx) is not None

    # I think dual dictionary is better than list, the following code is for list
    # def kp_match_idx(self, kp_idx, img_idx):
    #     """
    #     Use key point idx and an image index to return the matched key point in the img_idx image
    #     """
    #     match_list = self.kp_matches.get(kp_idx)
    #     for match in match_list:
    #         if match[0] == img_idx:
    #             return match[1]

    #     return False


class LandMark():
    """
    Store landmark information.
        self.point - 1X3 np array
        self.seen - the number of associated poses
    """

    def __init__(self, point=np.array([0, 0, 0]), seen=0):
        self.point = point
        self.seen = seen


class ImagePoints():
    """
    Store landmarks and poses corresponding information.
        self.kp_landmark - a dictionary, {key point index: landmark index}
    """

    def __init__(self, kp_landmark):
        # Initialize a dictionary to manage matching information between
        # keypoints indices and their corresponding landmarks indices
        self.kp_landmark = kp_landmark

    def kp_3d(self, kp_idx):
        """
        Use key point idx to return the corresponding landmark index
        """
        return self.kp_landmark.get(kp_idx)

    def kp_3d_exist(self, kp_idx):
        """
        Check if there are corresponding landmark for kp_idx keypoint
        """
        return self.kp_landmark.get(kp_idx) is not None


class ImagePose():
    """
    Store pose transformation and projection matrix.
        self.T - transformation matrix, gtsam.Pose3 object
        self.keypoints - projection matrix, 3X4 numpy array
    """

    def __init__(self, calibration):
        # Initialize image information including key points and feature descriptors
        self.T = Pose3(Rot3(), Point3())
        self.P = np.dot(calibration, np.hstack(
            (np.identity(3), np.zeros((3, 1)))))


class MappingBackEnd():
    """
    Mapping Back End.
        data_directory - input files(feature and feature match data) directory path
        num_images - number of input images
        pose_prior - pose prior data, used in initial estimation
        calibration - camera calibration, gtsam.Cal3_S2
        backprojection_depth - the estimated depth used in back projection
    """

    def __init__(self, data_directory, num_images, calibration, pose_estimates, measurement_noise, pose_prior_noise, min_landmark_seen, backprojection_depth=20):
        """Construct by reading from a data directory."""
        self._basedir = data_directory
        self._nrimages = num_images
        self._calibration = calibration
        self._min_landmark_seen = min_landmark_seen
        self._depth = backprojection_depth
        self._pose_estimates = pose_estimates
        self._measurement_noise = measurement_noise
        self._pose_prior_noise = pose_prior_noise

        self._image_features = [self.load_image_features(
            image_index) for image_index in range(self._nrimages)]
        self._image_matches = self.load_image_matches()
        self.image_points, self._landmark_estimates, self.image_poses = self.data_associate_all_images()

    def load_image_features(self, image_index):
        """ Load features from .key files
            features - keypoints:A Nx2 list of (x,y). Descriptors: A Nx256 list of descriptors.
        """
        feat_file = os.path.join(
            self._basedir, "{0:07}.key".format(image_index))
        keypoints, descriptors = load_features_list(feat_file)
        return ImageFeature(keypoints, descriptors)

    def get_matches_from_file(self, image_1, image_2):
        """ Load matches from .dat files
            matches - a list of [image 1 index, image 1 keypoint index, image 2 index, image 2 keypoint index]
        """
        matches_file = os.path.join(
            self._basedir, "match_{0}_{1}.dat".format(image_1, image_2))
        _, matches = get_matches(matches_file)
        return matches

    def load_image_matches(self):
        """
        Load feature match data of all image pairs iteratively and store all match information into one data structure.
            Input:
                self._basedir- the base directory that stores all images.
                self._nrimages- number of images
            Output:
                image_matches: a list of ImageMatches objects.
        """
        image_matches = [ImageMatches() for i in range(self._nrimages)]
        # Iterate through all images and add frame i and frame i+1 matching data
        for i in range(self._nrimages-1):
            matches = self.get_matches_from_file(i, i+1)
            for match in matches:
                # Update kp_matches dictionary in both frame i and frame i+1
                if image_matches[i].kp_matches.get(match[1]) is None:
                    image_matches[i].kp_matches[match[1]] = {}
                image_matches[i].kp_matches[match[1]][i+1] = match[3]
                if image_matches[i+1].kp_matches.get(match[3]) is None:
                    image_matches[i+1].kp_matches[match[3]] = {}
                image_matches[i+1].kp_matches[match[3]][i] = match[1]

        return image_matches

    def initial_estimate(self, idx, kp_idx_matches, input_image_poses):
        """
        Estimate pose and landmark values.
            idx - 
            kp_idx_matches - 
            input_image_poses -
        """
        kp_matches = [[self._image_features[idx].keypoints[int(
            match[0])], self._image_features[idx+1].keypoints[int(match[1])]] for match in kp_idx_matches]
        src = np.expand_dims(np.array(kp_matches)[:, 0], axis=1)
        dst = np.expand_dims(np.array(kp_matches)[:, 1], axis=1)

        next_frame = False
        if src.shape[0] < 6:
            print("Warning:Not enough points to generate essential matrix for image_%d and image_%d.", (idx, idx+1))
            next_frame = True

        image_poses = copy.copy(input_image_poses)
        E, mask = cv2.findEssentialMat(
            dst, src, cameraMatrix=self._calibration.matrix(), method=cv2.RANSAC, prob=0.9, threshold=3)

        _, local_R, local_t, _ = cv2.recoverPose(
            E, dst, src, cameraMatrix=self._calibration.matrix())
        # print("R=", local_R)
        # print("t=", local_t)

        T = Pose3(Rot3(local_R), Point3(local_t[0], local_t[1], local_t[2]))

        image_poses[idx + 1].T = transform_from(T, image_poses[idx].T)
        cur_R = image_poses[idx+1].T.rotation().matrix()
        cur_t = image_poses[idx+1].T.translation().vector()
        cur_t = np.expand_dims(cur_t, axis=1)
        image_poses[idx+1].P = np.dot(self._calibration.matrix(),
                                      np.hstack((cur_R.T, -1*np.dot(cur_R.T, cur_t))))

        points4D = cv2.triangulatePoints(
            projMatr1=image_poses[idx].P, projMatr2=image_poses[idx+1].P, projPoints1=src, projPoints2=dst)
        points4D = points4D / np.tile(points4D[-1, :], (4, 1))
        pt3d = points4D[:3, :].T
        return pt3d, mask, image_poses, next_frame

    def data_associate_single_image(self, idx, in_image_points, in_landmarks, in_image_poses):
        """
        Find feature data association within input image features.
        """
        # Get keypoints and keypoint indices
        matches = self._image_matches[idx]
        kp_idx_matches = [[int(k), int(matches.kp_match_idx(k, idx+1))]
                          for k in matches.kp_matches if matches.kp_match_idx(k, idx+1) is not None]

        pt3d, mask, image_poses, next_frame = self.initial_estimate(
            idx, kp_idx_matches, in_image_poses)
        if next_frame:
            return in_image_points, in_landmarks, in_image_poses

       # data association update
        image_points = copy.deepcopy(in_image_points)
        landmarks = copy.deepcopy(in_landmarks)
        for j, kp_idx_match in enumerate(kp_idx_matches):
            src_idx = kp_idx_match[0]
            if mask[j]:
                match_idx = kp_idx_match[1]
                if image_points[idx].kp_3d_exist(src_idx):
                    image_points[idx +
                                 1].kp_landmark[match_idx] = image_points[idx].kp_3d(src_idx)
                    landmarks[image_points[idx].kp_3d(
                        src_idx)].point += pt3d[j]
                    landmarks[image_points[idx +
                                           1].kp_3d(match_idx)].seen += 1
                else:
                    new_landmark = LandMark(pt3d[j], 2)
                    landmarks.append(new_landmark)

                    image_points[idx].kp_landmark[src_idx] = len(
                        landmarks) - 1
                    image_points[idx +
                                 1].kp_landmark[match_idx] = len(landmarks) - 1

        return image_points, landmarks, image_poses

    def data_associate_all_images(self):
        """
        Find feature data association across all images incrementally.
        """
        image_poses = [ImagePose(self._calibration.matrix())
                       for i in range(self._nrimages)]
        image_points = [ImagePoints({}) for i in range(self._nrimages)]
        landmarks = [LandMark()]
        for i in range(self._nrimages-1):
            image_points, landmarks, image_poses = self.data_associate_single_image(
                i, image_points, landmarks, image_poses)

        for j, landmark in enumerate(landmarks):
            update_landmarks = copy.deepcopy(landmarks)
            if landmark.seen >= 3:
                update_landmarks[j].point /= (update_landmarks[j].seen - 1)

        return image_points, update_landmarks, image_poses

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

    def create_index_sets(self):
        """Create two sets with valid pose and point indices."""
        point_indices = set()
        pose_indices = set()
        for img_idx, features in enumerate(self._image_features):
            for kp_idx, _ in enumerate(features.keypoints):
                if self.image_points[img_idx].kp_3d_exist(kp_idx):
                    landmark_id = self.image_points[img_idx].kp_3d(kp_idx)
                    if self._landmark_estimates[landmark_id].seen >= self._min_landmark_seen:
                        # Filter invalid landmarks
                        point_indices.add(landmark_id)
                        # Filter invalid image poses
                        pose_indices.add(img_idx)

        return pose_indices, point_indices

    def create_initial_estimate(self, pose_indices):
        """Create initial estimate from data association."""
        initial_estimate = gtsam.Values()

        # Create dictionary for initial estimation
        for img_idx, features in enumerate(self._image_features):
            for kp_idx, keypoint in enumerate(features.keypoints):
                if self.image_points[img_idx].kp_3d_exist(kp_idx):
                    landmark_id = self.image_points[img_idx].kp_3d(kp_idx)
                    # Filter invalid landmarks
                    if self._landmark_estimates[landmark_id].seen >= self._min_landmark_seen:
                        key_point = Point2(keypoint[0], keypoint[1])
                        key = P(landmark_id)
                        if not initial_estimate.exists(key):
                            # do back-projection
                            pose = self._pose_estimates[img_idx]
                            landmark_3d_point = self.back_projection(
                                key_point, pose, self._depth)
                            initial_estimate.insert(
                                P(landmark_id), landmark_3d_point)

        # Initial estimate for poses
        for idx in pose_indices:
            pose_i = self._pose_estimates[idx]
            initial_estimate.insert(X(idx), pose_i)

        return initial_estimate

    def bundle_adjustment(self):
        """
        Bundle Adjustment.
        Input:
            self._image_features
            self._image_points
        Output:
            result - gtsam optimzation result
        """
        # Initialize factor Graph
        graph = gtsam.NonlinearFactorGraph()

        pose_indices, point_indices = self.create_index_sets()

        initial_estimate = self.create_initial_estimate(pose_indices)

        # """
        #   Create measurement noise for bundle adjustment:
        #   sigma = 1.0
        #   measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, sigma)
        # """
        # Create Projection Factors
        for img_idx, features in enumerate(self._image_features):
            for kp_idx, keypoint in enumerate(features.keypoints):
                if self.image_points[img_idx].kp_3d_exist(kp_idx):
                    landmark_id = self.image_points[img_idx].kp_3d(kp_idx)
                    if self._landmark_estimates[landmark_id].seen >= self._min_landmark_seen:
                        key_point = Point2(keypoint[0], keypoint[1])
                        graph.add(gtsam.GenericProjectionFactorCal3_S2(
                            key_point, self._measurement_noise,
                            X(img_idx), P(landmark_id), self._calibration))

        # """
        #   Create pose prior noise:
        #   rotation_sigma = np.radians(60)
        #   translation_sigma = 1
        #   pose_noise_sigmas = np.array([rotation_sigma, rotation_sigma, rotation_sigma,
        #                             translation_sigma, translation_sigma, translation_sigma])
        # """
        # Create priors for first two poses
        for idx in (0, 1):
            pose_i = initial_estimate.atPose3(X(idx))
            graph.add(gtsam.PriorFactorPose3(
                X(idx), pose_i, self._pose_prior_noise))

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        sfm_result = optimizer.optimize()

        return sfm_result, pose_indices, point_indices
