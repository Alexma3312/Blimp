"""This is the Trajectory Estimator class."""
import math
import time

import cv2
import gtsam
import numpy as np
from gtsam import Point3, Pose3

import torch
from atrium_control.mapping_and_localization_data import Features
from sfm import sfm_data
from SuperPointPretrainedNetwork.demo_superpoint import *


def X(i):
    """Create key for pose i."""
    return gtsam.symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return gtsam.symbol(ord('p'), j)


def read_image(impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(
        grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim


def read_images(downsample_width, downsample_height, width, height):
    """ Read images and store image in an image list
    Parameters:
        downsample_width - image width downsample factor
        downsample_height - image height downsample factor
        width - input image width
        height - input image height
    Returns:
        image_list - a list of images
    """
    image_size = [width/downsample_width, height/downsample_height]
    image_list = []
    image_1 = cv2.read_image(
        'datasets/wall_corresponding_feature_data/raw_frame_left.jpg', image_size)
    image_2 = cv2.read_image(
        'datasets/wall_corresponding_feature_data/raw_frame_middle.jpg', image_size)
    image_3 = cv2.read_image(
        'datasets/wall_corresponding_feature_data/raw_frame_right.jpg', image_size)
    image_list.append(image_1)
    image_list.append(image_2)
    image_list.append(image_3)
    return image_list


def nearest_match(desc1, desc2, nn_thresh, pts1, pts2, pts_x_distance, pts_y_distance):
    """
    This function is develop base on demo_superpoint nn_match_two_way(self, desc1, desc2, nn_thresh) 
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    Matches feature points that can meet the requirement of |pts1[0] - pts2[0]| < pts_x_distance 
    and |pts1[1] - pts2[1]| < pts_y_distance.

    Keyword arguments:
        desc1 -- MxN numpy matrix of previous image N corresponding M-dimensional descriptors.
        desc2 -- MxN numpy matrix of present image N corresponding M-dimensional descriptors.
        nn_thresh -- Optional descriptor distance below which is a good match.
        pts1 -- 3xN numpy array of previous image 2D point observations [x_i, y_i, confidence_i]^T.
        pts2 -- 3xN numpy array of present image 2D point observations [x_i, y_i, confidence_i]^T.
        pts_x_distance -- x coordinate matching distance range of feature pairs.
        pts_y_distance -- y coordinate matching distance range of feature pairs.

    Returns:
        matches -- 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = (np.arange(len(idx)) == idx2[idx])
    keep = np.logical_and(keep, keep_bi)
    # Check if the coordinates of matched feature pairs are within distance ranges.
    m_idx1 = np.arange(desc1.shape[1])
    m_idx2 = idx
    if pts1.size is not 0 and pts2.size is not 0:
        keep_distance_x = abs(
            pts1[0, m_idx1]-pts2[0, m_idx2]) <= pts_x_distance
        keep_distance_y = abs(
            pts1[1, m_idx1]-pts2[1, m_idx2]) <= pts_y_distance
        keep = np.logical_and(keep, keep_distance_x)
        keep = np.logical_and(keep, keep_distance_y)
    # Get the surviving point indices.
    scores = scores[keep]
    m_idx1 = m_idx1[keep]
    m_idx2 = m_idx2[keep]
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


class TrajectoryEstimator(object):

    def __init__(self, atrium_map, past_pose, downsample_width, downsample_height, descriptor_threshold, feature_distance_threshold):
        """ A class to estimate the current camera pose with prebuilt map, an image, and the past pose.
        Args:
            atrium_map - a map object that include a N*3 numpy array of gtsam.Point3 and a N*M numpy array of M dimensions descriptors
            fov_in_degrees - camera horizontal field of view (fov) in degrees
            downsample_width - image width downsample factor
            downsample_height - image height downsample factor
            descriptor_threshold
            feature_distance_threshold
        """
        self.downsample_w = downsample_width
        self.downsample_h = downsample_height
        self.image_width = 640
        self.image_height = 480
        self.fov_in_degrees = 128
        self.calibration = gtsam.Cal3_S2(
            self.fov_in_degrees, self.image_width, self.image_height)
        self.atrium_map = atrium_map
        self.pose = past_pose
        self.threshold_d = descriptor_threshold
        self.threshold_f = feature_distance_threshold

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Returns:
            superpoint - Nx2 numpy array of 2D point observations.
            descriptors - Nx256 numpy array of corresponding unit normalized descriptorss.
        """

        # Refer to /SuperPointPretrainedNetwork/demo_superpoint for more information about the parameters
        # Output of SuperpointFrontend.run():
        #   corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
        #   desc - 256xN numpy array of corresponding unit normalized descriptors.
        #   heatmap - HxW numpy heatmap in range [0,1] of point confidences.
        fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                nms_dist=4,
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=False)
        superpoints, descriptors, _ = fe.run(image)

        superpoints = np.transpose(superpoints[:2, ])
        descriptors = np.transpose(descriptors)

        return Features(superpoints, descriptors)

    def landmarks_projection(self):
        """ Project landmark points in the map to the camera to filter landmark points outside the view of the current camera pose 
        Parameters:
            pose: gtsam.Point3, the pose of a camera
        Returns:

        """

        points = []
        descriptors = []
        map_indices = []

        for i, landmark_point in enumerate(self.atrium_map.landmark_list):
            camera = gtsam.PinholeCameraCal3_S2(self.pose, self.calibration)
            # feature is gtsam.Point2 object
            feature_point = camera.project(landmark_point)
            print("feature:", feature_point)
            if (feature_point.x() > 0 and feature_point.x() < self.image_width
                    and feature_point.y() > 0 and feature_point.y() < self.image_height):
                points.append(feature_point)
                descriptors.append(self.atrium_map.get_descriptor_from_list(i))
                map_indices.append(i)
        print("projected_feature_points:", points)
        print("projected_feature_points:", descriptors)
        print("projected_feature:", map_indices)

        return Features(np.array(points), np.array(descriptors)), map_indices

    def data_association(self, superpoint_features, projected_features, map_indices):
        """ Associate feature points with landmark points by matching all superpoint features with projected features.
        Parameters:
            superpoint_features:
            projected_features:

        Returns:
            matched_features:
            visible_map:
        """

        pts1 = superpoint_features.key_points
        desc1 = superpoint_features.descriptors

        pts2 = projected_features.key_points
        desc2 = projected_features.descriptors

        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1, desc2.T)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = (np.arange(len(idx)) == idx2[idx])
        keep = np.logical_and(keep, keep_bi)
        # Check if the coordinates of matched feature pairs are within distance ranges.
        m_idx1 = np.arange(desc1.shape[1])
        m_idx2 = idx
        if pts1.size is not 0 and pts2.size is not 0:
            keep_distance_x = abs(
                pts1[0, m_idx1]-pts2[0, m_idx2]) <= pts_x_distance
            keep_distance_y = abs(
                pts1[1, m_idx1]-pts2[1, m_idx2]) <= pts_y_distance
            keep = np.logical_and(keep, keep_distance_x)
            keep = np.logical_and(keep, keep_distance_y)
        # Get the surviving point indices.
        scores = scores[keep]
        m_idx1 = m_idx1[keep]
        m_idx2 = m_idx2[keep]
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores

        matched_features_points = []
        matched_features_descriptors = []

        matched_landmark_points = []
        matched_landmark_decriptors = []

        for idx in m_idx1:
            matched_features_points.append(
                superpoint_features.get_keypoint_from_list(idx))
            matched_features_descriptors.append(
                superpoint_features.get_descriptor_from_list(idx))

        for idx in m_idx2:
            matched_landmark_points.append(
                self.atrium_map.get_landmark_from_list(map_indices(idx)))
            matched_landmark_decriptors.append(
                self.atrium_map.get_descriptor_from_list(map_indices(idx)))

        return Features(np.array(matched_features_points), np.array(matched_features_descriptors)), Map(np.array(matched_landmark_points), np.array(matched_landmark_decriptors))

    def trajectory_estimator(self, features, visible_map):
        """
        Input features with corresponding landmark indices.
        This will be used to create factors to connect estimate pose with landmarks in the map
        """

        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        for i,feature in enumerate(features.key_point_list):
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    feature, measurementNoise,
                    X(0), P(i), self.calibration))

        # Create initial estimate for the first pose
        initialEstimate.insert(X(0), self.pose)

        # Create priors for visual 
        posePriorNoise = gtsam.noiseModel_Isotropic.Sigma(6, 0)
        for i,landmark in enumerate(visible_map.landmark_list):
            graph.add(gtsam.PriorFactorPoint3(P(i),
                                landmark, pointPriorNoise))
            initialEstimate.insert(P(i), landmark)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()

        return result
