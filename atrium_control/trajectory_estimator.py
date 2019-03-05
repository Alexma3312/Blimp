"""This is the Trajectory Estimator class."""
import math
import time

import cv2
import gtsam
import numpy as np
from gtsam import Point3, Point2, Pose3, symbol

import torch
from atrium_control.mapping_and_localization_data import Features, Map
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


class TrajectoryEstimator(object):

    def __init__(self, atrium_map, downsample_width, downsample_height, l2_thresh, distance_thresh):
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
        self.distance_threshold = distance_thresh
        self.l2_threshold = l2_thresh

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
        print(superpoints)

        # Transform superpoints from 3*N numpy array to N*2 numpy array
        # Transform descriptors from 256*N numpy array to N*256 numpy array
        superpoints = np.transpose(superpoints[:2, ])
        descriptors = np.transpose(descriptors)
        print(type(superpoints))
        print(superpoints[0])
        print(np.array(superpoints[0:3:]))

        # Transform superpoint into gtsam.Point2
        # superpoints_reformat = []
        # for point in superpoints:
        #     superpoints_reformat.append(Point2(point))
        superpoints_reformat = [Point2(point) for point in superpoints]
        print(superpoints_reformat[0:3:])
        # print(superpoints_reformat)
        return Features(np.array(superpoints_reformat), descriptors)

    def landmarks_projection(self, pose):
        """ Project landmark points in the map to the camera to filter landmark points outside the view of the current camera pose 
        Parameters:
            pose: gtsam.Point3, the pose of a camera
        Returns:

        """
        # Check if the atrium map is empty
        assert self.atrium_map.get_length(), "the atrium map is empty"

        points = []
        descriptors = []
        map_indices = []

        for i, landmark_point in enumerate(self.atrium_map.landmark_list):
            camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
            # feature is gtsam.Point2 object
            feature_point = camera.project(landmark_point)
            if (feature_point.x() > 0 and feature_point.x() < self.image_width
                    and feature_point.y() > 0 and feature_point.y() < self.image_height):
                points.append(feature_point)
                descriptors.append(self.atrium_map.get_descriptor_from_list(i))
                map_indices.append(i)
        # print("projected_feature_points:\n", points)
        # print("projected_feature_descriptors:", descriptors)
        # print("projected_feature_map_indices:", map_indices)

        return Features(np.array(points), np.array(descriptors)), map_indices

    def data_association(self, superpoint_features, projected_features, map_indices):
        """ Associate feature points with landmark points by matching all superpoint features with projected features.
        Parameters:
            superpoint_features: 2*N numpy array
            projected_features: 256*N numpy array

        Returns:
            matched_features:
            visible_map:

        1. For each superpoint feature calculate the distances between itself and all the points in the map
        2. Filter the points within the distance threshold
        3. Calculate the L2 distance of the superpoint feature point and the filtered points
        4. Choose the smallest L2 distance point

        """

        if superpoint_features.get_length() == 0 or projected_features.get_length() == 0:
            return Features(np.array([]), np.array([])), Map(np.array([]), np.array([]))
        if self.l2_threshold < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')

        matched_features_points = []
        matched_features_descriptors = []

        matched_landmark_points = []
        matched_landmark_decriptors = []

        for i, superpoint in enumerate(superpoint_features.key_point_list):

            feature_index_list = []
            min_score = self.l2_threshold
            matched_landmark_idx = 0

            for j, projected_point in enumerate(projected_features.key_point_list):
                if(abs(superpoint.x() - projected_point.x()) < self.distance_threshold and abs(superpoint.y() - projected_point.y()) < self.distance_threshold):
                    feature_index_list.append(j)

            for feature_index in feature_index_list:
                # Compute L2 distance. Easy since vectors are unit normalized.
                dmat = np.dot(superpoint_features.descriptor(
                    i), projected_features.descriptor(feature_index).T)
                dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))

                if dmat < min_score:
                    min_score = dmat
                    matched_landmark_idx = feature_index

            if min_score < self.l2_threshold:
                matched_features_points.append(
                    superpoint_features.get_keypoint_from_list(i))
                matched_features_descriptors.append(
                    superpoint_features.get_descriptor_from_list(i))

                matched_landmark_points.append(
                    self.atrium_map.get_landmark_from_list(map_indices[matched_landmark_idx]))
                matched_landmark_decriptors.append(
                    self.atrium_map.get_descriptor_from_list(map_indices[matched_landmark_idx]))
        return Features(np.array(matched_features_points), np.array(matched_features_descriptors)), Map(np.array(matched_landmark_points), np.array(matched_landmark_decriptors))

    def trajectory_estimator(self, features, visible_map, pose):
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
        for i, feature in enumerate(features.key_point_list):
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                feature, measurementNoise,
                X(0), P(i), self.calibration))

        # Create initial estimate for the pose
        # Because the robot moves slowly, we can use the previous pose as an initial estimation of the current pose.
        initialEstimate.insert(X(0), pose)

        # Create priors for visual
        pointPriorNoise = gtsam.noiseModel_Isotropic.Sigma(3, 0.01)
        for i, landmark in enumerate(visible_map.landmark_list):
            graph.add(gtsam.PriorFactorPoint3(P(i),
                                              landmark, pointPriorNoise))
            initialEstimate.insert(P(i), landmark)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()

        print(result.atPose3(symbol(ord('x'), 0)))

        return result
