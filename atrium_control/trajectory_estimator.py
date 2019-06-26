"""This is the Trajectory Estimator module."""
import math
import sys
import time

import cv2
import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3, Rot3, symbol

import torch
from atrium_control.feature import Features
from atrium_control.map import Map
from sfm import sfm_data
from SuperPointPretrainedNetwork.demo_superpoint import *

sys.path.append('../')


def X(i):
    """Create key for pose i."""
    return gtsam.symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return gtsam.symbol(ord('p'), j)


def read_image(impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs:
      impath - Path to input image.
      img_size - (W, H) tuple specifying resize size.
    Returns:
      grayim - float32 numpy array sized H x W with values in range [0, 1].
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
        """ A class to estimate the current camera pose with prebuilt map, an image, and a trajectory included the past pose.
        Args:
            atrium_map - a map object that includes a N*3 numpy array of gtsam.Point3 and a N*M numpy array of M dimensions descriptors
            downsample_width - image width downsample factor
            downsample_height - image height downsample factor
            l2_thresh - l2 distance threshold of two descriptors 
            distance_thresh - feature coordinate distance threshold
        """
        # Camera settings
        self.image_width = 640
        self.image_height = 480
        self.fov_in_degrees = 128
        self.calibration = gtsam.Cal3_S2(fx=333.4, fy=314.7,s=0,u0=303.6, v0=247.6)
        # gtsam.Cal3_S2(
        #     self.fov_in_degrees, self.image_width, self.image_height)
        self.distortion = np.array([-0.282548, 0.054412, -0.001882, 0.004796, 0.000000])

        self.projection = np.array([[226.994629 ,0.000000, 311.982613, 0.000000],[0.000000, 245.874146, 250.410089, 0.000000],[0.000000, 0.000000, 1.000000, 0.000000]])


        self.atrium_map = atrium_map
        self.downsample_w = downsample_width
        self.downsample_h = downsample_height
        self.l2_threshold = l2_thresh
        self.distance_threshold = distance_thresh

    def undistort_points(self, points):
        output = np.array([], dtype=np.float).reshape(0,2)
        rectification = np.identity(3)
        undistort_points = cv2.undistortPoints(np.expand_dims(points, axis=1), cameraMatrix=self.calibration.matrix(), distCoeffs=self.distortion, P=self.projection)
        for j,point in enumerate(undistort_points):
            output = np.vstack((output,point))
        return output

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Returns:
            superpoint - Nx2 (gtsam.Point2) numpy array of 2D point observations.
            descriptors - Nx256 numpy array of corresponding unit normalized descriptors.

        Refer to /SuperPointPretrainedNetwork/demo_superpoint for more information about the parameters
        Output of SuperpointFrontend.run():
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
        """
        fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                nms_dist=4,
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=False)
        superpoints, descriptors, _ = fe.run(image)

        # Transform superpoints from 3*N numpy array to N*2 numpy array
        superpoints = 4*np.transpose(superpoints[:2, ])
        superpoints=self.undistort_points(superpoints)

        # Transform descriptors from 256*N numpy array to N*256 numpy array
        descriptors = np.transpose(descriptors)

        # Transform superpoint into gtsam.Point2
        superpoints_reformat = [Point2(point) for point in superpoints]

        return Features(np.array(superpoints_reformat), descriptors)

    def landmarks_projection(self, pose):
        """ Project landmark points in the map to the given camera pose. 
            And filter landmark points outside the view of the current camera pose.
        Parameters:
            pose: gtsam.Point3, the pose of a camera
        Returns:
            Features: A Feature object. Contains all the projected features that can be viewed at the current camera pose.  
            map_indices: A list of indices. Records the corresponding indices in the map of projected feature. 
                        The length is the number of feature points in Features.
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

            # Check if the projected feature is within the field of view.
            if (feature_point.x() > 0 and feature_point.x() < self.image_width
                    and feature_point.y() > 0 and feature_point.y() < self.image_height):
                points.append(feature_point)
                descriptors.append(self.atrium_map.get_descriptor_from_list(i))
                map_indices.append(i)
        return Features(np.array(points), np.array(descriptors)), map_indices

    def data_association_distance(self):
        return
        
    def data_association(self, superpoint_features, projected_features, map_indices):
        """ Associate Superpoint feature points with landmark points by matching all superpoint features with projected features.
        Parameters:
            superpoint_features: N element list each element is a Feature Object
            projected_features: M element list each element is a Feature Object
            map_indices: A list of indices. Records the corresponding indices in the map of projected feature. 
                        The length is the number of feature points in Features.

        Returns:
            matched_features: A Feature Object. Stores all matched Superpoint features.
            visible_map: A Map Object. Stores all matched Landmark points.

        1. For each superpoint feature calculate the distances between itself and all the points in the map
        2. Filter the points within the distance threshold
        3. Calculate the L2 distance of the superpoint feature point and the filtered points
        4. Select the smallest L2 distance point

        # """
        if superpoint_features.get_length() == 0 or projected_features.get_length() == 0:
            return Features(np.array([]), np.array([])), Map(np.array([]), np.array([]))
        if self.l2_threshold < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')

        matched_features_points = []
        matched_features_descriptors = []

        matched_landmark_points = []
        matched_landmark_decriptors = []

        for i, projected_point in enumerate(projected_features.key_point_list):

            feature_index_list = []
            min_score = self.l2_threshold
            matched_landmark_idx = 0

            # Calculate the pixels distances between current superpoint and all the points in the map
            for j, superpoint in enumerate(superpoint_features.key_point_list):

                # print('x',abs(superpoint.x() - projected_point.x()))
                # print('y',abs(superpoint.y() - projected_point.y()))
                x_diff = abs(superpoint.x() - projected_point.x())
                y_diff = abs(superpoint.y() - projected_point.y())
                if(abs(superpoint.x() - projected_point.x()) < self.distance_threshold and abs(superpoint.y() - projected_point.y()) < self.distance_threshold):
                    feature_index_list.append(j)
                    # print('diff',i,j,x_diff,y_diff)
            if feature_index_list == []:
                continue
            # print(feature_index_list)
            for feature_index in feature_index_list:
                # Compute L2 distance. Easy since vectors are unit normalized.
                dmat = np.dot(superpoint_features.descriptor(
                    feature_index), projected_features.descriptor(i).T)
                dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
                # print("dmat",i ,feature_index,dmat)

                # Select the minimal L2 distance point
                if dmat < min_score and dmat>0.8:
                    min_score = dmat
                    matched_landmark_idx = feature_index

            # Append feature and landmark if the L2 distance satisfy the L2 threshold.
            if min_score < self.l2_threshold:
                matched_features_points.append(
                    superpoint_features.get_keypoint_from_list(matched_landmark_idx))
                matched_features_descriptors.append(
                    superpoint_features.get_descriptor_from_list(matched_landmark_idx))

                matched_landmark_points.append(
                    self.atrium_map.get_landmark_from_list(map_indices[i]))
                matched_landmark_decriptors.append(
                    self.atrium_map.get_descriptor_from_list(map_indices[i]))
            # print(matched_features_points)
        return Features(np.array(matched_features_points), np.array(matched_features_descriptors)), Map(np.array(matched_landmark_points), np.array(matched_landmark_decriptors))

    def trajectory_estimator(self, features, visible_map):
        """ Estimate current pose with matched features through GTSAM and update the trajectory
        Parameters:
            features: A Feature Object. Stores all matched Superpoint features.
            visible_map: A Map Object. Stores all matched Landmark points.
            pose: gtsam.pose3 Object. The pose at the last state from the atrium map trajectory.
        Returns:
            current_pose: gtsam.pose3 Object. The current estimate pose.

        Use input matched features as the projection factors of the graph.  
        Use input visible_map (match landmark points of the map) as the landmark priors of the graph.
        Use the last time step pose from the trajectory as the initial estimate value of the current pose
        """
        assert len(self.atrium_map.trajectory) != 0, "Trajectory is empty."
        pose = self.atrium_map.trajectory[len(self.atrium_map.trajectory)-1]

        # Initialize factor graph
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add projection factors with matched feature points for all measurements
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
        # Because the map is known, we use the landmarks from the visible map with nearly zero error as priors.
        pointPriorNoise = gtsam.noiseModel_Isotropic.Sigma(3, 0.01)
        for i, landmark in enumerate(visible_map.landmark_list):
            
            graph.add(gtsam.PriorFactorPoint3(P(i),
                                              landmark, pointPriorNoise))
            initialEstimate.insert(P(i), landmark)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()

        # Add current pose to the trajectory
        current_pose = result.atPose3(symbol(ord('x'), 0))
        self.atrium_map.append_trajectory(current_pose)

        return current_pose
