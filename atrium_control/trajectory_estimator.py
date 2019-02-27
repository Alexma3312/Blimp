"""This is the Trajectory Estimator class."""
import math
import time

import cv2
import gtsam
import numpy as np
from gtsam import Point3, Pose3

import torch
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
    """ Read images and store image in an mage list
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
    image_1 = cv2.read_image('dataset/wall_corresponding_feature_data/raw_frame_left.jpg', image_size)
    image_2 = cv2.read_image('dataset/wall_corresponding_feature_data/raw_frame_middle.jpg', image_size)
    image_3 = cv2.read_image('dataset/wall_corresponding_feature_data/raw_frame_right.jpg', image_size)
    image_list.append(image_1)
    image_list.append(image_2)
    image_list.append(image_3)
    return image_list


class TrajectoryEstimator(object):

    def __init__(self, atrium_map, downsample_width, downsample_height, descriptor_threshold, feature_distance_threshold):
        """
        Args:
            atrium_map - a list of gtsam.Point3 objects
            fov_in_degrees - camera horizontal field of view (fov) in degrees
            downsample_width - image width downsample factor
            downsample_height - image height downsample factor
            image_width - input image width
            image_height - input image height
        """
        self.downsample_w = downsample_width
        self.downsample_h = downsample_height
        self.image_width = 640
        self.image_height = 480
        self.fov_in_degrees = 128
        self.image_list = read_images(self.downsample_w, self.downsample_h, self.image_width, self.image_height)
        self.calibration = gtsam.Cal3_S2(self.fov_in_degrees, self.image_width, self.image_height)
        self.atrium_map = atrium_map
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

        return superpoints, descriptors

    def landmarks_projection(self, pose):
        """
        Parameters:
            pose: gtsam.Point3, the pose of a camera
            image_width
        Returns:

        """
        projected_feature_points = []
        for point in self.atrium_map.landmark_points:
            camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
            # feature is gtsam.Point2 object
            feature = camera.project(point)
            if (feature.x() > 0 and feature.x() < self.image_width
                    and feature.y() > 0 and feature.y() < self.image_height):
                projected_feature_points.append(feature)

        return projected_feature_points

    def data_association(self, superpoints, superpoint_descriptors, projected_feature_points, projected_feature_descriptors):
        return

    def trajectory_estimator(self):
        """
        Input features with corresponding landmark indices.
        This will be used to create factors to connect estimate pose with landmarks in the map
        """
        return

#     def feature_point_extraction(self, superpoints, descriptors):

#         features_points = 0
#         return features_points

#     def feature_point_matching(self, landmark, features_points):

#         i = len(self.estimate_trajectory)


#         for point in landmark:
#             camera = gtsam.PinholeCameraCal3_S2(self.estimate_trajectory[i-1], self.calibration)
#             feature_data = camera.project(point)

#         return features_points


#     def initial_iSAM(self):

#         # Define the camera calibration parameters
#         fov_in_degrees, w, h = 128, 160, 120
#         self.calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)

#         # Define the camera observation noise model
#         self.measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
#         2, 1.0)  # one pixel in u and v

#         # Create an iSAM2 object.
#         parameters = gtsam.ISAM2Params()
#         parameters.setRelinearizeThreshold(0.01)
#         parameters.setRelinearizeSkip(1)
#         self.isam = gtsam.ISAM2(parameters)

#         # Create a Factor Graph and Values to hold the new data
#         self.graph = gtsam.NonlinearFactorGraph()
#         self.initial_estimate = gtsam.Values()

#         for i, point in enumerate(self.atrium_map):
#             pointNoiseSigmas = np.array([0.1, 0.1, 0.1])
#             pointPriorNoise = gtsam.noiseModel_Diagonal.Sigmas(pointNoiseSigmas)
#             self.graph.add(gtsam.PriorFactorPoint3(P(i),
#                                     point, pointPriorNoise))
#             # Add initial estimates for Points.

#         # Create the initial pose, X(0)
#         angle = 0
#         theta = np.radians(angle)
#         self.wRc = gtsam.Rot3(np.array([[0, math.cos(
#             theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)
#         wTi = gtsam.Pose3(self.wRc, gtsam.Point3(0, 0, 1.5))

#         pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array(
#                 [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
#         self.graph.push_back(gtsam.PriorFactorPose3(X(0), wTi, pose_noise))

#         #Update the estimate_trajectory
#         self.estimate_trajectory.append(wTi)

#     def first_frame_process(self, landmarks, extracted_feature_points):

#         # Project landmarks into the initial camera, which is placed at the initial pose to generate expected feature points
#         expected_feature_points = []
#         for point in landmarks:
#             camera = gtsam.PinholeCameraCal3_S2(self.estimate_trajectory[0], self.calibration)
#             expected_feature_points.append(camera.project(point))

#         # Compare and match expected_feature_points with extracted_feature_points to generate actual input feature points
#         feature_points = []
#         for i, point in enumerate(expected_feature_points):
#             feature_info = []
#             for feature in extracted_feature_points:
#                 if (point.equals(feature, 1e-2)):
#                     feature_info.append(point)
#                     feature_info.append(i)
#                     print("feature_info:",feature_info)
#                 feature_points.append(feature_info)
#         print(feature_points)

#         for i, feature in enumerate(feature_points):
#             print(feature[i][0])
#             print(feature[i][1])
#             self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
#                 feature[i][0], self.measurement_noise, X(0), P(feature[i][1]), self.calibration))

#         self.isam.update(self.graph, self.initial_estimate)

#         self.isam.update()
#         current_estimate = self.isam.calculateEstimate()
#         return current_estimate


#     def update_iSAM(self, input_features):

#         i = len(self.estimate_trajectory)

#         # Add factors for each landmark observation
#         for j, feature in enumerate(input_features):
#             self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
#                 feature, self.measurement_noise, X(i), P(j), self.calibration))

#         # Add an initial guess for the current pose
#         # Intentionally initialize the variables off from the ground truth
#         self.initial_estimate.insert(X(i), self.estimate_trajectory(i-1))

#         # Update iSAM with the new factors
#         self.isam.update(self.graph, self.initial_estimate)
#         # Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
#         # If accuracy is desired at the expense of time, update(*) can be called additional
#         # times to perform multiple optimizer iterations every step.
#         self.isam.update()
#         current_estimate = self.isam.calculateEstimate()

#         return current_estimate

#     def trajectory_estimator(self, image):
#         """Trajectory Estimator is a function based on iSAM to generate the estimate trajectory and the estimate state

#         Args:
#             image -- (H,W) numpy array of intensities, H is image height and W is image width

#         Returns:
#             estimate_trajectory -- (nrstate+1,6) numpy array of Pose3 values
#             estimate_state -- (1,12) numpy array of [x,y,z,row,pitch,yaw,dx,dy,dz,d(row),d(pitch),d(yaw)]
#         """

#         estimate_trajectory = []
#         current_estimate_state = []
#         return estimate_trajectory, current_estimate_state


# if __name__ == "__main__":
#     # Use the output of SFM as the map input
#     Atrium_Map = SFMdata.createPoints()

#     # Create a new Trajectory estimator object
#     Trajectory_Estimator = TrajectoryEstimator(Atrium_Map)

#     # superpoints_1, descriptors_1 = Trajectory_Estimator.superpoint_generator(
#     #     Trajectory_Estimator.image_1)

#     # Initialize a trajectory estimator object
#     Trajectory_Estimator.initial_iSAM()
