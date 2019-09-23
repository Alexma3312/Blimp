"""Trajectory Estimator"""
# cSpell: disable
# pylint: disable=no-name-in-module,wrong-import-order, no-member,ungrouped-imports, invalid-name
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import gtsam
from feature_matcher.mapping_result_helper import load_map_from_file
from gtsam import Point2, Point3, symbol
from localization.features import Features
from localization.landmarks import Landmarks
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from utilities.plotting import plot_trajectory
from utilities.video_streamer import VideoStreamer


def X(i):  # pylint: disable=invalid-name
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):  # pylint: disable=invalid-name
    """Create key for landmark j."""
    return symbol(ord('p'), j)


class TrajectoryEstimator():
    """Trajectory Estimator"""

    def __init__(self, initial_pose, directory_name, calibration, image_size, l2_threshold, distance_thresh):
        self.initial_pose = initial_pose

        file_name = directory_name+"map.dat"

        def load_map(file_name):
            """Load map data from file."""
            if os.path.isfile(file_name) is False:
                print("\nFile name does not exist, the map will be empty.")
                return Landmarks([], [])
            landmark_3d_pts, landmark_desc = load_map_from_file(file_name)
            input_map = Landmarks(landmark_3d_pts, landmark_desc)
            return input_map
        self.map = load_map(file_name)
        # calibration data type gtsam.Cal3_S2
        self.calibration = calibration
        self.width = image_size[0]
        self.height = image_size[1]
        self.l2_threshold = l2_threshold
        self.x_distance_thresh = distance_thresh[0]
        self.y_distance_thresh = distance_thresh[1]

    def load_map(self, landmarks):
        """Load map data"""
        self.map = landmarks

    def detect_bad_frame(self, next_frame):
        """Check to see if the next frame is a bad frame."""
        return False

    def trajectory_generator(self, calibration, distortion, images_input, camid, skip, img_glob, start_index):
        """The top level function, use to generate trajectory.
        Input:
            initial_pose - Pose3 Object
            calibration - np.array
            distortion - np.array
            images_input - "camera" if input is Webcam. directory if input is images or video file.
            camid - Webcam id
            skip - number of frames to skip
            img_glob - image suffix(extension), e.g. "*.jpg"
            start_index - the number of frames to jump at the beginning
        Output:
            trajectory - a list of Pose3 objects

        """
        trajectory = [self.initial_pose]
        # This class helps load input images from different sources.
        vs = VideoStreamer(images_input, camid, self.height,
                           self.width, skip, img_glob, start_index)
        while True:
            # Get a new image.
            frame, status = vs.next_frame()
            if status is False:
                break
            if self.detect_bad_frame(frame):
                continue

            # Undistort input image
            img = cv2.undistort(frame, calibration, distortion)
            cv2.imshow('image', img)
            cv2.waitKey(1)
            # Get the previous pose
            pre_pose = trajectory[-1]
            superpoint_features = self.superpoint_generator(img)
            print("superpoint_features: ", len(superpoint_features.keypoints))
            observed_landmarks = self.landmark_projection(pre_pose)
            print("observed_landmarks: ", len(observed_landmarks.keypoints))
            observations = self.landmark_association(
                superpoint_features, observed_landmarks)
            print("Data association: ", observations)
            current_pose = self.pose_estimate(observations, pre_pose)
            trajectory.append(current_pose)
            plot_trajectory(self.map.landmarks, trajectory)

        plt.ioff()
        plt.show()
        # cv2.destoryAllWindows()
        return trajectory

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Returns:
            superpoint_features - a Features Object
        """
        fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                nms_dist=4,
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=False)
        superpoints, descriptors, _ = fe.run(image)

        return Features(superpoints[:2, ].T.tolist(), descriptors.T.tolist())

    def landmark_projection(self, pose):
        """ Project landmark points in the map to the given camera pose. 
            And filter landmark points outside the view of the current camera pose.
        Parameters:
            pose - gtsam.Point3, the pose of a camera
            self.map - A Landmarks Object
        Returns:
            observed_landmarks - A Landmarks Object
        """
        # Check if the atrium map is empty
        assert self.map.get_length(), "the map is empty"

        observed_landmarks = Landmarks([], [])
        for i, landmark_point in enumerate(self.map.landmarks):
            camera = gtsam.PinholeCameraCal3_S2(pose, self.calibration)
            # feature is gtsam.Point2 object
            landmark_point = Point3(
                landmark_point[0], landmark_point[1], landmark_point[2])
            feature_point = camera.project(landmark_point)
            # Check if the projected feature is within the field of view.
            if (feature_point.x() >= 0 and feature_point.x() < self.width
                    and feature_point.y() >= 0 and feature_point.y() < self.height):
                observed_landmarks.append(self.map.landmarks[i], self.map.descriptors[i], [
                                          feature_point.x(), feature_point.y()])
        return observed_landmarks

    def landmark_association(self, superpoint_features, observed_landmarks):
        """ Associate Superpoint feature points with landmark points by matching all superpoint features with projected features.
        Parameters:
            superpoint_features - A Features Object
            observed_landmarks - A Landmarks Object
        Returns:
            observations - [(Point2(), Point3())]

        """
        if superpoint_features.get_length() == 0 or observed_landmarks.get_length() == 0:
            print("Input data is empty.")
            return [[]]
        if self.l2_threshold < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')

        observations = []
        for i, projected_point in enumerate(observed_landmarks.keypoints):
            nearby_feature_indices = []
            min_score = self.l2_threshold
            # Calculate the pixels distances between current superpoint and all the points in the map
            for j, superpoint in enumerate(superpoint_features.keypoints):
                x_diff = abs(superpoint[0] - projected_point[0])
                y_diff = abs(superpoint[1] - projected_point[1])
                if(x_diff < self.x_distance_thresh and y_diff < self.y_distance_thresh):
                    # if((x_diff*x_diff+y_diff*y_diff)**0.5 < 2):
                    nearby_feature_indices.append(j)
            # print("nearby_feature_indices", nearby_feature_indices)
            if nearby_feature_indices == []:
                continue

            for feature_index in nearby_feature_indices:
                # Compute L2 distance. Easy since vectors are unit normalized.
                dmat = np.dot(
                    np.array(superpoint_features.descriptors[feature_index]), np.array(observed_landmarks.descriptors[i]).T)
                dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
                # Select the minimal L2 distance point
                if dmat < min_score:
                    min_score = dmat
                    key_point = superpoint_features.keypoints[feature_index]
                    landmark = observed_landmarks.landmarks[i]
                    observations.append(
                        (Point2(key_point[0], key_point[1]), Point3(landmark[0], landmark[1], landmark[2])))

        return observations

    def new_landmark_association(self, superpoint_features, observed_landmarks):
        """ Associate Superpoint feature points with landmark points by matching all superpoint features with projected features.
        Parameters:
            superpoint_features - A Features Object
            observed_landmarks - A Landmarks Object
        Returns:
            observations - [(Point2(), Point3())]

        """
        if superpoint_features.get_length() == 0 or observed_landmarks.get_length() == 0:
            print("Input data is empty.")
            return [[]]
        if self.l2_threshold < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')

        observations = []
        src = np.array([], dtype=np.float).reshape(0, 2)
        dst = np.array([], dtype=np.float).reshape(0, 2)
        for i, projected_point in enumerate(observed_landmarks.keypoints):
            nearby_feature_indices = []
            min_score = self.l2_threshold
            # Calculate the pixels distances between current superpoint and all the points in the map
            for j, superpoint in enumerate(superpoint_features.keypoints):
                x_diff = abs(superpoint[0] - projected_point[0])
                y_diff = abs(superpoint[1] - projected_point[1])
                if(x_diff < self.x_distance_thresh and y_diff < self.y_distance_thresh):
                    # if((x_diff*x_diff+y_diff*y_diff)**0.5 < 2):
                    nearby_feature_indices.append(j)
            # print("nearby_feature_indices", nearby_feature_indices)
            if nearby_feature_indices == []:
                continue

            for feature_index in nearby_feature_indices:
                # Compute L2 distance. Easy since vectors are unit normalized.
                dmat = np.dot(
                    np.array(superpoint_features.descriptors[feature_index]), np.array(observed_landmarks.descriptors[i]).T)
                dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
                # Select the minimal L2 distance point
                if dmat < min_score:
                    min_score = dmat
                    key_point = superpoint_features.keypoints[feature_index]
                    landmark = observed_landmarks.landmarks[i]

                    observations.append(
                        (Point2(key_point[0], key_point[1]), Point3(landmark[0], landmark[1], landmark[2])))

                    src = np.vstack((src, int(round(key_point[0])), int(round(key_point[1]))))
                    dst = np.vstack((dst, [observed_landmarks.keypoints[i].x(), observed_landmarks.keypoints[i].y()]))

        bad_essential_matrix, filtered_observations = self.ransac_filter(observations, src, dst)

        if bad_essential_matrix:
            print("Not enough points to generate essential matrix.")
            return []
            
        return observations

    def ransac_filter(self,observations, src, dst, threshold = 1):
        """Use opencv ransac to filter matches."""
        src = np.array([], dtype=np.float).reshape(0, 2)
        dst = np.array([], dtype=np.float).reshape(0, 2)

        # if src.shape[0] < 20:
        #     return True, np.array([])

        src = np.expand_dims(src, axis=1)
        dst = np.expand_dims(dst, axis=1)
        E, mask = cv2.findEssentialMat(
            dst, src, cameraMatrix=self.calibration.matrix(), method=cv2.RANSAC, prob=0.999, threshold=threshold)
        # fundamental_mat, mask = cv2.findFundamentalMat(
        #     src, dst, cv2.FM_RANSAC, 1, 0.99)
        # print("fundamental_mat:\n", fundamental_mat)

        if mask is None:
            return True, np.array([])
        filtered_observations = [observations[i] for i, score in enumerate(mask) if score == 1]

        return False, filtered_observations

    def pose_estimate(self, observations, estimated_pose):
        """ Estimate current pose with matched features through GTSAM and update the trajectory
        Parameters:
            observations - [(Point2(), Point3())]
            pose - gtsam.pose3 Object. The pose at the last state from the trajectory.
        Returns:
            current_pose - gtsam.pose3 Object. The current estimate pose.
        """
        if len(observations) == 0:
            print("No observation")
        # Initialize factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        measurement_noise_sigma = 1.0
        measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurement_noise_sigma)

        # Because the map is known, we use the landmarks from the visible map with nearly zero error as priors.
        point_prior_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.01)
        for i, observation in enumerate(observations):
            print(i)
            # Add projection factors with matched feature points
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                observation[0], measurement_noise, X(0), P(i), self.calibration))
            # Create priors for all observations
            graph.add(gtsam.PriorFactorPoint3(
                P(i), observation[1], point_prior_noise))
            initial_estimate.insert(P(i), observation[1])
        # Create initial estimate for the pose
        # Because the robot moves slowly, we can use the previous pose as an initial estimation of the current pose.
        initial_estimate.insert(X(0), estimated_pose)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        result = optimizer.optimize()
        # Add current pose to the trajectory
        current_pose = result.atPose3(X(0))
        print(current_pose)

        return current_pose
