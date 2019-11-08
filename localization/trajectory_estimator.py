"""Trajectory Estimator"""
# cSpell: disable
# pylint: disable=no-name-in-module,wrong-import-order, no-member,ungrouped-imports, invalid-name
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import gtsam
from mapping.bundle_adjustment.mapping_result_helper import load_map_from_file
from gtsam import Point2, Point3, symbol, Pose3
from localization.features import Features
from localization.observed_landmarks import ObservedLandmarks
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from utilities.plotting import plot_trajectory, plot_map, plot_map_ax, plot_trajectory_ax
from utilities.video_streamer import VideoStreamer
from localization.trajectory_estimator_helper import save_feature_image, save_feature_to_file, save_match_image
import time
from sklearn.neighbors import NearestNeighbors
import math
from sklearn.metrics import pairwise_distances_argmin_min



def X(i):  # pylint: disable=invalid-name
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):  # pylint: disable=invalid-name
    """Create key for landmark j."""
    return symbol(ord('p'), j)


class TrajectoryEstimator():
    """Trajectory Estimator"""

    def __init__(self, initial_pose, directory_name, camera, l2_threshold, distance_thresh, measurement_noise, point_prior_noise, debug_enable=True):
        self.initial_pose = initial_pose
        self._directory_name = directory_name

        def load_map(file_name):
            """Load map data from file."""
            if os.path.isfile(file_name) is False:
                print("\nFile name does not exist, the map will be empty.")
                return ObservedLandmarks([], [])
            landmark_3d_pts, landmark_desc = load_map_from_file(file_name)
            input_map = ObservedLandmarks(landmark_3d_pts, landmark_desc)
            return input_map
        self.map = load_map(directory_name+"/map/map.dat")
        self._camera = camera
        if l2_threshold < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        self.l2_threshold = l2_threshold
        self.x_distance_thresh = distance_thresh[0]
        self.y_distance_thresh = distance_thresh[1]
        self._diagonal_thresh = distance_thresh[0]**2 + distance_thresh[1]**2
        self._debug = debug_enable
        self._measurement_noise = measurement_noise
        self._point_prior_noise = point_prior_noise

    def load_map(self, landmarks):
        """Load map data"""
        self.map = landmarks

    def detect_bad_frame(self, next_frame):
        """Check to see if the next frame is a bad frame."""
        # TODO Shicong: What is a bad frame?
        return False

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Returns:
            superpoint_features - an *Features* Object
        """
        fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                nms_dist=4,
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=True)
        superpoints, descriptors, _ = fe.run(image)

        return Features(superpoints[:2, ].T.tolist(), descriptors.T.tolist())

    def landmark_projection(self, pose):
        """ Project landmark points in the map to the given camera pose.
            And filter landmark points outside the view of the current camera pose.
        Parameters:
            pose - gtsam.Point3, the pose of a camera
            self.map - An *ObservedLandmarks* Object
        Returns:
            observed_landmarks - An *ObservedLandmarks* Object
        """
        # Check if the atrium map is empty
        assert self.map.get_length(), "The map is empty."

        if not self._camera.distort_enable:
            height, width = self._camera.image_size
            calibration = self._camera.calibration
        else:
            height, width = self._camera.undistort_image_size
            calibration = self._camera.undistort_calibration

        observed_landmarks = ObservedLandmarks([], [])
        for i, landmark_point in enumerate(self.map.landmarks):
            simple_camera = gtsam.SimpleCamera(pose, calibration)
            # feature is gtsam.Point2 object
            landmark_point = Point3(
                landmark_point[0], landmark_point[1], landmark_point[2])
            feature_point, cheirality_check = simple_camera.projectSafe(
                landmark_point)
            if cheirality_check is False:
                continue
            # Check if the projected feature is within the field of view.
            if (feature_point.x() >= 0 and feature_point.x() < width
                    and feature_point.y() >= 0 and feature_point.y() < height):
                observed_landmarks.append(self.map.landmarks[i], self.map.descriptors[i], [
                                          feature_point.x(), feature_point.y()])
        return observed_landmarks

    def find_keypoints_within_boundingbox(self, src_keypoint, dst_keypoints):
        """Find points within a bounding box around a given keypoint."""
        dst_keypoints = np.array(dst_keypoints)
        x_diff = dst_keypoints[:, 0] - src_keypoint[0]
        y_diff = dst_keypoints[:, 1] - src_keypoint[1]
        diff = np.multiply(x_diff, x_diff) + np.multiply(y_diff, y_diff)
        nearby_indices = [i for i in range(
            len(dst_keypoints)) if diff[i] < self._diagonal_thresh]
        return nearby_indices

    def find_smallest_l2_distance_keypoint(self, feature_indices, features, landmark, landmark_desc):
        """Find the keypoint with the smallest l2 distance within the bounding box."""
        # Vectorization is slower?
        # descriptors = np.array(features.descriptors)[feature_indices,:]
        # dmat = np.dot(descriptors, np.array([landmark_desc]).T)
        # dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        # min_score = np.amin(dmat)
        # if min_score < self.l2_threshold:
        #     feature_index = np.where(dmat == min_score)
        #     key_point = features.keypoints[int(feature_index[0])]
        #     return Point2(key_point[0], key_point[1]), Point3(landmark[0], landmark[1], landmark[2])

        min_score = self.l2_threshold
        for feature_index in feature_indices:
            # Compute L2 distance. Easy since vectors are unit normalized. This method is from the superpoint pretrain script.
            dmat = np.dot(
                np.array(features.descriptors[feature_index]), np.array(landmark_desc).T)
            dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
            # Select the minimal L2 distance point
            if dmat < min_score:
                min_score = dmat
                key_point = features.keypoints[feature_index]
        if min_score < self.l2_threshold:
            return Point2(key_point[0], key_point[1]), Point3(landmark[0], landmark[1], landmark[2])
        
        # sklearn
        # descriptors = np.array(features.descriptors)[feature_indices,:]
        # landmark_desc = np.array([landmark_desc])

        # feature_index, min_score = pairwise_distances_argmin_min(descriptors, landmark_desc)
        # if min_score[0] < self.l2_threshold:
        #     key_point = features.keypoints[feature_index[0]]
        #     return Point2(key_point[0], key_point[1]), Point3(landmark[0], landmark[1], landmark[2])

    def landmark_association(self, superpoint_features, observed_landmarks):
        """ Associate Superpoint feature points with landmark points by matching all superpoint features with projected features.
        Parameters:
            superpoint_features - An *Features* Object
            observed_landmarks - An *ObservedLandmarks* Object
        Returns:
            observations - [[Point2(), Point3()]]
            match_keypoints - [[x,y]]

        """
        # Associate features with points in the map
        # """
        # If the map contains less noise and more structure, which means if we trust the map.
        # Then we map the extracted feature to the observed landmarks.
        # """"
        

        # """"
        # Test with knn
        # """"
        tic_ba = time.time()
        samples = superpoint_features.keypoints
        neigh = NearestNeighbors(radius=65.0)
        #neigh = NearestNeighbors(60)
        neigh.fit(samples)
        NearestNeighbors(algorithm='kd_tree', leaf_size=30)
        toc_ba = time.time()
        print('KNN spents ', toc_ba-tic_ba, 's')

        def associate_features_to_map_knn(i, projected_point):
            """Associate features to the projected feature points."""
            # Calculate the pixels distances between current superpoint and all the points in the map
            tic_ba = time.time()
            # _, indices = neigh.kneighbors(np.array([projected_point]))
            _, indices = neigh.radius_neighbors(np.array([projected_point]), radius =65)
            toc_ba = time.time()
            print('KNN Find Nearby Indices spents ', toc_ba-tic_ba, 's')
            nearby_indices =indices[0].tolist()
            
            tic_ba = time.time()
            nearby_indices = self.find_keypoints_within_boundingbox(
                projected_point, superpoint_features.keypoints)
            toc_ba = time.time()
            print('Find Nearby Indices spents ', toc_ba-tic_ba, 's')

            # If no matches, continue
            if nearby_indices == []:
                pass
            # If there are more than one feature in the bounding box, return the keypoint with the smallest l2 distance
            # tic_ba = time.time()
            keypoint = self.find_smallest_l2_distance_keypoint(nearby_indices, superpoint_features, observed_landmarks.landmarks[i], observed_landmarks.descriptors[i]), observed_landmarks.keypoints[i]
            # toc_ba = time.time()
            # print('Find keypoint spents ', toc_ba-tic_ba, 's')
            return keypoint


        def associate_features_to_map(i, projected_point):
            """Associate features to the projected feature points."""
            # Calculate the pixels distances between current superpoint and all the points in the map
            nearby_indices = self.find_keypoints_within_boundingbox(
                projected_point, superpoint_features.keypoints)
            # If no matches, continue
            if nearby_indices == []:
                pass
            # If there are more than one feature in the bounding box, return the keypoint with the smalles l2 distance
            return self.find_smallest_l2_distance_keypoint(nearby_indices, superpoint_features, observed_landmarks.landmarks[i], observed_landmarks.descriptors[i]), observed_landmarks.keypoints[i]

        observations = [associate_features_to_map_knn(
            i, projected_point) for i, projected_point in enumerate(observed_landmarks.keypoints)]

        match_keypoints = [observation[1]
                           for observation in observations if observation[0]]
        observations = [observation[0]
                        for observation in observations if observation[0]]

        return observations, match_keypoints

    def DLT_ransac(self, observations):
        """Use 6 points DLT ransac to filter data and generate initial pose estimation.
            Parameters:
                observations - a list, [(Point2(), Point3())]
            Return:
                pose - Pose3
                new_observations - a list, [(Point2(), Point3())]
        """
        # Use opencv to solve PnP ransac
        # https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

        # new estimate pose  = cv2.solvePnPRansac
        # Use the new estimate pose to back project the observed 3D points to select for outlier
        pass

    def BA_pose_estimation(self, observations, estimated_pose):
        """ Estimate current pose with matched features through GTSAM Bundle Adjustment.
        Parameters:
            observations - [[Point2(), Point3()]]
            pose - gtsam.pose3 Object. The estimated pose.
        Returns:
            current_pose - gtsam.pose3 Object. The current calculated pose.
        """
        # Need to consider the situation when the number of observation is not enough.
        assert observations, "The observation is empty."

        # Initialize factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        for i, observation in enumerate(observations):
            # Add projection factors with matched feature points
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                observation[0], self._measurement_noise, X(0), P(i), self._camera.calibration))
            # Create priors for all observated landmark points
            graph.add(gtsam.PriorFactorPoint3(
                P(i), observation[1], self._point_prior_noise))
            initial_estimate.insert(P(i), observation[1])
        # Create initial estimate for the pose
        initial_estimate.insert(X(0), estimated_pose)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        result = optimizer.optimize()
        current_pose = result.atPose3(X(0))
        return current_pose

    def pose_estimator(self, image, frame_count, pre_pose, color_image, save_for_debug=False):
        """The full pipeline to estimates the current pose."""
        # """
        # Superpoint Feature Extraction.
        # """
        tic_ba = time.time()
        superpoint_features = self.superpoint_generator(image)
        toc_ba = time.time()
        print('Superpoint Extraction spents ', toc_ba-tic_ba, 's')
        # Check if to see if features are empty.
        if superpoint_features.get_length() == 0:
            print("No feature extracted.")
            return Pose3(), False
        #
        if save_for_debug:
            print('Number of Extracted Features:{}'.format(
                superpoint_features.get_length()))
            save_feature_to_file(self._directory_name +
                                 "features/", superpoint_features, frame_count)
            save_feature_image(self._directory_name+"feature_images/",
                               superpoint_features, np.copy(color_image), frame_count)

        # """
        # Landmark Backprojection.
        # """
        tic_ba = time.time()
        observed_landmarks = self.landmark_projection(pre_pose)
        toc_ba = time.time()
        print('Landmark Projection spents ', toc_ba-tic_ba, 's')
        # Check if to see if the observed landmarks is empty.
        if observed_landmarks.get_length() == 0:
            print("No projected landmarks.")
            return Pose3(), False
        if save_for_debug:
            print('Number of Back Projected Features:{}'.format(
                observed_landmarks.get_length()))
            save_feature_to_file(
                self._directory_name+"project_features/", observed_landmarks, frame_count)
            save_feature_image(self._directory_name+"project_feature_images/",
                               observed_landmarks, np.copy(color_image), frame_count, color=(255, 0, 0))

        # TODO:save_landmark_to_file()
        # 3d point and poses
        # kp, desc, an image of all feature patch
        # if save_for_debug:

        # """
        # Landmark Association.
        # """
        tic_ba = time.time()
        observations, keypoints = self.landmark_association(
            superpoint_features, observed_landmarks)
        toc_ba = time.time()
        print('Landmark Association spents ', toc_ba-tic_ba, 's')
        # Check if to see if the associate landmarks is empty.
        if not observations:
            print("No projected landmarks.")
            return Pose3(), False
        if save_for_debug:
            print('Number of Matched Features:{}'.format(len(keypoints)))
            save_match_image(self._directory_name+"match_images/",
                             observations, keypoints, np.copy(color_image), frame_count)

        # If number of observations<6 pass
        if len(observations) < 6:
            print("Number of Observations less than 6.")
            return Pose3(), False

        # """
        # 6 Point DLT RANSAC.
        # """
        # TODO:estimated_pose, new_observations = self.DLT_ransac(observations)
        # TODO:Check estimated_pose and prepose

        # """
        # Bundle Adjustment.
        # """
        tic_ba = time.time()
        current_pose = self.BA_pose_estimation(observations, pre_pose)
        toc_ba = time.time()
        print('BA spents ', toc_ba-tic_ba, 's')

        return current_pose, True

    def trajectory_generator(self, input_src, camid, skip, img_glob, start_index):
        """The top level function, use to generate trajectory.
        Input:
            initial_pose - Pose3 Object
            calibration - np.array
            distortion - np.array
            input_src - "camera" if input is Webcam. directory if input is images or video file.
            camid - Webcam id
            skip - number of frames to skip
            img_glob - image suffix(extension), e.g. "*.jpg"
            start_index - the number of frames to jump at the beginning
        Output:
            trajectory - a list of Pose3 objects

        """
        trajectory = [self.initial_pose]
        # This class helps load input images from different sources.
        height, width = self._camera.image_size
        vs = VideoStreamer(input_src, camid, height, width,
                           skip, img_glob, start_index)
        # Display the map
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)
        # plot_map_ax(self.map.landmarks, ax1)
        # plt.show(block=False)

        # Help to index the input image
        frame_count = 0
        while True:
            # Get a new image.
            frame, color_image, status = vs.next_frame()
            if status is False:
                break
            if self.detect_bad_frame(frame):
                continue

            # Get the previous pose
            pre_pose = trajectory[-1]
            current_pose, status = self.pose_estimator(
                frame, frame_count, pre_pose, color_image)
            if not status:
                frame_count += 1
                continue

            # If current pose too different from previous pose, use the previous pose
            distance = np.linalg.norm(current_pose.translation().vector() - pre_pose.translation().vector())
            if distance > 1:
                 current_pose = pre_pose
            trajectory.append(current_pose)

            if self._debug is True:
                ax2.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                plot_trajectory_ax(trajectory, ax1)

                plt.show(block=False)

            frame_count += 1

        # cv2.destoryAllWindows()
        return trajectory

    # def trajectory_generator_distort(self, input_src, camid, skip, img_glob, start_index):
    #     """The top level function, use to generate trajectory.
    #     Input:
    #         initial_pose - Pose3 Object
    #         calibration - np.array
    #         distortion - np.array
    #         input_src - "camera" if input is Webcam. directory if input is images or video file.
    #         camid - Webcam id
    #         skip - number of frames to skip
    #         img_glob - image suffix(extension), e.g. "*.jpg"
    #         start_index - the number of frames to jump at the beginning
    #     Output:
    #         trajectory - a list of Pose3 objects

    #     """
    #     trajectory = [self.initial_pose]
    #     # This class helps load input images from different sources.

    #     height, width = self._camera.raw_image_size
    #     vs = VideoStreamer(input_src, camid, height, width,
    #                        skip, img_glob, start_index)

    #     while True:
    #             # Get a new image.
    #         frame, status = vs.next_frame()
    #         if status is False:
    #             break
    #         if self.detect_bad_frame(frame):
    #             continue

    #         # Get the previous pose
    #         pre_pose = trajectory[-1]

    #         # Undistort input image
    #         # TODO(shicong): Add undistort
    #         img = self._camera.undistort(frame)

    #         if self._debug is True:
    #             cv2.imshow('image', img)
    #             cv2.waitKey(1)

    #         current_pose = self.pose_estimator(img, pre_pose)
    #         trajectory.append(current_pose)
    #         plot_trajectory(self.map.landmarks, trajectory)

    #     plt.ioff()
    #     plt.show()
    #     # cv2.destoryAllWindows()
    #     return trajectory
