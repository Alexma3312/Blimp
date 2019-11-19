"""Trajectory Estimator"""
# cSpell: disable
# pylint: disable=no-name-in-module,wrong-import-order, no-member,ungrouped-imports, invalid-name
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import gtsam
from mapping.bundle_adjustment.mapping_result_helper import load_map_from_file
from gtsam import Point2, Point3, symbol, Pose3, Rot3
from localization.features import Features
from localization.observed_landmarks import ObservedLandmarks
from localization.landmark_map import LandmarkMap
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from utilities.plotting import plot_trajectory, plot_map, plot_map_ax, plot_trajectory_ax
from utilities.video_streamer import VideoStreamer
from localization.trajectory_estimator_helper import save_feature_image, save_feature_to_file, save_match_image
import time
from sklearn.neighbors import NearestNeighbors
import math
from sklearn.metrics import pairwise_distances_argmin_min
import scipy
from line_profiler import LineProfiler
import atexit
profile = LineProfiler()
atexit.register(profile.print_stats)


def X(i):  # pylint: disable=invalid-name
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):  # pylint: disable=invalid-name
    """Create key for landmark j."""
    return symbol(ord('p'), j)


class TrajectoryEstimator():
    """Estimate current pose based on input frame and the previous pose.\n    
    Parameters:\n
        initial pose:
        directory name:
        map:
        camera:
        l2 threshold:
        distance thresh:
        debug:
        noise models:
    """

    def __init__(self, initial_pose, directory_name, camera, l2_threshold, distance_thresh, noise_models, visualize_enable=True, debug_enable=False):
        self.initial_pose = initial_pose
        self._directory_name = directory_name

        def load_map(file_name):
            """Load map data from file."""
            if os.path.isfile(file_name) is False:
                print("\nFile name does not exist, the map will be empty.")
                return LandmarkMap(np.array([]), np.array([]))
            landmark_3d_pts, landmark_desc = load_map_from_file(file_name)
            input_map = LandmarkMap(
                np.array(landmark_3d_pts), np.array(landmark_desc))
            return input_map
        self.map = load_map(directory_name+"/map/map.dat")
        self._camera = camera
        if l2_threshold < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        self.l2_threshold = l2_threshold
        self.x_distance_thresh = distance_thresh[0]
        self.y_distance_thresh = distance_thresh[1]
        self._diagonal_thresh = distance_thresh[0]**2 + distance_thresh[1]**2
        self._visualize = visualize_enable
        self._debug = debug_enable
        self._measurement_noise = noise_models[0]
        self._point_prior_noise = noise_models[1]
        self._pose_translation_prior_noise = noise_models[2]

    def load_map(self, landmarks):
        """Load map data
        Member Dependencies:
        """
        self.map = landmarks

    def detect_bad_frame(self, next_frame):
        """Check to see if the next frame is a bad frame."""
        # TODO Shicong: What is a bad frame?
        return False

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Arguments:
            image: array
        Returns:
            superpoint_features: an *Features* Object
        """
        fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                nms_dist=4,
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=False)
        superpoints, descriptors, _ = fe.run(image)

        return Features(superpoints[:2, ].T, descriptors.T)

    def landmark_projection(self, pose):
        """ Project landmark points in the map to the given camera pose.
            And filter landmark points outside the view of the current camera pose.\n
        Parameters:\n
            pose - gtsam.Point3, the pose of a camera
        Member Dependencies:\n
            map - An ObservedLandmarks Object
            camera - An Camera Object
        Returns:\n
            observed landmarks - An ObservedLandmarks Object
        """
        # Check if the atrium map is empty
        assert self.map.get_length(), "The map is empty."

        if not self._camera.distort_enable:
            height, width = self._camera.image_size
            calibration = self._camera.calibration
        else:
            height, width = self._camera.undistort_image_size
            calibration = self._camera.undistort_calibration

        observed_landmarks = ObservedLandmarks()
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
                observed_landmarks.append(self.map.landmark(i), self.map.descriptor(i), np.array([
                                          [feature_point.x(), feature_point.y()]]))
        return observed_landmarks

    def find_keypoints_within_boundingbox(self, src_keypoint, dst_keypoints):
        """Find points within a bounding box around a given keypoint."""
        x_diff = dst_keypoints[:, 0] - src_keypoint[0]
        y_diff = dst_keypoints[:, 1] - src_keypoint[1]
        diff = np.multiply(x_diff, x_diff) + np.multiply(y_diff, y_diff)
        nearby_indices = [i for i, score in enumerate(
            diff) if score < self._diagonal_thresh]
        return nearby_indices

    # @profile
    def find_smallest_l2_distance_keypoint(self, feature_indices, features, landmark, landmark_desc):
        """Find the keypoint with the smallest l2 distance within the bounding box.
        Member Variables:
            l2 threshold: l2 norm threshold
        Parameters:
            feature_indices: array
            features: a Features object
            landmark: (2,) array
            landmark_desc: (256,) array
        Returns:
            Point2, Point3
        """
        # L2 norm for normalized vectors
        dmat = scipy.linalg.blas.dgemm(alpha=1., a=features.descriptors[feature_indices.tolist(
        )], b=np.array([landmark_desc]), trans_b=True)
        # dmat = np.dot(features.descriptors, landmark_desc)
        dmat.clip(-1, 1, out=dmat)
        dmat = np.sqrt(2-2*dmat)
        # find the minimal score
        # TODO:shicong, add a ratio test to filter bad points
        min_score = np.amin(dmat)
        if min_score < self.l2_threshold:
            feature_index = np.where(dmat == min_score)
            keypoint_index = feature_indices[int(feature_index[0][0])]
            key_point = features.keypoint(keypoint_index)
            return Point2(key_point[0], key_point[1]), Point3(landmark[0], landmark[1], landmark[2])

    # @profile
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
        # """"
        # If the map contains less noise and more structure, which means if we trust the map.
        # Then we map the extracted feature to the observed landmarks.
        # """"

        # """"
        # Initialize KNN
        # """"
        neigh = NearestNeighbors(radius=self.x_distance_thresh)
        neigh.fit(superpoint_features.keypoints, observed_landmarks.keypoints)
        NearestNeighbors(algorithm='auto', leaf_size=30)

        # @profile
        def associate_features_to_map_knn(i, projected_point):
            """Associate features to the projected feature points."""
            # Calculate the pixels distances between current superpoint and all the points in the map
            indices = neigh.radius_neighbors(
                projected_point.reshape(1, 2), radius=self.x_distance_thresh, return_distance=False)
            # If no matches, continue
            if indices[0].shape[0] == 0:
                return None
            # If there are more than one feature in the bounding box, return the keypoint with the smallest l2 distance
            return self.find_smallest_l2_distance_keypoint(indices[0], superpoint_features, observed_landmarks.landmark(i), observed_landmarks.descriptor(i))

        observations = [associate_features_to_map_knn(
            i, projected_point) for i, projected_point in enumerate(observed_landmarks.keypoints)]

        return list(filter(None, observations))

    def pnp_ransac(self, observations):
        """Use 6 points DLT ransac to filter data and generate initial pose estimation.
            Parameters:
                observations - a list, [[Point2(), Point3()]]
            Return:
                pose - Pose3
                new_observations - a list, [(Point2(), Point3())]
        """
        # Use opencv to solve PnP ransac
        # https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        image_points = np.empty((2,))
        object_points = np.empty((3,))
        for observation in observations:
            image_points = np.vstack(
                (image_points, [observation[0].x(), observation[0].y()]))
            object_points = np.vstack(
                (object_points, [observation[1].x(), observation[1].y(), observation[1].z()]))

        #https://stackoverflow.com/questions/35650105/what-are-python-constants-for-cv2-solvepnp-method-flag
        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            object_points, image_points, self._camera.calibration.matrix(), None, None, None, False, cv2.SOLVEPNP_P3P)
        rotation, _ = cv2.Rodrigues(rvecs)
        rotation = rotation.T
        translation = np.dot(rotation.T, -tvecs)

        if inliers is None or inliers.shape[0]<0.5*len(observations):
            return [], None, None

        # Filter observations
        return [observations[int(index)-1] for index in inliers], rotation, translation

    def BA_pose_estimation(self, observations, previous_pose):
        """ Estimate current pose with matched features through GTSAM Bundle Adjustment.
        Parameters:
            observations - [[Point2(), Point3()]]
            pose - gtsam.pose3 Object. The estimated pose.
        Returns:
            current_pose - gtsam.pose3 Object. The current calculated pose.
        """
        # observations, R, t = self.pnp_ransac(observations)
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
        # initial_estimate.insert(X(0), Pose3(
        #     Rot3(R.reshape(3, 3)), Point3(t.reshape(3,))))
        initial_estimate.insert(X(0),previous_pose)
        # TODO: Shicong, Add a prior factor by using the previous pose
        graph.add(gtsam.PoseTranslationPrior3D(
                X(0), previous_pose, self._pose_translation_prior_noise))

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        result = optimizer.optimize()
        current_pose = result.atPose3(X(0))
        return current_pose

    def pose_estimator(self, image, frame_count, pre_pose, color_image):
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
        if self._debug:
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
        if self._debug:
            print('Number of Back Projected Features:{}'.format(
                observed_landmarks.get_length()))
            save_feature_to_file(
                self._directory_name+"project_features/", observed_landmarks, frame_count)
            save_feature_image(self._directory_name+"project_feature_images/",
                               observed_landmarks, np.copy(color_image), frame_count, color=(255, 0, 0))

        # TODO:save_landmark_to_file()
        # 3d point and poses
        # kp, desc, an image of all feature patch
        # if self._debug:

        # """
        # Landmark Association.
        # """
        tic_ba = time.time()
        observations = self.landmark_association(
            superpoint_features, observed_landmarks)
        toc_ba = time.time()
        print('Landmark Association spents ', toc_ba-tic_ba, 's')
        # Check if to see if the associate landmarks is empty.
        if not observations:
            print("No projected landmarks.")
            return Pose3(), False
        if self._debug:
            print('Number of Matched Features:{}'.format(len(observations)))

            def get_keypoints(observation):
                landmark = np.array(
                    [observation[1].x(), observation[1].y(), observation[1].z()])
                index = np.where(observed_landmarks.landmarks == landmark)
                return observed_landmarks.keypoint(int(index[0][0]))
            keypoints = [get_keypoints(observation)
                         for observation in observations]
            save_match_image(self._directory_name+"match_images/",
                             observations, keypoints, np.copy(color_image), frame_count)

        # If number of observations less than 12 pass
        if len(observations) < 12:
            print("Number of Observations less than 12.")
            return Pose3(), False

        # """
        # 6 Point DLT RANSAC.
        # """
        # TODO:estimated_pose, new_observations = self.DLT_ransac(observations)
        # TODO:Check estimated_pose and prepose
        observations, R, t = self.pnp_ransac(observations)

        if len(observations) < 6:
            print("Number of Observations less than 6.")
            return Pose3(), False

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
            # distance = np.linalg.norm(current_pose.translation(
            # ).vector() - pre_pose.translation().vector())
            # if distance > 1:
            #     current_pose = pre_pose
            trajectory.append(current_pose)

            if self._visualize is True:
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
