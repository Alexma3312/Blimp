"""Trajectory Estimator"""
import atexit
import math
# cSpell: disable
# pylint: disable=no-name-in-module,wrong-import-order, no-member,ungrouped-imports, invalid-name
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from line_profiler import LineProfiler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

import gtsam
from gtsam import Point2, Point3, Pose3, Rot3, symbol
from localization.features import Features
from localization.landmark_map import LandmarkMap
from localization.observed_landmarks import ObservedLandmarks
from localization.trajectory_estimator_helper import (get_keypoints,
                                                      save_feature_image,
                                                      save_feature_to_file,
                                                      save_match_image)
from mapping.bundle_adjustment.mapping_result_helper import load_map_from_file
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from utilities.plotting import (plot_map, plot_map_ax, plot_trajectory,
                                plot_trajectory_ax)
from utilities.video_streamer import VideoStreamer

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
        self._fe = fe = SuperPointFrontend(weights_path="SuperPointPretrainedNetwork/superpoint_v1.pth",
                                           nms_dist=16,
                                           conf_thresh=0.015,
                                           nn_thresh=0.7,
                                           cuda=True)

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
        superpoints, descriptors, _ = self._fe.run(image)

        return Features(superpoints[:2, ].T, descriptors.T)

    def landmark_projection(self, pose):
        """ Project landmark points in the map to the given camera pose.
        #         And filter landmark points outside the view of the current camera pose.\n
        #     Parameters:\n
        #         pose - gtsam.Point3, the pose of a camera
        #     Member Dependencies:\n
        #         map - An ObservedLandmarks Object
        #         camera - An Camera Object
        #     Returns:\n
        #         observed landmarks - An ObservedLandmarks Object
        #     """
        # Check if the atrium map is empty
        assert self.map.get_length(), "The map is empty."

        height, width = self._camera.image_size
        calibration = self._camera.calibration.matrix()
        cPw = pose.inverse().matrix()[:3, :]
        _map = self.map.landmarks.T
        homogenous_map = np.pad(
            _map, [(0, 1), (0, 0)], mode='constant', constant_values=1)
        homogenous_keypoints = calibration @ cPw @ homogenous_map

        keypoints = homogenous_keypoints[:2, :]/homogenous_keypoints[2, :]
        check = np.where((keypoints[0, :] >= 0) & (keypoints[0, :] < width) & (
            keypoints[1, :] >= 0) & (keypoints[1, :] < height) & (homogenous_keypoints[2, :] > 0))

        observed_landmarks = ObservedLandmarks()
        observed_landmarks.keypoints = keypoints.T[check]
        observed_landmarks.landmarks = self.map.landmarks[check]
        observed_landmarks.descriptors = self.map.descriptors[check]

        return observed_landmarks

    def find_keypoints_within_boundingbox(self, src_keypoint, dst_keypoints):
        """Find points within a bounding box around a given keypoint."""
        x_diff = dst_keypoints[:, 0] - src_keypoint[0]
        y_diff = dst_keypoints[:, 1] - src_keypoint[1]
        diff = np.multiply(x_diff, x_diff) + np.multiply(y_diff, y_diff)
        nearby_indices = [i for i, score in enumerate(
            diff) if score < self._diagonal_thresh]
        return nearby_indices

    def find_smallest_l2_distance_keypoint(self, landmark_indices, landmarks, key_point, feature_desc):
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
        dmat = scipy.linalg.blas.dgemm(alpha=1., a=landmarks.descriptors[landmark_indices.tolist(
        )], b=np.array([feature_desc]), trans_b=True)
        # dmat = np.dot(features.descriptors, landmark_desc)
        dmat.clip(-1, 1, out=dmat)
        dmat = np.sqrt(2-2*dmat)
        # find the minimal score
        # TODO:shicong, add a ratio test to filter bad points
        min_score = np.amin(dmat)
        # second_min_score = np.amin(dmat[dmat != np.amin(dmat)])
        # if min_score < 0.7*second_min_score:
        #     print(key_point[0], key_point[1])
        if min_score < self.l2_threshold:
            desc_index = np.where(dmat == min_score)
            landmark_index = landmark_indices[int(desc_index[0][0])]
            landmark = landmarks.landmark(landmark_index)
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
        # neigh.fit(superpoint_features.keypoints, observed_landmarks.keypoints)
        neigh.fit(observed_landmarks.keypoints, superpoint_features.keypoints)
        NearestNeighbors(algorithm='auto', leaf_size=30)

        def associate_features_to_map_knn(i, superpoint):
            """Associate features to the projected feature points."""
            # Calculate the pixels distances between current superpoint and all the points in the map
            indices = neigh.radius_neighbors(
                superpoint.reshape(1, 2), radius=self.x_distance_thresh, return_distance=False)
            # If no matches, continue
            if indices[0].shape[0] == 0:
                return None
            # If there are more than one feature in the bounding box, return the keypoint with the smallest l2 distance
            return self.find_smallest_l2_distance_keypoint(indices[0], observed_landmarks, superpoint_features.keypoint(i), superpoint_features.descriptor(i))

        observations = [associate_features_to_map_knn(
            i, superpoint) for i, superpoint in enumerate(superpoint_features.keypoints)]

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

        # https://stackoverflow.com/questions/35650105/what-are-python-constants-for-cv2-solvepnp-method-flag
        # retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(
        #     object_points, image_points, self._camera.calibration.matrix(), None, None, None, False, cv2.SOLVEPNP_DLS)
        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            object_points, image_points, self._camera.calibration.matrix(), None, None, None, False, cv2.SOLVEPNP_P3P)
        rotation, _ = cv2.Rodrigues(rvecs)
        rotation = rotation.T
        translation = -np.dot(rotation.T, tvecs)

        if inliers is None or inliers.shape[0] < 0.5*len(observations):
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
        initial_estimate.insert(X(0), previous_pose)
        # TODO: Shicong, Add a prior factor by using the previous pose
        graph.add(gtsam.PoseTranslationPrior3D(
            X(0), previous_pose, self._pose_translation_prior_noise))

        # Optimization
        # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        params = gtsam.GaussNewtonParams()
        params.setLinearSolverType("MULTIFRONTAL_QR")
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        current_pose = result.atPose3(X(0))
        return current_pose

    def pose_estimator(self, image, frame_count, pre_pose, color_image):
        """The full pipeline to estimates the current pose."""
        
        #############################################################################
        # 0 Create debug folder
        #############################################################################
        # if self._debug and not os.path.exists(self._directory_name+"debug/"):
        #     os.mkdir(self._directory_name+"debug/")
        table = [frame_count]
        total_time = 0.0
        
        #############################################################################
        # 1 Image undistort.
        #############################################################################
        tic_ba = time.time()
        image = cv2.undistort(image, self._camera.calibration.matrix(), self._camera.distortion)
        toc_ba = time.time()
        print('Undistort spents ', toc_ba-tic_ba, 's')
        if self._debug:
            color_image = cv2.undistort(color_image, self._camera.calibration.matrix(), self._camera.distortion)
            # if not os.path.exists(self._directory_name+"debug/undistort_images/"):
            #     os.mkdir(self._directory_name+"debug/undistort_images/")
            # output_path = self._directory_name+'debug/undistort_images/frame_%d' % frame_count+'.jpg'
            # cv2.imwrite(output_path, image*255)
            table.append(toc_ba-tic_ba)
            total_time+=toc_ba-tic_ba

        #############################################################################
        # 2 Superpoint Feature Extraction.
        #############################################################################
        tic_ba = time.time()
        superpoint_features = self.superpoint_generator(image)
        toc_ba = time.time()
        print('Superpoint Extraction spents ', toc_ba-tic_ba, 's')
        # Check if to see if features are empty.
        if superpoint_features.get_length() == 0:
            print("No feature extracted.")
            return Pose3(), False
        if self._debug:
            print('Number of Extracted Features:{}'.format(
                superpoint_features.get_length()))
            # save_feature_to_file(self._directory_name +
            #                      "debug/features/", superpoint_features, frame_count)
            save_feature_image(self._directory_name+"debug/feature_images/",
                               superpoint_features, np.copy(color_image), frame_count)
            table.append(toc_ba-tic_ba)
            table.append(superpoint_features.get_length())
            total_time+=toc_ba-tic_ba

        #############################################################################
        # 3 Landmark Projection.
        #############################################################################
        tic_ba = time.time()
        observed_landmarks = self.landmark_projection(pre_pose)
        toc_ba = time.time()
        print('Landmark Projection spents ', toc_ba-tic_ba, 's')
        # Check if to see if the observed landmarks is empty.
        if observed_landmarks.get_length() == 0:
            print("No projected landmarks.")
            return Pose3(), False
        if self._debug:
            print('Number of Projected Features:{}'.format(
                observed_landmarks.get_length()))
            # save_feature_to_file(
            #     self._directory_name+"debug/project_features/", observed_landmarks, frame_count)
            save_feature_image(self._directory_name+"debug/project_feature_images/",
                               observed_landmarks, np.copy(color_image), frame_count, color=(255, 0, 0))
            table.append(toc_ba-tic_ba)
            table.append(observed_landmarks.get_length())
            total_time+=toc_ba-tic_ba

        #############################################################################
        # 4 Landmark Association.
        ##############################################################################
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

            keypoints = [get_keypoints(observation, observed_landmarks)
                         for observation in observations]
            save_match_image(self._directory_name+"debug/match_images/",
                             observations, keypoints, np.copy(color_image), frame_count)
            table.append(toc_ba-tic_ba)
            table.append(len(observations))
            total_time+=toc_ba-tic_ba

        #############################################################################
        # 5 Point DLT RANSAC.
        #############################################################################
        # # If number of observations less than 12 pass
        # if len(observations) < 12:
        #     print("Number of Observations less than 12.")
        #     return Pose3(), False

        # tic_ba = time.time()
        # observations, R, t = self.pnp_ransac(observations)
        # toc_ba = time.time()
        # print('PnP spents ', toc_ba-tic_ba, 's')

        # # # Check if to see if the associate landmarks is empty.
        # if self._debug:
        #     print('Number of Filter Matched Features:{}'.format(len(observations)))

        #     keypoints = [get_keypoints(observation, observed_landmarks)
        #                  for observation in observations]
        #     save_match_image(self._directory_name+"debug/filter_match_images/",
        #                      observations, keypoints, np.copy(color_image), frame_count)
        #     # table.append(toc_ba-tic_ba)
        #     # table.append(len(observations))
        #     # total_time+=toc_ba-tic_ba


        ##############################################################################
        # 6 Bundle Adjustment.
        ##############################################################################
        if len(observations) < 3:
            print("Number of Observations less than 3.")
            return Pose3(), False

        tic_ba = time.time()
        current_pose = self.BA_pose_estimation(observations, pre_pose)
        toc_ba = time.time()
        print('BA spents ', toc_ba-tic_ba, 's')

        if self._debug:
            print('Number of Matched Features:{}'.format(len(observations)))
            observed_landmarks = self.landmark_projection(current_pose)

            # Rearrange the keypoints list to have it match with the landmarks
            keypoints = [get_keypoints(observation, observed_landmarks)
                         for observation in observations]
            # project_keypoints = [kp for kp in observed_landmarks.keypoints]
            save_match_image(self._directory_name+"debug/final_project_images/",
                             observations, keypoints, np.copy(color_image), frame_count, draw_line=True)
            table.append(toc_ba-tic_ba)
            total_time+=toc_ba-tic_ba
            table.append(total_time)

        ##############################################################################
        # 7 Save Debug table.
        ##############################################################################
        if self._debug:
            with open(self._directory_name+"debug/table.dat", "a") as myfile:
                np.savetxt(myfile, [table], fmt="%.5f")

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
        # Display the trajectory
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        
        # ax2 = fig.add_subplot(1, 2, 2) # diplay the current detected image
        # plot_map_ax(self.map.landmarks, ax1) # display the map
        plt.show(block=False)

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

            trajectory.append(current_pose)
            if self._debug:
                r = current_pose.rotation().matrix()
                t = current_pose.translation().vector()
                with open(self._directory_name+"debug/poses.dat", "a") as myfile:
                    myfile.write(str(frame_count)+" ")
                    np.savetxt(myfile, [[t[0], t[1], t[2], r[0][0], r[0][1], r[0]
                                    [2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]]])

            if self._visualize is True:
                # ax2.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                plot_trajectory_ax(trajectory[-5:], ax1)

                plt.show(block=False)

            frame_count += 1

        # cv2.destoryAllWindows()
        return trajectory
