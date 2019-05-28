# cSpell: disable=invalid-name
"""
A Mapping Pipeline: feature match info parser, data association, and Bundle Adjustment(gtsam).
"""
# pylint: disable=invalid-name, no-name-in-module, no-member, assignment-from-no-return, line-too-long


import copy
import os
import time
from parser import get_matches, load_features_list

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Point2, Point3, Pose3, Rot3, symbol


def X(i):
    """Create key for pose i."""
    return symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return symbol(ord('p'), j)


def transform_from(T, pose):
    """ Calculate the Euclidean transform of a Pose3
    Parameters:
        pose - Pose3 object
    Returns:
        Pose3 object
    """
    R = T.rotation().compose(pose.rotation())
    translation = T.rotation().rotate(pose.translation()).vector() + \
        T.translation().vector()
    return Pose3(R, Point3(translation))


class ImageFeature(object):
    """
    Store both image and pose information.
        self._descriptors - Nx256 np.array
        self.keypoints - Nx2 np.array
    """

    def __init__(self, keypoints, descriptors):
        # Initialize image information including key points and feature descriptors
        self.keypoints = keypoints
        self._descriptors = descriptors


class ImageMatches(object):
    """
    Store both image and pose information.
        self._descriptors - Nx256 np.array
        self.keypoints - Nx2 np.array
    """

    def __init__(self, key_point_matches):
        # Initialize a dictionary to manage matching information between
        # current image keypoints indices and keypoints in other images
        self.kp_matches = key_point_matches

    def kp_match_idx_list(self, kp_idx, img_idx):
        """
        Use key point idx and an image index to return the matched key point in the img_idx image
        """
        match_list = self.kp_matches.get(kp_idx)
        for match in match_list:
            if match[0] == img_idx:
                return match[1]

        return False


class LandMark(object):
    """
    Store landmark information.
    """

    def __init__(self, point=np.array([0, 0, 0]), seen=0):
        self.point = point
        self.seen = seen


class ImagePoints(object):
    """
    Store both image and pose information.
        self._descriptors - Nx256 np.array
        self.keypoints - Nx2 np.array
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


class ImagePose(object):
    """
    Store both image and pose information.
        self._descriptors - Nx256 np.array
        self.keypoints - Nx2 np.array
    """

    def __init__(self, calibration):
        # Initialize image information including key points and feature descriptors
        self.T = Pose3(Rot3(), Point3())
        self.P = np.dot(calibration, np.hstack(
            (np.identity(3), np.zeros((3, 1)))))


class MappingBackEnd(object):
    """
    Mapping Back End.
    """

    def __init__(self, data_directory, num_images=3, delta_z=1):
        """Construct by reading from a data directory."""
        self.basedir = data_directory
        self.nrimages = num_images

        # Camera calibration
        fov, w, h = 60, 1280, 720
        self._calibration = gtsam.Cal3_S2(fov, w, h)

        # Pose distance along z axis
        # fps = 30
        # velocity = 10  # m/s
        # self.delta_z = velocity / fps
        self.delta_z = delta_z
        self._min_landmark_seen = 3

        def get_image_features(image_index):
            """ Load features
                features - keypoints:A Nx2 list of (x,y). Descriptors: A Nx256 list of desc.
            """
            feat_file = os.path.join(
                self.basedir, "{0:07}.key".format(image_index))
            keypoints, descriptors = load_features_list(feat_file)
            return ImageFeature(keypoints, descriptors)

        self._image_features = [get_image_features(
            image_index) for image_index in range(self.nrimages)]
        self._image_matches = self.create_image_matches()
        self.image_points, self.landmark, self.image_poses = self.data_association()

    def create_image_matches(self):
        """
        Create feature match across all images.
            Input:
                basedir- the base directory that stores all images.
                nrimages- number of images
            Output:
                image_matches: a list of ImageMatches objects.
        """
        def load_matches(basedir, image_1, image_2):
            """ Load matches
                matches - a list of [image_1, keypt_1, image_2, keypt_2]
            """
            matches_file = os.path.join(
                basedir, "match_{0}_{1}.dat".format(image_1, image_2))
            _, matches = get_matches(matches_file)
            return matches

        def kp_matches_handle(img_idx):
            """ Load match data from files and update dictionary.
            Returns - Both img_idx and img_idx+1 matches.
            """
            matches = load_matches(self.basedir, img_idx, img_idx+1)
            kp_matches = {match[1]: [[img_idx+1, match[3]]]
                          for match in matches}
            kp_matches_reverse = {match[3]: [
                [img_idx, match[1]]] for match in matches}
            return [kp_matches, kp_matches_reverse]

        def kp_matches_merge(pair_matches, img_idx):
            """ Marge key point matches dictionary.
            """
            if img_idx == 0:
                return ImageMatches(pair_matches[0][0])

            # Merge dictionaries
            s_matches = pair_matches[img_idx-1][1]
            d_matches = pair_matches[img_idx][0]
            image_matches = {item[0]: d_matches[item[0]] + s_matches[item[0]] if d_matches.get(
                item[0]) is not None else item[1] for item in s_matches.items()}

            return ImageMatches(image_matches)

        pair_matches = [kp_matches_handle(i) for i in range(self.nrimages-1)]
        image_matches = [kp_matches_merge(pair_matches, i)
                         for i in range(self.nrimages-1)]

        return image_matches

    def data_association(self):
        """
        Find feature data association across all images incrementally.
        """
        def initial_estimate(idx, src, dst, input_image_poses):
            """
            Estimate pose and landmark.
            """
            image_poses = copy.copy(input_image_poses)
            E, mask = cv2.findEssentialMat(
                dst, src, cameraMatrix=self._calibration.matrix(), method=cv2.RANSAC, prob=0.9, threshold=3)

            _, local_R, local_t, _ = cv2.recoverPose(
                E, dst, src, cameraMatrix=self._calibration.matrix())
            print("R=", local_R)
            print("t=", local_t)

            T = Pose3(Rot3(local_R), Point3(
                local_t[0], local_t[1], local_t[2]))

            image_poses[idx +
                        1].T = transform_from(T, image_poses[idx].T)
            cur_R = image_poses[idx+1].T.rotation().matrix()
            cur_t = image_poses[idx+1].T.translation().vector()
            cur_t = np.expand_dims(cur_t, axis=1)
            image_poses[idx+1].P = np.dot(self._calibration.matrix(),
                                          np.hstack((cur_R.T, -1*np.dot(cur_R.T, cur_t))))

            points4D = cv2.triangulatePoints(
                projMatr1=image_poses[idx].P, projMatr2=image_poses[idx+1].P, projPoints1=src, projPoints2=dst)
            points4D = points4D / np.tile(points4D[-1, :], (4, 1))
            pt3d = points4D[:3, :].T
            return pt3d, mask, image_poses

        def data_associate(idx, in_image_points, in_landmarks, in_image_poses):
            """
            Find feature data association within idx image.
            """
            # Get keypoints and keypoint indices
            matches = self._image_matches[idx]
            # print(self._image_matches[1].kp_matches)
            features = self._image_features
            kp_idx_matches = [[int(k), int(matches.kp_match_idx_list(k, idx+1))]
                              for k in matches.kp_matches if matches.kp_match_idx_list(k, idx+1) is not None]
            kp_matches = [[features[idx].keypoints[int(
                match[0])], features[idx+1].keypoints[int(match[1])]] for match in kp_idx_matches]
            src = np.expand_dims(np.array(kp_matches)[:, 0], axis=1)
            dst = np.expand_dims(np.array(kp_matches)[:, 1], axis=1)

            if src.shape[0] < 6:
                print(
                    "Warning:Not enough points to generate essential matrix for image_%d and image_%d.", (idx, idx+1))
                return in_image_points, in_landmarks, in_image_poses

            pt3d, mask, image_poses = initial_estimate(
                idx, src, dst, in_image_poses)

            image_points = copy.deepcopy(in_image_points)
            landmarks = copy.deepcopy(in_landmarks)

            # Find good triangulated points
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

        image_poses = [ImagePose(self._calibration.matrix())
                       for i in range(self.nrimages)]
        image_points = [ImagePoints({}) for i in range(self.nrimages)]
        landmarks = [LandMark()]
        for i in range(self.nrimages-1):
            image_points, landmarks, image_poses = data_associate(
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

    def bundle_adjustment(self):
        """
        Bundle Adjustment.
        Input:
            self._image_features
            self._image_points
        Output:
            result-
        """
        MIN_LANDMARK_SEEN = 3  # minimal number of corresponding keypoints to recover a landmarks
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0)  # camera to world rotation
        depth = 20  # Back projection depth

        # Initialize factor Graph
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        # Set pose prior noise
        s = np.radians(60)
        poseNoiseSigmas = np.array([s, s, s, 1, 1, 1])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)

        landmark_id_list = {}
        img_pose_id_list = {}
        for img_idx, features in enumerate(self._image_features):
            for kp_idx in range(len(features.keypoints)):
                if self.image_points[img_idx].kp_3d_exist(kp_idx):
                    landmark_id = self.image_points[img_idx].kp_3d(kp_idx)
                    if self.landmark[landmark_id].seen >= MIN_LANDMARK_SEEN:
                        key_point = Point2(
                            features.keypoints[kp_idx][0], features.keypoints[kp_idx][1])
                        # Filter invalid landmarks
                        if landmark_id_list.get(landmark_id) is None:
                            pose = Pose3(wRc, Point3(
                                0, self.delta_z*img_idx, 2))
                            landmark_3d_point = self.back_projection(
                                key_point, pose, depth)
                            landmark_id_list[landmark_id] = landmark_3d_point
                        # Filter invalid image poses
                        if img_pose_id_list.get(img_idx) is None:
                            img_pose_id_list[img_idx] = True
                        graph.add(gtsam.GenericProjectionFactorCal3_S2(
                            key_point, measurementNoise,
                            X(img_idx), P(landmark_id), self._calibration))
        # def find_correspondences(img_idx, features, kp_idx):
        #     if self.image_points[img_idx].kp_3d_exist(kp_idx):
        #         landmark_id = self.image_points[img_idx].kp_3d(kp_idx)
        #         if self.landmark[landmark_id].seen >= MIN_LANDMARK_SEEN:
        #             key_point = Point2(
        #                 features.keypoints[kp_idx][0], features.keypoints[kp_idx][1])
        #             # Filter invalid landmarks
        #             if landmark_id_list.get(landmark_id) is None:
        #                 pose = Pose3(wRc, Point3(0, self.delta_z*img_idx, 2))
        #                 landmark_3d_point = self.back_projection(
        #                     key_point, pose, depth)
        #                 landmark_id_list[landmark_id] = landmark_3d_point
        #             # Get  image poses
        #             if img_pose_id_list.get(img_idx) is None:
        #                 img_pose_id_list[img_idx] = True
        #             graph.add(gtsam.GenericProjectionFactorCal3_S2(
        #                 key_point, measurementNoise,
        #                 X(img_idx), P(landmark_id), self._calibration))
        #     return landmark_id_list, img_pose_id_list

        # find_correspondences(i, features, k, landmark_id_list, img_pose_id_list) for i, features in enumerate(self._image_features) for k in range(len(features.keypoints))

        # Initial estimate for poses
        for idx in img_pose_id_list:
            pose_i = Pose3(
                wRc, Point3(0, self.delta_z*idx, 2))
            initialEstimate.insert(X(idx), pose_i)
            # Create priors for poses
            if idx in (0, 1):
                graph.add(gtsam.PriorFactorPose3(
                    X(idx), pose_i, posePriorNoise))
            # Create priors for all poses
            # graph.add(gtsam.PriorFactorPose3(X(idx), pose_i, posePriorNoise))

        # Initialize estimation for landmarks
        for idx in landmark_id_list:
            point_j = landmark_id_list.get(idx)
            initialEstimate.insert(P(idx), point_j)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        sfm_result = optimizer.optimize()

        # Marginalization
        marginals = gtsam.Marginals(graph, sfm_result)
        for idx in img_pose_id_list:
            marginals.marginalCovariance(X(idx))
        for idx in landmark_id_list:
            marginals.marginalCovariance(P(idx))

        return sfm_result, img_pose_id_list, landmark_id_list

    def plot_sfm_result(self, result):
        """
        Plot mapping result.
        """
        # Declare an id for the figure
        _fignum = 0
        fig = plt.figure(_fignum)
        axes = fig.gca(projection='3d')
        plt.cla()
        # Plot points
        gtsam_plot.plot_3d_points(_fignum, result, 'rx')
        # Plot cameras
        i = 0
        while result.exists(X(i)):
            pose_i = result.atPose3(X(i))
            gtsam_plot.plot_pose3(_fignum, pose_i, 1)
            i += 1
        # Draw
        axes.set_xlim3d(-20, 20)
        axes.set_ylim3d(-20, 20)
        axes.set_zlim3d(-20, 20)
        plt.legend()
        plt.show()


def run():
    """Execution."""
    back_end = MappingBackEnd('feature_matcher/sim_match_data/')
    # print(back_end._feature_matches[1].kp_matches)
    back_end.data_association()
    tic_ba = time.time()
    sfm_result, _, _ = back_end.bundle_adjustment()
    toc_ba = time.time()
    print('BA spents ', toc_ba-tic_ba, 's')
    back_end.plot_sfm_result(sfm_result)


if __name__ == "__main__":
    run()
