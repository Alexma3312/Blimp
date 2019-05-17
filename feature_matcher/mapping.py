"""
A Mapping Pipeline: feature match info parser, data association, and Bundle Adjustment(gtsam).
"""
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import gtsam
import gtsam.utils.plot as gtsam_plot
from feature_matcher.parser import get_matches, load_features
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


class ImagePose(object):
    """
    Store both image and pose information.
        self.desc - Nx256 np.array
        self.kp - Nx2 np.array
    """

    def __init__(self, calibration):
        # Initialize image information including key points and feature descriptors
        self.desc = np.array([])
        self.kp = np.array([])
        # Initialize Pose information
        self.R = Rot3()
        self.t = Point3()
        self.T = Pose3(self.R, self.t)
        self.P = np.dot(calibration, np.hstack(
            (np.identity(3), np.zeros((3, 1)))))
        # Initialize a dictionary to manage matching information between current frame keypoints indices and keypoints in other frames
        self.kp_matches = {}
        # Initialize a dictionary to manage matching information between keypoints indices sand their corresponding landmarks indices
        self.kp_landmark = {}

    def kp_match_idx(self, kp_idx, img_idx):
        # Use key point idx and an image index to return the matched key point in the img_idx image
        return self.kp_matches.get(kp_idx).get(img_idx)

    def kp_match_exist(self, kp_idx, img_idx):
        # Check if kp_idx keypoint has matched keypoints in the img index image
        if self.kp_matches.get(kp_idx) is None:
            return False
        else:
            # Return True if img_idx is in self.kp_matches[kp_idx] else False
            return self.kp_matches.get(kp_idx).get(img_idx) is not None

    def kp_3d(self, kp_idx):
        # Use key point idx to return the corresponding landmark index
        return self.kp_landmark.get(kp_idx)

    def kp_3d_exist(self, kp_idx):
        # Check if there are corresponding landmark for kp_idx keypoint
        return self.kp_landmark.get(kp_idx) is not None


class LandMark(object):
    """
    Store landmark information.
    """

    def __init__(self, point=np.array([0, 0, 0]), seen=0):
        self.point = point
        self.seen = seen


class MappingFrontEnd(object):
    def __init__(self, data_directory='feature_matcher/sim_match_data/', num_frames=3):
        self.basedir = data_directory
        self.nrframes = num_frames

        fov,  w, h = 75, 1280, 720
        self.cal = gtsam.Cal3_S2(fov, w, h)
        self.calibration = self.cal.matrix()

        fps = 30
        velocity = 10  # m/s
        self.delta_z = velocity / fps

        self.img_pose = []
        self.landmark = []

    def load_features(self, frame_index):
        """ Load features
            features - A Nx2 np.array of (x,y), A Nx256 np.array of desc 
        """
        feat_file = os.path.join(
            self.basedir, "{0:07}.key".format(frame_index))
        features = load_features(feat_file)
        return features

    def load_matches(self, frame_1, frame_2):
        """ Load matches
            matches - a list of [frame_1, keypt_1, frame_2, keypt_2]
        """
        matches_file = os.path.join(
            self.basedir, "match_{0}_{1}.dat".format(frame_1, frame_2))
        _, matches = get_matches(matches_file)
        return matches

    def get_all_image_features(self):
        """
        Transfer feature extraction information into ImagePose object
        """
        for idx in range(self.nrframes):
            image_pose = ImagePose(self.calibration)
            image_pose.kp, image_pose.desc = self.load_features(idx)
            self.img_pose.append(image_pose)

    def get_feature_matches(self):
        """
        Transfer feature match information into ImagePose object
        """
        # Iterate through all images and add frame i and frame i+1 matching data
        for i in range(0, len(self.img_pose)-1):
            matches = self.load_matches(i, i+1)
            good_match_count = 0
            for match in matches:
                # Update kp_matches dictionary in both frame i and frame i+1
                if self.img_pose[i].kp_matches.get(match[1]) is None:
                    self.img_pose[i].kp_matches[match[1]] = {}
                self.img_pose[i].kp_matches[match[1]][i+1] = match[3]
                if self.img_pose[i+1].kp_matches.get(match[3]) is None:
                    self.img_pose[i+1].kp_matches[match[3]] = {}
                self.img_pose[i+1].kp_matches[match[3]][i] = match[1]

                good_match_count += 1
            print("Feature matching ", i, " ", i+1, " ==> ", len(matches))

    def initial_estimation(self):
        """
        Find feature data association across all images incrementally.  
        Currently is data association but it can be developed into pose and landmark initial estimation.
        """
        for i in range(len(self.img_pose)-1):

            src = np.array([], dtype=np.float).reshape(0, 2)
            dst = np.array([], dtype=np.float).reshape(0, 2)
            kp_src_idx = []
            kp_dst_idx = []

            for k in self.img_pose[i].kp_matches:
                if self.img_pose[i].kp_match_exist(k, i+1):
                    k = int(k)

                    match_idx = self.img_pose[i].kp_match_idx(k, i+1)
                    match_idx = int(match_idx)

                    src = np.vstack((src, self.img_pose[i].kp[k]))
                    dst = np.vstack((dst, self.img_pose[i+1].kp[match_idx]))
                    kp_dst_idx.append(match_idx)
                    kp_src_idx.append(k)

            if src.shape[0] < 6:
                print("Not enough points to generate essential matrix for image_",
                      i, " and image_", i+1)
                continue

            src = np.expand_dims(src, axis=1)
            dst = np.expand_dims(dst, axis=1)
            # E, mask = cv2.findEssentialMat(dst,src,cameraMatrix = self.calibration,method =cv2.LMEDS,prob=0.999)
            E, mask = cv2.findEssentialMat(
                dst, src, cameraMatrix=self.calibration, method=cv2.RANSAC, prob=0.999, threshold=1)

            _, local_R, local_t, mask = cv2.recoverPose(
                E, dst, src, cameraMatrix=self.calibration, mask=mask)
            print(local_R)
            print(local_t)

            T = Pose3(Rot3(local_R), Point3(
                local_t[0], local_t[1], local_t[2]))

            self.img_pose[i+1].T = transform_from(T, self.img_pose[i].T)
            cur_R = self.img_pose[i+1].T.rotation().matrix()
            cur_t = self.img_pose[i+1].T.translation().vector()
            cur_t = np.expand_dims(cur_t, axis=1)
            self.img_pose[i+1].P = np.dot(self.calibration,
                                          np.hstack((cur_R.T, -1*np.dot(cur_R.T, cur_t))))

            points4D = cv2.triangulatePoints(
                projMatr1=self.img_pose[i].P, projMatr2=self.img_pose[i+1].P, projPoints1=src, projPoints2=dst)
            points4D = points4D / np.tile(points4D[-1, :], (4, 1))
            pt3d = points4D[:3, :].T

            # Find good triangulated points
            for j, k in enumerate(kp_src_idx):
                # if(mask[j]):
                match_idx = self.img_pose[i].kp_match_idx(k, i+1)

                if (self.img_pose[i].kp_3d_exist(k)):
                    self.img_pose[i +
                                  1].kp_landmark[match_idx] = self.img_pose[i].kp_3d(k)
                    self.landmark[self.img_pose[i].kp_3d(k)].point += pt3d[j]
                    self.landmark[self.img_pose[i +
                                                1].kp_3d(match_idx)].seen += 1

                else:
                    new_landmark = LandMark(pt3d[j], 2)
                    self.landmark.append(new_landmark)

                    self.img_pose[i].kp_landmark[k] = len(self.landmark) - 1
                    self.img_pose[i +
                                  1].kp_landmark[match_idx] = len(self.landmark) - 1

        for j in range(len(self.landmark)):
            if(self.landmark[j].seen >= 3):
                self.landmark[j].point /= (self.landmark[j].seen - 1)

    def back_projection(self, key_point=Point2(), pose=Pose3(), depth=20):
        # Normalize input key_point
        pn = self.cal.calibrate(key_point)
        # Transfer normalized key_point into homogeneous coordinate and scale with depth
        ph = Point3(depth*pn.x(), depth*pn.y(), depth)
        # Transfer the point into the world coordinate
        return pose.transform_from(ph)

    def bundle_adjustment(self):
        
        MIN_LANDMARK_SEEN = 3 # minimal number of corresponding keypoints to recover a landmarks 
        wRc = Rot3(1, 0, 0, 0, 0, 1, 0, -1, 0) # camera to world rotation
        depth = 20 # Back projection depth

        # Initialize factor Graph
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)

        landmark_id_list = {}
        img_pose_id_list = {}
        for i, img_pose in enumerate(self.img_pose):
            for k in range(len(img_pose.kp)):
                if img_pose.kp_3d_exist(k):
                    landmark_id = img_pose.kp_3d(k)
                    if(self.landmark[landmark_id].seen >= MIN_LANDMARK_SEEN):
                        key_point = Point2(
                            img_pose.kp[k][0], img_pose.kp[k][1])
                        # Filter valid landmarks
                        if landmark_id_list.get(landmark_id) == None:
                            pose = Pose3(wRc, Point3(0, self.delta_z*i, 2))
                            landmark_point = self.back_projection(
                                key_point, pose, depth)
                            landmark_id_list[landmark_id] = landmark_point
                        # Filter valid image poses
                        if img_pose_id_list.get(i) == None:
                            img_pose_id_list[i] = True
                        graph.add(gtsam.GenericProjectionFactorCal3_S2(
                            key_point, measurementNoise,
                            X(i), P(landmark_id), self.cal))

        # Set pose prior noise
        s = np.radians(60)
        poseNoiseSigmas = np.array([s, s, s, 1, 1, 1])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)

        # Initial estimate for poses
        for i, idx in enumerate(img_pose_id_list):
            pose_i = Pose3(
                wRc, Point3(0, self.delta_z*idx, 2))
            initialEstimate.insert(X(idx), pose_i)
            # Create priors for poses
            if(idx == 0 or idx ==1):
                graph.add(gtsam.PriorFactorPose3(X(idx), pose_i, posePriorNoise))
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
        # Declare an id for the figure
        fignum = 0
        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plt.cla()
        # Plot points
        gtsam_plot.plot_3d_points(fignum, result, 'rx')
        # Plot cameras
        i = 0
        while result.exists(X(i)):
            pose_i = result.atPose3(X(i))
            gtsam_plot.plot_pose3(fignum, pose_i, 1)
            i += 1
        # Draw
        axes.set_xlim3d(-20, 20)
        axes.set_ylim3d(-20, 20)
        axes.set_zlim3d(-20, 20)
        plt.legend()
        plt.show()


def run():
    fe = MappingFrontEnd()
    fe.get_all_image_features()
    fe.get_feature_matches()
    fe.initial_estimation()
    tic_ba = time.time()
    sfm_result, nrCamera, nrPoint = fe.bundle_adjustment()
    toc_ba = time.time()
    print(sfm_result)
    print('BA spents ', toc_ba-tic_ba, 's')
    fe.plot_sfm_result(sfm_result)


if __name__ == "__main__":
    run()
