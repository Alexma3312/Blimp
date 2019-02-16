"""This is the Trajectory Estimator class."""
import numpy as np
import torch
import cv2
import gtsam
from gtsam import Pose3, Point3
import math
from SuperPointPretrainedNetwork import demo_superpoint
import time


def X(i):
    """Create key for pose i."""
    return gtsam.symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return gtsam.symbol(ord('p'), j)

# def read_images():
#     image_1 = cv2.imread('dataset/wall_data/raw_frame_left.jpg',0)
#     image_2 = cv2.imread('dataset/wall_data/raw_frame_middle.jpg')
#     image_3 = cv2.imread('dataset/wall_data/raw_frame_right.jpg')
#     image_size = [160,120]
#     interp = cv2.INTER_AREA
#     image_4 = cv2.resize(image_1, (image_size[1], image_size[0]), interpolation=interp)
#     image_4 = (image_4.astype('float32') / 255.)
#     image_4 = video_streamer.read_image('dataset/wall_data/raw_frame_left.jpg', 160)
#     return image_1, image_2, image_3, image_4

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
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim


class TrajectoryEstimator(object):

    def __init__(self):
        """
        Args:
            self.estimate_trajectory -- (nrstate,6) numpy array of Pose3 values
            self.atrium_map -- (nrpoints,3) numpy array of gtsam.point3 values
        """
        self.image_size = [160,120]
        self.image_1 = read_image('dataset/wall_data/raw_frame_left.jpg', self.image_size)
        self.image_2 = read_image('dataset/wall_data/raw_frame_middle.jpg', self.image_size)
        self.image_3 = read_image('dataset/wall_data/raw_frame_right.jpg', self.image_size)

        self.estimate_trajectory = []
        self.atrium_map = [Point3(0, 10.0, 10.0),
                           Point3(-10.0, 10.0, 10.0),
                           Point3(-10.0, -10.0, 10.0),
                           Point3(10.0, -10.0, 10.0),
                           Point3(10.0, 10.0, -10.0),
                           ]

    def back_project(self, feature_point, calibration, depth):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, depth*pn.x(), 1.5-pn.y()*depth)

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Return:
            superpoint_feature - N*2 numpy array (u,v)
        """

        # Refer to /SuperPointPretrainedNetwork/demo_superpoint for more information about the parameters
        fe = demo_superpoint.SuperPointFrontend(weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
                          nms_dist=4,
                          conf_thresh=0.015,
                          nn_thresh=0.7,
                          cuda=False)
        superpoints, descriptors, _ = fe.run(self.image_1)
        
        return superpoints, descriptors

    def feature_extract(self, superpoints, descriptors):

        input_features = 0
        return input_features

    def feature_matching(self):
        return

    def process_first_image(self, input_features):
        return

    def process_next_image(self, input_features):
        return

    def initial_iSAM(self):
        
        # Define the camera calibration parameters
        fov_in_degrees, w, h = 128, 160, 120
        self.calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)

        # Define the camera observation noise model
        self.measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
        2, 1.0)  # one pixel in u and v

        # Create an iSAM2 object.
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)
        self.isam = gtsam.ISAM2(parameters)

        # Create a Factor Graph and Values to hold the new data
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        for i, point in enumerate(self.atrium_map):
            pointNoiseSigmas = np.array([0.1, 0.1, 0.1])
            pointPriorNoise = gtsam.noiseModel_Diagonal.Sigmas(pointNoiseSigmas)
            self.graph.add(gtsam.PriorFactorPoint3(P(i),
                                    point, pointPriorNoise))
            # Add initial estimates for Points.
        
        self.angle = 0
        theta = np.radians(-y*angle)
        self.wRc = gtsam.Rot3(np.array([[0, math.cos(
            theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)
        wTi = gtsam.Pose3(wRc, gtsam.Point3(0, 0, 1.5))

        pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array(
                [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        self.graph.push_back(gtsam.PriorFactorPose3(X(0), wTi, pose_noise))
        
        #Update the estimate_trajectory
        self.estimate_trajectory.append(wTi)


    def update_iSAM(self, input_features):

        
        # Add factors for each landmark observation
        for j, feature in enumerate(input_features):
            i = len(self.estimate_trajectory)
            self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                feature, self.measurement_noise, X(i), P(j), self.calibration))

            point = self.back_project(feature, self.calibration, 10)
        
        # Add an initial guess for the current pose
        # Intentionally initialize the variables off from the ground truth
        initial_estimate.insert(X(i), pose.compose(gtsam.Pose3(
            gtsam.Rot3.Rodrigues(-0.1, 0.2, 0.25), gtsam.Point3(0.05, -0.10, 0.20))))

        s = np.radians(30)
        poseNoiseSigmas = np.array([s, s, s, 5, 5, 5])
        # poseNoiseSigmas = np.array([0.3, 0.3, 0.3, 5, 5, 5])
        pose_noise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        graph.push_back(gtsam.PriorFactorPose3(X(0), poses[0], 0))

        # Update iSAM with the new factors
        isam.update(graph, initial_estimate)
        # Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
        # If accuracy is desired at the expense of time, update(*) can be called additional
        # times to perform multiple optimizer iterations every step.
        isam.update()
        current_estimate = isam.calculateEstimate()

        return

    def trajectory_estimator(self, image):
        """Trajectory Estimator is a function based on iSAM to generate the estimate trajectory and the estimate state

        Args:
            image -- (H,W) numpy array of intensities, H is image height and W is image width

        Returns:
            estimate_trajectory -- (nrstate+1,6) numpy array of Pose3 values
            estimate_state -- (1,12) numpy array of [x,y,z,row,pitch,yaw,dx,dy,dz,d(row),d(pitch),d(yaw)]
        """

        estimate_trajectory = []
        current_estimate_state = []
        return estimate_trajectory, current_estimate_state


if __name__ == "__main__":
    Trajectory_Estimator = TrajectoryEstimator()
    superpoints_1, descriptors_1 = Trajectory_Estimator.superpoint_generator(
        Trajectory_Estimator.image_1)
    Trajectory_Estimator.initial_iSAM()

