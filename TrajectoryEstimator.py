"""This is the Trajectory Estimator class."""
import numpy as np

import cv2
import gtsam
from gtsam import Pose3, Point3
# from SuperPointPretrainedNetwork import demo_superpoint


def read_images():
    image_1 = cv2.imread('dataset/wall_data/raw_frame_left.jpg')
    image_2 = cv2.imread('dataset/wall_data/raw_frame_middle.jpg')
    image_3 = cv2.imread('dataset/wall_data/raw_frame_right.jpg')
    return image_1, image_2, image_3


class TrajectoryEstimator(object):

    def __init__(self):
        """
        Args:
            self.estimate_trajectory -- (nrstate,6) numpy array of Pose3 values
            self.atrium_map -- (nrpoints,3) numpy array of gtsam.point3 values
        """
        self.image_1, self.image_2, self.image_3 = read_images()

        self.estimate_trajectory = []
        self.atrium_map = [Point3(0, 10.0, 10.0),
                           Point3(-10.0, 10.0, 10.0),
                           Point3(-10.0, -10.0, 10.0),
                           Point3(10.0, -10.0, 10.0),
                           Point3(10.0, 10.0, -10.0),
                           ]

    def superpoint_generator(self, image):
        """Use superpoint to extract features in the image
        Return:
            superpoint_feature - N*2 numpy array (u,v)
        """

        # Refer to /SuperPointPretrainedNetwork/demo_superpoint for more information about the parameters
        # fe = SuperPointFrontend(weights_path='/SuperPointPretrainedNetwork/superpoint_v1.pth',
        #                   nms_dist=4,
        #                   conf_thresh=0.015,
        #                   nn_thresh=0.7,
        #                   cuda=False)
        # start1 = time.time()
        # superpoints, descriptors, _ = fe.run(image)
        # end1 = time.time()
        superpoints = [Point3(0,0,0)]
        descriptors = [Point3(0,0,0)]
        return superpoints, descriptors

    def feature_extract(self, superpoints, descriptors):

        input_features = 0
        return input_features

    def process_first_image(self, input_features):
        return

    def process_next_image(self, input_features):
        return

    def initial_iSAM(self):
        
        # Define the camera calibration parameters
        fov_in_degrees, w, h = 128, 160, 120
        self.calibration = gtsam.Cal3_S2(fov_in_degrees, w, h)

        # Define the camera observation noise model
        measurement_noise = gtsam.noiseModel_Isotropic.Sigma(
        2, 1.0)  # one pixel in u and v

        # Create an iSAM2 object.
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)
        isam = gtsam.ISAM2(parameters)

        # Create a Factor Graph and Values to hold the new data
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array(
                [0.0, 0.3, 0.3, 0.1, 0.1, 0.1]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        graph.push_back(gtsam.PriorFactorPose3(X(0), poses[0], 0))

        for i, point in enumerate(self.atrium_map):


    def update_iSAM(self, input_features):

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

