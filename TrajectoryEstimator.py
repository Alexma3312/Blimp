"""This is the Trajectory Estimator class."""
import numpy as np
import gtsam


class TrajectoryEstimator(object):

    def __init__(self):
        """
        Args:
            self.estimate_trajectory -- (nrstate,6) numpy array of gtsam.Pose3 values
            self.atrium_map -- (nrpoints,3) numpy array of gtsam.point3 values
        """
        self.estimate_trajectory = []
        self.atrium_map = [gtsam.Point3(0, 10.0, 10.0),
                           gtsam.Point3(-10.0, 10.0, 10.0),
                           gtsam.Point3(-10.0, -10.0, 10.0),
                           gtsam.Point3(10.0, -10.0, 10.0),
                           gtsam.Point3(10.0, 10.0, -10.0),
                           ]

    def superpoint_feature(self, image):
        """Use superpoint to extract features in the image
        """
        superpoint_feature = 0
        return superpoint_feature

    def feature_extract(self, superpoint_feature):

        input_feature = 0
        return input_feature

    def process_first_image(self, input_feature):
        return

    def process_next_image(self, input_feature):
        return

    def atrium_iSAMExample(self):
        return

    
    def trajectory_estimator(self, image):
        """Trajectory Estimator is a function based on iSAM to generate the estimate trajectory and the estimate state

        Args:
            image -- (H,W) numpy array of intensities, H is image height and W is image width

        Returns:
            new_estimate_trajectory -- (nrstate+1,6) numpy array of gtsam.Pose3 values
            estimate_state -- (1,12) numpy array of [x,y,z,row,pitch,yaw,dx,dy,dz,d(row),d(pitch),d(yaw)]
        """

        new_estimate_trajectory = np.array([])
        current_estimate_state = np.array([])
        return new_estimate_trajectory, current_estimate_state
