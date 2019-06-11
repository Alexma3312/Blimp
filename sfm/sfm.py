"""
Mapping based on Structure from Motion (SfM) with GTSAM
"""
import sys

import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3

sys.path.append('../')


def X(i):
    """Create key for pose i."""
    return gtsam.symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return gtsam.symbol(ord('p'), j)


class SfM(object):
    """
    Structure from Motion with GTSAM
    """

    def __init__(self, nrCameras, nrPoints, fov_in_degrees, image_width, image_height):
        """
        Parameters:
            nrCameras -- Number of cameras
            nrPoints -- Number of landmarks 
            fov_in_degrees -- Based on the official document of Runcam Swift 2: horizontal FOV = 128, Vertical FOV = 91, Diagonal FOV = 160
            image_width, image_height -- camera output image [640,480]. Superpoint can extract features with downsampled images, therefore the output of Superpoint is flexible.
        """
        self.nrCameras = nrCameras
        self.nrPoints = nrPoints
        self.calibration = gtsam.Cal3_S2(
            fov_in_degrees, image_width, image_height)

    def back_project(self, feature_point, calibration, depth):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, depth*(pn.x()), 1.5-(pn.y())*depth)

    def bundle_adjustment(self, data, y_distance, rotation_error, translation_error):
        """ Use GTSAM to solve Structure from Motion bundle adjustment.
        Parameters:
            data -- a Data Object, input feature point data from the sfm_data.py file 
            y_distance -- distances between two continuous poses along the y axis 
            rotation_error - pose prior rotation error
            translation_error - pose prior translation error
        Returns:
            result -- GTSAM optimization result
        """
        # Initialize factor Graph
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        # Add factors for all measurements
        measurementNoiseSigma = 1.0
        measurementNoise = gtsam.noiseModel_Isotropic.Sigma(
            2, measurementNoiseSigma)
        for i in range(len(data.Z)):
            for k in range(len(data.Z[i])):
                j = data.J[i][k]
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    data.Z[i][j], measurementNoise,
                    X(i), P(j), self.calibration))

        # Create priors and initial estimate
        s = np.radians(60)
        poseNoiseSigmas = np.array([s, s, s, 10, 10, 10])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)

        for i, y in enumerate([0, y_distance, 2*y_distance]):
            # Do not consider camera rotation when creating Rot3
            wRc = gtsam.Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)

            # The approximate height measurement is 1.5
            wTi = gtsam.Pose3(wRc, gtsam.Point3(0, y, 1.5))
            initialEstimate.insert(X(i), wTi)

        # Add priors for two poses
        # Add the first pose prior with error to test similarity transform.
        graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(
            gtsam.Rot3(np.array([[0+rotation_error, 1+rotation_error, 0+rotation_error], [0+rotation_error, 0+rotation_error, -1+rotation_error], [1+rotation_error, 0+rotation_error, 0+rotation_error]]).T), gtsam.Point3(0+translation_error, 0+translation_error, 1.5+translation_error)), posePriorNoise))
        graph.add(gtsam.PriorFactorPose3(X(1), gtsam.Pose3(
            wRc, gtsam.Point3(0, y_distance, 1.5)), posePriorNoise))

        # Add initial estimates for Points.
        for j in range(self.nrPoints):
            # Generate initial estimates by back projecting key points extracted at the initial pose. The scale is set based on experience.
            point_j = self.back_project(
                data.Z[0][j], self.calibration, 15)
            initialEstimate.insert(P(j), point_j)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()

        # Marginalization
        marginals = gtsam.Marginals(graph, result)

        return result
