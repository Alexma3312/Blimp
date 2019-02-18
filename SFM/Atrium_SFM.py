"""
Structure from motion based on GTSAM
"""
import math
import unittest

import gtsam
import numpy as np
from gtsam import Point2, Point3, Pose3, symbol

from SFM import SFMdata

def X(i):
    """Create key for pose i."""
    return gtsam.symbol(ord('x'), i)


def P(j):
    """Create key for landmark j."""
    return gtsam.symbol(ord('p'), j)


class AtriumSFM(object):

    def __init__(self, nrCameras, nrPoints, fov_in_degrees, image_width, image_height):
        """
        Parameters:
            nrCameras -- Number of cameras
            nrPoints -- Number of landmarks 
            fov_in_degrees -- Horizontal FOV = 128, Vertical FOV = 91, Diagonal FOV = 160
            image width and height -- camera output image [640,480], Superpoint output image [160,120]
        """
        self.nrCameras = nrCameras
        self.nrPoints = nrPoints
        self.calibration = gtsam.Cal3_S2(fov_in_degrees, image_width, image_height)
        self.truth = []

    def back_project(self, feature_point, calibration, depth):
        """back-project to 3D point at given depth, in camera coordinates."""
        pn = self.calibration.calibrate(feature_point)  # normalized
        return gtsam.Point3(depth, depth*(pn.x()), 1.5-(pn.y())*depth)

    def Atrium_SFM(self, data, rot_angle, y_distance):
        """
        Parameters:
            data -- a Data Object, input feature point data from the SFMdata.py file 
            rot_angle -- degree, camera rotation angle in the x-z camera coordinate
            y_distance -- distances between two continuous poses along the y axis 
        Returns:
            result -- GTSAM optimization result

        """
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
        s = np.radians(30)
        poseNoiseSigmas = np.array([s, s, s, 5, 5, 5])
        posePriorNoise = gtsam.noiseModel_Diagonal.Sigmas(poseNoiseSigmas)

        for i, y in enumerate([-1, 0, 1]):
            theta = np.radians(-y*rot_angle)
            # wRc = gtsam.Rot3(np.array([[0, math.cos(
            #     theta), -math.sin(theta)], [0, -math.sin(theta), -math.cos(theta)], [1, 0, 0]]).T)
            wRc = gtsam.Rot3(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T)
            wTi = gtsam.Pose3(wRc, gtsam.Point3(0, (y+1)*y_distance, 1))
            # graph.add(gtsam.PriorFactorPose3(X(i),
            #                                     wTi, posePriorNoise))
            initialEstimate.insert(X(i), wTi)

        # Add prior for two poses
        graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(
            wRc, gtsam.Point3(0, 0, 1.5)), posePriorNoise))
        graph.add(gtsam.PriorFactorPose3(X(2), gtsam.Pose3(
            wRc, gtsam.Point3(0, 5, 1.5)), posePriorNoise))

        # Add prior for the first point.
        # pointNoiseSigma = 5
        # pointPriorNoise = gtsam.noiseModel_Isotropic.Sigma(3, pointNoiseSigma)
        # graph.add(gtsam.PriorFactorPoint3(P(0),
        #                         Point3(10.0, 0.0, 0.0), pointPriorNoise))

        # Add initial estimates for Points.
        for j in range(self.nrPoints):
            point_j = self.back_project(
                data.Z[0][j], self.calibration, 10)
            initialEstimate.insert(P(j), point_j)

        # Optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()

        # Optimize with limited iterations
        # for i in range(5):
        #     optimizer.iterate()
        # result = optimizer.values()

        # Marginalization
        marginals = gtsam.Marginals(graph, result)
        marginals.marginalCovariance(X(0))

        return result


if __name__ == '__main__':
    # Initial the number of landmark points and cameras 
    nrCameras = 3
    nrPoints = 5

    AtriumSFM = AtriumSFM(nrCameras, nrPoints, 128, 640, 480)
    # Generate Structure from Motion input data
    data = SFMdata.Data(nrCameras, nrPoints)
    data.generate_data(2)

    # Generate Structure from Motion
    result = AtriumSFM.Atrium_SFM(data, 0, 2.5)
    print(result)
